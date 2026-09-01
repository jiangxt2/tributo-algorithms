"""Framework-native distributed GraphSAGE implementation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ray.train import ScalingConfig

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import FrameworkNativeAlgorithm
from tributo.util.annotations import PublicAPI


class _EvidenceCollector:
    """Collect exactly one bounded record from each Ray Train worker."""

    def __init__(self) -> None:
        self._records: dict[int, dict[str, object]] = {}

    def record(self, value: dict[str, object]) -> None:
        rank = value.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("graph evidence rank must be an integer")
        if rank in self._records:
            raise ValueError(f"duplicate graph worker evidence rank: {rank}")
        self._records[rank] = dict(value)

    def snapshot(self) -> list[dict[str, object]]:
        return [self._records[rank] for rank in sorted(self._records)]


def _build_model(
    input_features: int,
    hidden_features: int,
    num_classes: int,
    *,
    model_kind: str = "graphsage",
    num_relations: int = 1,
) -> object:
    import torch

    if model_kind == "rgcn":
        from torch_geometric.nn import RGCNConv

        class RGCN(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = RGCNConv(input_features, hidden_features, num_relations)
                self.conv2 = RGCNConv(hidden_features, num_classes, num_relations)

            def forward(
                self,
                x: object,
                edge_index: object,
                edge_type: object,
            ) -> object:
                values = self.conv1(
                    cast(Any, x), cast(Any, edge_index), cast(Any, edge_type)
                ).relu()
                return self.conv2(values, cast(Any, edge_index), cast(Any, edge_type))

        return RGCN()
    if model_kind != "graphsage":
        raise ValueError(f"unsupported graph model kind: {model_kind}")
    from torch_geometric.nn import SAGEConv

    class GraphSAGE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = SAGEConv(input_features, hidden_features)
            self.conv2 = SAGEConv(hidden_features, num_classes)

        def forward(self, x: object, edge_index: object) -> object:
            values = self.conv1(cast(Any, x), cast(Any, edge_index)).relu()
            return self.conv2(values, cast(Any, edge_index))

    return GraphSAGE()


def _node_lookup_model(model: object, logits: object) -> object:
    """Preserve trained weights and expose bounded transductive lookup."""
    import torch

    trained = cast(torch.nn.Module, model)

    class NodeLookupModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = cast(Any, trained).conv1
            self.conv2 = cast(Any, trained).conv2
            self.register_buffer(
                "inference_logits",
                cast(torch.Tensor, logits).detach().to(dtype=torch.float32),
                persistent=False,
            )

        def forward(self, node_id: object) -> object:
            values = cast(torch.Tensor, node_id).long()
            stored = cast(torch.Tensor, self.inference_logits)
            valid = values.ge(0).logical_and(values.lt(stored.shape[0]))
            safe_values = values.clamp(min=0, max=stored.shape[0] - 1)
            selected = stored[safe_values]
            logits = torch.where(
                valid.unsqueeze(-1), selected, torch.zeros_like(selected)
            )
            return torch.cat(
                (logits, valid.unsqueeze(-1).to(dtype=logits.dtype)), dim=1
            )

    return NodeLookupModel()


def _scaling_config(plan: ResolvedAlgorithmPlan) -> "ScalingConfig":
    """Build the public Ray Train resource and placement contract."""
    from ray.train import ScalingConfig

    return ScalingConfig(
        num_workers=plan.runtime.worker_count,
        use_gpu=plan.runtime.num_gpus > 0,
        resources_per_worker={
            "CPU": plan.runtime.num_cpus,
            "GPU": plan.runtime.num_gpus,
            **dict(plan.runtime.custom_resources),
        },
        placement_strategy="SPREAD",
    )


def _columns(iterator: object, names: tuple[str, ...]) -> dict[str, list[object]]:
    iter_batches = getattr(iterator, "iter_batches", None)
    if not callable(iter_batches):
        raise AlgorithmExecutionError("graph dataset shard is not iterable")
    output: dict[str, list[object]] = {name: [] for name in names}
    for batch in iter_batches(batch_format="numpy"):
        if not isinstance(batch, Mapping):
            raise AlgorithmExecutionError("graph dataset produced a non-columnar batch")
        for name in names:
            if name not in batch:
                raise AlgorithmExecutionError(f"graph dataset is missing {name!r}")
            value = batch[name]
            converted = value.tolist() if hasattr(value, "tolist") else value
            if not isinstance(converted, (list, tuple)):
                raise AlgorithmExecutionError(
                    f"graph column {name!r} is not a bounded sequence"
                )
            output[name].extend(converted)
    return output


def _state_digest(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = cast(Any, state[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(value: object) -> str:
    """Return a stable digest for one derived tensor without persisting its rows."""
    tensor = cast(Any, value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _graph_identity_digest(
    *,
    topology_digest: str,
    node_feature_digest: str,
    node_feature_names: tuple[str, ...],
) -> str:
    """Bind transductive node IDs to topology and feature values."""
    payload = {
        "schema_version": 1,
        "topology_digest": topology_digest,
        "node_feature_digest": node_feature_digest,
        "node_feature_names": list(node_feature_names),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _graph_source_fingerprint(
    *,
    model_digest: str,
    graph_identity_digest: str,
    inference_logits_digest: str,
) -> str:
    """Identify every state component embedded in the exported lookup model."""
    payload = {
        "schema_version": 1,
        "model_digest": model_digest,
        "graph_identity_digest": graph_identity_digest,
        "inference_logits_digest": inference_logits_digest,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AlgorithmExecutionError(f"graph checkpoint {name} is invalid")
    return value


def _graph_train_loop(config: dict[str, Any]) -> None:
    import ray
    import torch
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import get_device, prepare_model

    names = cast(dict[str, Any], config["columns"])
    nodes = _columns(
        train.get_dataset_shard("nodes"),
        (names["node_id"], *names["node_features"]),
    )
    edge_columns = [names["edge_source"], names["edge_destination"]]
    edge_relation_name = names.get("edge_relation")
    if edge_relation_name is not None:
        edge_columns.append(edge_relation_name)
    edges = _columns(train.get_dataset_shard("edges"), tuple(edge_columns))
    seeds = _columns(
        train.get_dataset_shard("train"),
        (names["seed_node_id"], names["seed_label"]),
    )
    raw_node_ids = [int(cast(Any, value)) for value in nodes[names["node_id"]]]
    node_order = sorted(range(len(raw_node_ids)), key=raw_node_ids.__getitem__)
    node_ids = [raw_node_ids[index] for index in node_order]
    if node_ids != list(range(len(node_ids))):
        raise AlgorithmExecutionError(
            "graph training requires contiguous node_id values ordered from zero"
        )
    if not seeds[names["seed_node_id"]]:
        raise AlgorithmExecutionError("every graph worker requires labeled seeds")

    device = get_device()
    x = torch.tensor(
        [
            [
                float(cast(Any, nodes[feature][source_index]))
                for feature in names["node_features"]
            ]
            for source_index in node_order
        ],
        dtype=torch.float32,
        device=device,
    )
    raw_source = [int(cast(Any, value)) for value in edges[names["edge_source"]]]
    raw_destination = [
        int(cast(Any, value)) for value in edges[names["edge_destination"]]
    ]
    relations: list[int] = []
    if edge_relation_name is not None:
        edge_records = sorted(
            zip(
                raw_source,
                raw_destination,
                (int(cast(Any, value)) for value in edges[edge_relation_name]),
                strict=True,
            )
        )
        edge_pairs = [(left, right) for left, right, _ in edge_records]
        relations = [relation for _, _, relation in edge_records]
    else:
        edge_pairs = sorted(zip(raw_source, raw_destination, strict=True))
    source = [left for left, _ in edge_pairs]
    destination = [right for _, right in edge_pairs]
    if not source or len(source) != len(destination):
        raise AlgorithmExecutionError("GraphSAGE requires a non-empty edge list")
    model_kind = str(config.get("model_kind", "graphsage"))
    if edge_relation_name is not None:
        edge_index_values = [source, destination]
    else:
        edge_index_values = [source + destination, destination + source]
    edge_index = torch.tensor(edge_index_values, dtype=torch.long, device=device)
    edge_type = (
        torch.tensor(relations, dtype=torch.long, device=device) if relations else None
    )
    seed_index = torch.tensor(
        [int(cast(Any, value)) for value in seeds[names["seed_node_id"]]],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor(
        [int(cast(Any, value)) for value in seeds[names["seed_label"]]],
        dtype=torch.long,
        device=device,
    )

    model_config = cast(dict[str, Any], config["model"])
    model = prepare_model(
        cast(
            torch.nn.Module,
            _build_model(
                len(names["node_features"]),
                int(model_config["hidden_features"]),
                int(model_config["num_classes"]),
                model_kind=model_kind,
                num_relations=int(model_config.get("num_relations", 1)),
            ),
        )
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    epochs = int(config["training"]["epochs"])
    loss_value = 0.0
    accuracy = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = (
            model(x, edge_index, edge_type)
            if edge_type is not None
            else model(x, edge_index)
        )
        selected = logits[seed_index]
        loss = torch.nn.functional.cross_entropy(selected, labels)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        accuracy = float((selected.argmax(dim=1) == labels).float().mean().cpu())

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    unwrapped = cast(
        torch.nn.Module,
        model.module if hasattr(model, "module") else model,
    )
    unwrapped.eval()
    with torch.no_grad():
        inference_logits = (
            (
                unwrapped(x, edge_index, edge_type)
                if edge_type is not None
                else unwrapped(x, edge_index)
            )
            .detach()
            .cpu()
        )
    state = cast(Mapping[str, object], unwrapped.state_dict())
    model_digest = _state_digest(state)
    topology_digest = hashlib.sha256(
        json.dumps(
            {
                "nodes": node_ids,
                "source": source,
                "destination": destination,
                "relations": relations,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    node_feature_digest = _tensor_digest(x)
    graph_identity_digest = _graph_identity_digest(
        topology_digest=topology_digest,
        node_feature_digest=node_feature_digest,
        node_feature_names=tuple(str(value) for value in names["node_features"]),
    )
    inference_logits_digest = _tensor_digest(inference_logits)

    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    runtime = ray.get_runtime_context()
    assigned = runtime.get_assigned_resources()
    evidence = {
        "worker_id": str(runtime.get_worker_id()),
        "node_id": str(runtime.get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "shard_id": hashlib.sha256(
            f"{config['binding_digest']}:{rank}/{world_size}".encode("ascii")
        ).hexdigest(),
        "rows_processed": len(labels),
        "input_rows": {
            "train": len(labels),
            "nodes": len(node_ids),
            "edges": len(source),
        },
        "batch_count": epochs,
        "collective_steps": epochs,
        "model_state_digest": model_digest,
        "topology_digest": topology_digest,
        "node_feature_digest": node_feature_digest,
        "graph_identity_digest": graph_identity_digest,
        "inference_logits_digest": inference_logits_digest,
        "topology_kind": "relational" if relations else "homogeneous",
        "resources": {
            "num_cpus": float(assigned.get("CPU", 0.0)),
            "num_gpus": float(assigned.get("GPU", 0.0)),
            "custom": {
                str(name): float(value)
                for name, value in assigned.items()
                if name not in {"CPU", "GPU", "memory", "object_store_memory"}
            },
        },
    }
    ray.get(config["evidence_actor"].record.remote(evidence))

    checkpoint = None
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix="tributo-graphsage-") as directory:
            root = Path(directory)
            torch.save(dict(state), root / "model.pt")
            torch.save(inference_logits, root / "inference_logits.pt")
            (root / "model_config.json").write_text(
                json.dumps(
                    {
                        "input_features": len(names["node_features"]),
                        "hidden_features": int(model_config["hidden_features"]),
                        "num_classes": int(model_config["num_classes"]),
                        "node_features": list(names["node_features"]),
                        "model_kind": model_kind,
                        "num_relations": int(model_config.get("num_relations", 1)),
                        "graph_identity_version": 1,
                        "topology_digest": topology_digest,
                        "node_feature_digest": node_feature_digest,
                        "graph_identity_digest": graph_identity_digest,
                        "inference_logits_digest": inference_logits_digest,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            checkpoint = Checkpoint.from_directory(root)
            train.report(
                {"train_loss": loss_value, "accuracy": accuracy},
                checkpoint=checkpoint,
            )
    else:
        train.report({"train_loss": loss_value, "accuracy": accuracy})


@PublicAPI(stability="alpha")
class DistributedGraphSAGE(FrameworkNativeAlgorithm):
    """Train homogeneous full-neighborhood GraphSAGE with Ray Train DDP."""

    model_kind = "graphsage"

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        self._collector: Any | None = None
        if plan.runtime.resume_from is not None:
            raise AlgorithmConfigurationError(
                f"{self.model_kind} checkpoint resume is not supported by version 0.1"
            )

    def validate_environment(self) -> None:
        try:
            import torch
            import torch_geometric
            from ray.train.torch import TorchTrainer
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "graph training requires Ray Train, torch, and torch-geometric"
            ) from exc
        if (
            not torch.__version__
            or not torch_geometric.__version__
            or TorchTrainer is None
        ):
            raise AlgorithmConfigurationError("graph training environment is invalid")

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        if set(datasets) != {"edges", "nodes", "train"}:
            raise AlgorithmConfigurationError(
                "graph training requires nodes, edges, and train datasets"
            )
        return dict(datasets)

    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        import ray
        from ray.train import DataConfig, RunConfig
        from ray.train.torch import TorchTrainer

        model_config = dict(cast(Mapping[str, Any], config.get("model", {})))
        training_config = dict(cast(Mapping[str, Any], config.get("training", {})))
        ray_config = dict(cast(Mapping[str, Any], config.get("ray", {})))
        model_config.setdefault("hidden_features", 16)
        model_config.setdefault("num_classes", 2)
        if self.model_kind == "rgcn":
            model_config.setdefault("num_relations", 2)
        training_config.setdefault("epochs", 2)
        training_config.setdefault("learning_rate", 0.01)
        if any(
            int(model_config[name]) < 1 for name in ("hidden_features", "num_classes")
        ):
            raise AlgorithmConfigurationError("graph model dimensions must be positive")
        if self.model_kind == "rgcn" and int(model_config["num_relations"]) < 1:
            raise AlgorithmConfigurationError("R-GCN num_relations must be positive")
        if (
            int(training_config["epochs"]) < 1
            or float(training_config["learning_rate"]) <= 0
        ):
            raise AlgorithmConfigurationError("graph training controls are invalid")

        bindings = self.plan.input_bindings
        nodes = bindings.get("nodes")
        edges = bindings.get("edges")
        train = bindings.get("train")
        if train.label_name is None:
            raise AlgorithmConfigurationError("graph train binding requires labels")
        if self.model_kind == "rgcn" and len(edges.feature_names) != 3:
            raise AlgorithmConfigurationError(
                "R-GCN edge binding requires source, destination, and relation"
            )
        collector_type = ray.remote(_EvidenceCollector).options(num_cpus=0)
        self._collector = collector_type.remote()
        return TorchTrainer(
            train_loop_per_worker=_graph_train_loop,
            train_loop_config={
                "binding_digest": self.plan.primary_input_descriptor.binding_digest,
                "columns": {
                    "node_id": nodes.feature_names[0],
                    "node_features": list(nodes.feature_names[1:]),
                    "edge_source": edges.feature_names[0],
                    "edge_destination": edges.feature_names[1],
                    **(
                        {"edge_relation": edges.feature_names[2]}
                        if self.model_kind == "rgcn"
                        else {}
                    ),
                    "seed_node_id": train.feature_names[0],
                    "seed_label": train.label_name,
                },
                "evidence_actor": self._collector,
                "model_kind": self.model_kind,
                "model": model_config,
                "training": training_config,
            },
            scaling_config=_scaling_config(self.plan),
            datasets=cast(dict[str, Any], dict(datasets)),
            dataset_config=DataConfig(datasets_to_split=["train"]),
            run_config=RunConfig(
                name=f"tributo-{self.model_kind}",
                storage_path=ray_config.get("storage_path"),
            ),
        )

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        import ray

        del result
        if self._collector is None:
            raise AlgorithmExecutionError("graph evidence collector is missing")
        workers = ray.get(self._collector.snapshot.remote())
        ray.kill(self._collector, no_restart=True)
        self._collector = None
        if len(workers) != self.plan.runtime.worker_count:
            raise AlgorithmExecutionError("graph training did not report every worker")
        model_digests = {str(item.get("model_state_digest")) for item in workers}
        topology_digests = {str(item.get("topology_digest")) for item in workers}
        node_feature_digests = {
            str(item.get("node_feature_digest")) for item in workers
        }
        graph_identity_digests = {
            str(item.get("graph_identity_digest")) for item in workers
        }
        inference_logits_digests = {
            str(item.get("inference_logits_digest")) for item in workers
        }
        topology_kinds = {str(item.get("topology_kind")) for item in workers}
        if (
            len(model_digests) != 1
            or len(topology_digests) != 1
            or len(node_feature_digests) != 1
            or len(graph_identity_digests) != 1
            or len(inference_logits_digests) != 1
            or len(topology_kinds) != 1
        ):
            raise AlgorithmExecutionError(
                "GraphSAGE workers did not share model and topology state"
            )
        return {
            "workers": workers,
            "state": {
                "coordination": "framework_native",
                "synchronized": True,
                "bounded": True,
                "global_model_digest": next(iter(model_digests)),
                "details": {
                    "framework": "pytorch-geometric",
                    "sampling": "full_neighborhood",
                    "topology_kind": next(iter(topology_kinds)),
                    "topology_digest": next(iter(topology_digests)),
                    "node_feature_digest": next(iter(node_feature_digests)),
                    "graph_identity_digest": next(iter(graph_identity_digests)),
                    "inference_logits_digest": next(iter(inference_logits_digests)),
                },
            },
            "input_complete": True,
        }

    def checkpoint_source(self, result: object) -> object:
        checkpoint = getattr(result, "checkpoint", None)
        if checkpoint is None:
            raise AlgorithmExecutionError("graph training result has no checkpoint")
        return checkpoint


@PublicAPI(stability="alpha")
class DistributedRGCN(DistributedGraphSAGE):
    """Train a relation-aware static graph through PyG RGCNConv and DDP."""

    model_kind = "rgcn"


@PublicAPI(stability="alpha")
def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> FrameworkNativeAlgorithm:
    del artifacts
    if implementation is DistributedGraphSAGE:
        return DistributedGraphSAGE(plan)
    if implementation is DistributedRGCN:
        return DistributedRGCN(plan)
    raise AlgorithmConfigurationError("graph implementation drifted")


@PublicAPI(stability="alpha")
def export_result(
    *,
    result: object,
    checkpoint: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    import importlib.metadata

    import torch
    import torch_geometric
    from tributo.exporting.models import (
        BundleOutputConfig,
        CheckpointField,
        ExportCheckpointV1,
        ExportSource,
        ExportTarget,
    )
    from tributo.exporting.service import BundleExportService

    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError("graph training requires output.bundle_uri")
    as_directory = getattr(checkpoint, "as_directory", None)
    if not callable(as_directory):
        raise AlgorithmExecutionError("graph checkpoint is not readable")
    with as_directory() as directory:
        root = Path(directory)
        model_config = json.loads(
            (root / "model_config.json").read_text(encoding="utf-8")
        )
        model = cast(
            torch.nn.Module,
            _build_model(
                int(model_config["input_features"]),
                int(model_config["hidden_features"]),
                int(model_config["num_classes"]),
                model_kind=str(model_config.get("model_kind", "graphsage")),
                num_relations=int(model_config.get("num_relations", 1)),
            ),
        )
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        inference_logits = torch.load(
            root / "inference_logits.pt",
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(inference_logits, torch.Tensor) or inference_logits.ndim != 2:
            raise AlgorithmExecutionError("graph inference logits are malformed")
        if inference_logits.shape[0] < 1:
            raise AlgorithmExecutionError("graph inference logits must not be empty")
        num_classes = int(model_config["num_classes"])
        if inference_logits.shape[1] != num_classes or not bool(
            torch.isfinite(inference_logits).all()
        ):
            raise AlgorithmExecutionError(
                "graph inference logits do not match the finite class contract"
            )
        graph_identity_version = model_config.get("graph_identity_version")
        if (
            not isinstance(graph_identity_version, int)
            or isinstance(graph_identity_version, bool)
            or graph_identity_version != 1
        ):
            raise AlgorithmExecutionError("graph identity version is unsupported")
        topology_digest = _require_sha256(
            model_config.get("topology_digest"), name="topology_digest"
        )
        node_feature_digest = _require_sha256(
            model_config.get("node_feature_digest"), name="node_feature_digest"
        )
        graph_identity_digest = _require_sha256(
            model_config.get("graph_identity_digest"), name="graph_identity_digest"
        )
        node_feature_names = model_config.get("node_features")
        if not isinstance(node_feature_names, list) or not all(
            isinstance(value, str) and value for value in node_feature_names
        ):
            raise AlgorithmExecutionError("graph checkpoint node features are invalid")
        computed_graph_identity = _graph_identity_digest(
            topology_digest=topology_digest,
            node_feature_digest=node_feature_digest,
            node_feature_names=tuple(node_feature_names),
        )
        if graph_identity_digest != computed_graph_identity:
            raise AlgorithmExecutionError("graph identity digest mismatched")
        expected_logits_digest = _require_sha256(
            model_config.get("inference_logits_digest"),
            name="inference_logits_digest",
        )
        actual_logits_digest = _tensor_digest(inference_logits)
        if expected_logits_digest != actual_logits_digest:
            raise AlgorithmExecutionError("graph inference logits digest mismatched")
        bundle_model = cast(
            torch.nn.Module,
            _node_lookup_model(model, inference_logits),
        )
        model_digest = _state_digest(cast(Mapping[str, object], model.state_dict()))
        fingerprint = _graph_source_fingerprint(
            model_digest=model_digest,
            graph_identity_digest=graph_identity_digest,
            inference_logits_digest=actual_logits_digest,
        )
        source = ExportSource(
            source_kind="torch_module",
            model_object=bundle_model,
            architecture_id=plan.resolution.implementation_id,
            model_config_data=model_config,
            feature_schema={
                "input_names": ["node_id"],
                "node_features": node_feature_names,
                "node_id_min": 0,
                "node_id_max_exclusive": int(inference_logits.shape[0]),
                "graph_identity_version": 1,
                "topology_digest": topology_digest,
                "node_feature_digest": node_feature_digest,
                "graph_identity_digest": graph_identity_digest,
                "inference_logits_digest": actual_logits_digest,
                "output_layout": {
                    "logits": [0, int(model_config["num_classes"])],
                    "valid": int(model_config["num_classes"]),
                },
            },
            sample_inputs={"node_id": torch.tensor([0, 1], dtype=torch.int64)},
            metadata={
                "framework": "pytorch-geometric",
                "framework_versions": {
                    "torch": torch.__version__,
                    "torch-geometric": torch_geometric.__version__,
                },
                "task_type": "node_classification",
                "sampling": "full_neighborhood",
                "graph_identity_digest": graph_identity_digest,
                "topology_kind": (
                    "relational"
                    if model_config.get("model_kind") == "rgcn"
                    else "homogeneous"
                ),
                "producer_distribution": "tributo-algorithms-graph-pyg",
            },
            source_fingerprint=fingerprint,
            checkpoint_contract=ExportCheckpointV1(
                trainer_type=str(model_config.get("model_kind", "graphsage")),
                architecture_id=plan.resolution.implementation_id,
                input_schema=(
                    CheckpointField(
                        name="node_id",
                        dtype="int64",
                        shape=("batch",),
                    ),
                ),
                output_schema=(
                    CheckpointField(
                        name="output",
                        dtype="float32",
                        shape=("batch", int(model_config["num_classes"]) + 1),
                    ),
                ),
                preprocessing={
                    "type": "transductive_node_lookup",
                    "invalid_id_policy": "zero_output_with_valid_false",
                    "node_count": int(inference_logits.shape[0]),
                    "graph_identity_version": 1,
                    "topology_digest": topology_digest,
                    "node_feature_digest": node_feature_digest,
                    "graph_identity_digest": graph_identity_digest,
                    "inference_logits_digest": actual_logits_digest,
                },
                task_type="transductive_node_classification",
                framework="pytorch-geometric",
                framework_version=torch_geometric.__version__,
            ),
        )
        published = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=cast(str, output["bundle_uri"]),
                request_id=run_id,
                run_id=run_id,
                targets=[
                    ExportTarget(
                        name="graph-model",
                        format="safetensors",
                        exporter_id="torch-safetensors-v1",
                    ),
                    ExportTarget(
                        name="graph-inference",
                        format="onnx",
                        exporter_id="torch-onnx-v1",
                        options={"dynamo": False},
                    ),
                ],
                roles={
                    "model": "graph-model",
                    "inference": "graph-inference",
                },
            ),
            tributo_version=importlib.metadata.version("tributo"),
        )
    metrics = getattr(result, "metrics", {})
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics={
            key: value
            for key, value in (metrics.items() if isinstance(metrics, Mapping) else ())
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        outputs={
            "bundle_id": published.bundle_id,
            "bundle_uri": published.canonical_uri,
            "execution_id": published.execution_id,
            "manifest_sha256": published.manifest_sha256,
        },
    )


__all__ = [
    "DistributedGraphSAGE",
    "DistributedRGCN",
    "create_algorithm",
    "export_result",
]
