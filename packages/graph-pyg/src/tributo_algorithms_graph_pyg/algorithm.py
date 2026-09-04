"""RayTorchAdapter implementations for bounded PyG graph classifiers."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    TorchAccumulationWindow,
    TorchBackwardContext,
    TorchCheckpointPayloadDraft,
    TorchCheckpointRef,
    TorchLossContribution,
    TorchMetricContribution,
    TorchMetricPolicy,
    TorchMetricReductionContext,
    apply_torch_loss_backward,
    reduce_torch_metrics,
    report_torch_checkpoint,
)
from tributo.algorithms.spi import (
    RayTorchAdapter,
    TorchArtifactContext,
    TorchArtifactPlan,
    TorchCheckpointContext,
    TorchRuntimeContext,
    TorchStageContext,
    TorchWorkerCheckpointContext,
)
from tributo.util.annotations import PublicAPI


def _integer_values(values: object, field_name: str) -> list[int]:
    import numpy as np

    result: list[int] = []
    for raw_value in cast(Any, values):
        if isinstance(raw_value, (bool, np.bool_)):
            raise AlgorithmExecutionError(f"{field_name} must contain integer values")
        try:
            result.append(operator.index(raw_value))
        except TypeError as exc:
            raise AlgorithmExecutionError(
                f"{field_name} must contain integer values"
            ) from exc
    return result


def _gradient_clip_norm(training: Mapping[str, Any]) -> float:
    value = training.get("max_gradient_norm", 1.0)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise AlgorithmConfigurationError(
            "training.max_gradient_norm must be positive and finite"
        )
    return float(value)


def _build_model(
    input_features: int,
    hidden_features: int,
    num_classes: int,
    *,
    model_kind: str = "graphsage",
    num_relations: int = 1,
) -> object:
    import torch

    if min(input_features, hidden_features, num_classes, num_relations) < 1:
        raise ValueError("graph model dimensions must be positive")
    if model_kind == "rgcn":
        from torch_geometric.nn import RGCNConv

        class RGCN(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = RGCNConv(input_features, hidden_features, num_relations)
                self.conv2 = RGCNConv(hidden_features, num_classes, num_relations)

            def forward(
                self, x: object, edge_index: object, edge_type: object
            ) -> object:
                values = self.conv1(x, edge_index, edge_type).relu()
                return self.conv2(values, edge_index, edge_type)

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
            values = self.conv1(x, edge_index).relu()
            return self.conv2(values, edge_index)

    return GraphSAGE()


def _node_lookup_model(model: object, logits: object) -> object:
    import torch

    trained = cast(torch.nn.Module, model)

    class NodeLookupModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = cast(Any, trained).conv1
            self.conv2 = cast(Any, trained).conv2
            self.register_buffer(
                "inference_logits",
                cast(torch.Tensor, logits).detach().float(),
                persistent=False,
            )

        def forward(self, node_id: object) -> object:
            values = cast(torch.Tensor, node_id)
            if (
                values.dtype == torch.bool
                or values.is_floating_point()
                or values.is_complex()
            ):
                raise ValueError("graph node IDs must contain integer values")
            values = values.to(dtype=torch.long)
            stored = cast(torch.Tensor, self.inference_logits)
            valid = values.ge(0).logical_and(values.lt(stored.shape[0]))
            safe = values.clamp(min=0, max=stored.shape[0] - 1)
            selected = stored[safe]
            masked = torch.where(
                valid.unsqueeze(-1), selected, torch.zeros_like(selected)
            )
            return torch.cat((masked, valid.unsqueeze(-1).float()), dim=1)

    return NodeLookupModel()


def _columns(iterator: object, names: tuple[str, ...]) -> dict[str, list[object]]:
    method = getattr(iterator, "iter_batches", None)
    if not callable(method):
        raise AlgorithmExecutionError("graph dataset shard is not iterable")
    output: dict[str, list[object]] = {name: [] for name in names}
    for batch in method(batch_format="numpy"):
        if not isinstance(batch, Mapping):
            raise AlgorithmExecutionError("graph dataset produced a non-columnar batch")
        for name in names:
            if name not in batch:
                raise AlgorithmExecutionError(f"graph dataset is missing {name!r}")
            value = batch[name]
            values = value.tolist() if hasattr(value, "tolist") else value
            if not isinstance(values, (list, tuple)):
                raise AlgorithmExecutionError(
                    f"graph column {name!r} is not a sequence"
                )
            output[name].extend(values)
    return output


def _dataset_column_names(dataset: object) -> tuple[str, ...]:
    schema = getattr(dataset, "schema", None)
    if callable(schema):
        value = schema()
        names = getattr(value, "names", None)
        if isinstance(names, (list, tuple)) and all(
            isinstance(name, str) for name in names
        ):
            return tuple(names)
    return ()


def _state_digest(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = cast(Any, state[name]).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(value: object) -> str:
    tensor = cast(Any, value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise AlgorithmExecutionError(f"graph checkpoint {name} is invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AlgorithmExecutionError(f"graph checkpoint {name} is invalid") from exc
    return value


def _graph_identity_digest(
    *,
    topology_digest: str,
    node_feature_digest: str,
    node_feature_names: tuple[str, ...],
) -> str:
    payload = {
        "schema_version": 1,
        "topology_digest": topology_digest,
        "node_feature_digest": node_feature_digest,
        "node_feature_names": list(node_feature_names),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _graph_source_fingerprint(
    *, model_digest: str, graph_identity_digest: str, inference_logits_digest: str
) -> str:
    payload = {
        "schema_version": 1,
        "model_digest": model_digest,
        "graph_identity_digest": graph_identity_digest,
        "inference_logits_digest": inference_logits_digest,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _graph_train_loop(
    config: Mapping[str, Any], checkpoint_context: TorchWorkerCheckpointContext
) -> None:
    import ray
    import torch
    from ray import train
    from ray.train.torch import get_device, prepare_model

    columns = config.get("columns")
    names = (
        {str(key): value for key, value in columns.items()}
        if isinstance(columns, Mapping)
        else {}
    )
    nodes_dataset = train.get_dataset_shard("nodes")
    edges_dataset = train.get_dataset_shard("edges")
    train_dataset = train.get_dataset_shard("train")
    if not names:
        node_names = _dataset_column_names(nodes_dataset)
        edge_names = _dataset_column_names(edges_dataset)
        train_names = _dataset_column_names(train_dataset)
        if len(node_names) < 2 or len(edge_names) < 2 or len(train_names) < 2:
            raise AlgorithmConfigurationError(
                "graph adapter columns cannot be inferred"
            )
        names = {
            "node_id": node_names[0],
            "node_features": list(node_names[1:]),
            "edge_source": edge_names[0],
            "edge_destination": edge_names[1],
            "seed_node_id": train_names[0],
            "seed_label": train_names[-1],
        }
        if len(edge_names) >= 3:
            names["edge_relation"] = edge_names[2]
    node_features = tuple(str(value) for value in names["node_features"])
    nodes = _columns(nodes_dataset, (str(names["node_id"]), *node_features))
    edge_columns: list[str] = [
        str(names["edge_source"]),
        str(names["edge_destination"]),
    ]
    if names.get("edge_relation") is not None:
        edge_columns.append(str(names["edge_relation"]))
    edges = _columns(edges_dataset, tuple(edge_columns))
    seeds = _columns(
        train_dataset,
        (str(names["seed_node_id"]), str(names["seed_label"])),
    )
    raw_node_ids = _integer_values(nodes[str(names["node_id"])], "graph node IDs")
    order = sorted(range(len(raw_node_ids)), key=raw_node_ids.__getitem__)
    node_ids = [raw_node_ids[index] for index in order]
    if node_ids != list(range(len(node_ids))):
        raise AlgorithmExecutionError(
            "graph node IDs must be contiguous and zero-based"
        )
    if not seeds[str(names["seed_node_id"])]:
        raise AlgorithmExecutionError("every graph worker requires labeled seeds")
    device = get_device()
    x = torch.tensor(
        [
            [float(cast(Any, nodes[feature][index])) for feature in node_features]
            for index in order
        ],
        dtype=torch.float32,
        device=device,
    )
    source = _integer_values(edges[str(names["edge_source"])], "graph edge sources")
    destination = _integer_values(
        edges[str(names["edge_destination"])], "graph edge destinations"
    )
    relations = (
        _integer_values(edges[str(names["edge_relation"])], "graph edge relation IDs")
        if names.get("edge_relation") is not None
        else []
    )
    if not source or len(source) != len(destination):
        raise AlgorithmExecutionError("graph requires a non-empty edge list")
    if len(relations) not in {0, len(source)}:
        raise AlgorithmExecutionError("graph relation coverage is incomplete")
    if relations:
        edge_records = sorted(zip(source, destination, relations, strict=True))
        source = [left for left, _, _ in edge_records]
        destination = [right for _, right, _ in edge_records]
        relations = [relation for _, _, relation in edge_records]
    else:
        edge_pairs = sorted(zip(source, destination, strict=True))
        source = [left for left, _ in edge_pairs]
        destination = [right for _, right in edge_pairs]
    edge_index_values = (
        [source, destination]
        if relations
        else [source + destination, destination + source]
    )
    edge_index = torch.tensor(edge_index_values, dtype=torch.long, device=device)
    edge_type = (
        torch.tensor(relations, dtype=torch.long, device=device) if relations else None
    )
    seed_index = torch.tensor(
        _integer_values(seeds[str(names["seed_node_id"])], "graph seed node IDs"),
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor(
        _integer_values(seeds[str(names["seed_label"])], "graph labels"),
        dtype=torch.long,
        device=device,
    )
    model_config = config.get("model", {})
    training = config.get("training", {})
    if not isinstance(model_config, Mapping) or not isinstance(training, Mapping):
        raise AlgorithmConfigurationError("graph model/training config is invalid")
    node_count = len(node_ids)
    if any(
        value < 0 or value >= node_count
        for value in (*source, *destination, *seed_index.tolist())
    ):
        raise AlgorithmExecutionError("graph node IDs are out of range")
    num_classes = int(model_config.get("num_classes", 2))
    if any(value < 0 or value >= num_classes for value in labels.tolist()):
        raise AlgorithmExecutionError("graph labels are out of range")
    if relations:
        num_relations = int(model_config.get("num_relations", 1))
        if any(value < 0 or value >= num_relations for value in relations):
            raise AlgorithmExecutionError("graph relation IDs are out of range")
    model = prepare_model(
        cast(
            torch.nn.Module,
            _build_model(
                len(node_features),
                int(model_config.get("hidden_features", 16)),
                int(model_config.get("num_classes", 2)),
                model_kind=str(config.get("model_kind", "graphsage")),
                num_relations=int(model_config.get("num_relations", 1)),
            ),
        )
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 0.01))
    )
    max_gradient_norm = _gradient_clip_norm(training)
    context = train.get_context()
    rank, world_size = context.get_world_rank(), context.get_world_size()
    if checkpoint_context.source != "none" or checkpoint_context.checkpoint is not None:
        raise AlgorithmExecutionError("graph Stage does not accept an input Checkpoint")
    epochs = int(training.get("epochs", 2))
    loss_value = 0.0
    metric_rows = 0
    metric_correct = 0
    for epoch in range(epochs):
        logits = (
            model(x, edge_index, edge_type)
            if edge_type is not None
            else model(x, edge_index)
        )
        selected = logits[seed_index]
        numerator = torch.nn.functional.cross_entropy(selected, labels, reduction="sum")
        normalizer = int(labels.numel())

        def finalize(scale: float) -> None:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        def reduce_normalizer(value: float) -> float:
            tensor = torch.tensor(value, dtype=torch.float64, device=device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            return float(tensor.item())

        apply_torch_loss_backward(
            TorchLossContribution(numerator, normalizer),
            TorchAccumulationWindow(index=epoch, expected_micro_batches=1),
            TorchBackwardContext(
                world_size=world_size,
                backward=lambda value: value.backward(),
                reduce_normalizer=reduce_normalizer,
                finalize_window=finalize,
            ),
        )
        with torch.no_grad():
            loss_value = float(numerator.detach().item() / max(normalizer, 1))
            metric_rows = normalizer
            metric_correct = int((selected.argmax(dim=1) == labels).sum().item())

    def reduce_metric(
        name: str, contribution: TorchMetricContribution, reducer: str
    ) -> float:
        del name, reducer
        state = torch.tensor(
            [contribution.numerator, contribution.normalizer],
            dtype=torch.float64,
            device=device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(state, op=torch.distributed.ReduceOp.SUM)
        return float(state[0].item() / state[1].item()) if state[1].item() > 0 else 0.0

    reduced_metrics = reduce_torch_metrics(
        {
            "train_loss": TorchMetricContribution(
                loss_value * metric_rows, metric_rows
            ),
            "accuracy": TorchMetricContribution(metric_correct, metric_rows),
        },
        TorchMetricPolicy({"train_loss": "sum_count", "accuracy": "sum_count"}),
        TorchMetricReductionContext(reduce_metric),
    )
    model = cast(torch.nn.Module, model.module if hasattr(model, "module") else model)
    model.eval()
    with torch.no_grad():
        inference_logits = (
            (
                model(x, edge_index, edge_type)
                if edge_type is not None
                else model(x, edge_index)
            )
            .detach()
            .cpu()
        )
    model_digest = _state_digest(cast(Mapping[str, object], model.state_dict()))
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
        ).encode()
    ).hexdigest()
    node_feature_digest = _tensor_digest(x)
    graph_identity_digest = _graph_identity_digest(
        topology_digest=topology_digest,
        node_feature_digest=node_feature_digest,
        node_feature_names=node_features,
    )
    inference_logits_digest = _tensor_digest(inference_logits)
    assigned = ray.get_runtime_context().get_assigned_resources()
    worker = {
        "worker_id": str(ray.get_runtime_context().get_worker_id()),
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "shard_id": f"train-{rank}",
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
        },
    }
    workers: list[object] = [worker] * world_size
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_gather_object(workers, worker)
    descriptor_context = checkpoint_context.stage
    with tempfile.TemporaryDirectory(prefix="tributo-graph-checkpoint-") as directory:
        root = Path(directory)
        torch.save(model.state_dict(), root / "model.pt")
        torch.save(inference_logits, root / "inference_logits.pt")
        (root / "model_config.json").write_text(
            json.dumps(
                {
                    "input_features": len(node_features),
                    "hidden_features": int(model_config.get("hidden_features", 16)),
                    "num_classes": int(model_config.get("num_classes", 2)),
                    "node_features": list(node_features),
                    "model_kind": str(config.get("model_kind", "graphsage")),
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

        report_torch_checkpoint(
            {
                "train_loss": reduced_metrics.values["train_loss"],
                "accuracy": reduced_metrics.values["accuracy"],
                "execution_workers": workers,
                "model_state_digest": model_digest,
            },
            TorchCheckpointPayloadDraft(root),
            descriptor_context,
            epochs,
        )


@PublicAPI(stability="alpha")
class DistributedGraphSAGE(RayTorchAdapter):
    """Train homogeneous full-neighborhood GraphSAGE with Core-owned Ray Train."""

    model_kind = "graphsage"

    def validate_environment(self, context: TorchRuntimeContext) -> None:
        del context
        try:
            import torch
            import torch_geometric
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "GraphSAGE requires torch and torch-geometric"
            ) from exc
        if not torch.__version__ or not torch_geometric.__version__:
            raise AlgorithmConfigurationError("GraphSAGE environment is invalid")

    def bind_datasets(
        self, datasets: Mapping[str, object], context: TorchStageContext
    ) -> Mapping[str, object]:
        del context
        if set(datasets) != {"edges", "nodes", "train"}:
            raise AlgorithmConfigurationError(
                "graph training requires nodes, edges, and train datasets"
            )
        return dict(datasets)

    def worker_config(self, context: TorchStageContext) -> Mapping[str, object]:
        config = {
            key: value
            for key, value in context.runtime.algorithm_config.items()
            if key not in {"ray", "output"}
        }
        config.setdefault("model_kind", self.model_kind)
        bindings = context.runtime.input_bindings

        def binding_for(role: str) -> Mapping[str, object]:
            binding = bindings.get(role)
            if not isinstance(binding, Mapping):
                raise AlgorithmConfigurationError(
                    f"graph Stage context is missing the {role} binding"
                )
            return binding

        node_binding = binding_for("nodes")
        edge_binding = binding_for("edges")
        train_binding = binding_for("train")

        def feature_names(binding: Mapping[str, object]) -> tuple[object, ...]:
            values = binding.get("feature_names", ())
            return tuple(values) if isinstance(values, (list, tuple)) else ()

        node_features = feature_names(node_binding)
        edge_features = feature_names(edge_binding)
        train_features = feature_names(train_binding)
        label_name = train_binding.get("label_name")
        if (
            len(node_features) < 2
            or len(edge_features) < 2
            or len(train_features) != 1
            or not isinstance(label_name, str)
            or not label_name
        ):
            raise AlgorithmConfigurationError(
                "graph Stage input bindings have an invalid typed layout"
            )
        columns: dict[str, object] = {
            "node_id": str(node_features[0]),
            "node_features": [str(name) for name in node_features[1:]],
            "edge_source": str(edge_features[0]),
            "edge_destination": str(edge_features[1]),
            "seed_node_id": str(train_features[0]),
            "seed_label": label_name,
        }
        if len(edge_features) >= 3:
            columns["edge_relation"] = str(edge_features[2])
        config["columns"] = columns
        return config

    def train_loop_per_worker(
        self,
        worker_config: Mapping[str, object],
        checkpoint_context: TorchWorkerCheckpointContext,
    ) -> None:
        _graph_train_loop(worker_config, checkpoint_context)

    def checkpoint_source(
        self, result: object, context: TorchCheckpointContext
    ) -> object:
        del context
        checkpoint = getattr(result, "checkpoint", None)
        if checkpoint is None:
            raise AlgorithmExecutionError("graph training result has no checkpoint")
        return checkpoint

    def metric_plan(self, context: TorchRuntimeContext) -> Any:
        del context
        from tributo.algorithms.spi import TorchMetricPlan

        return TorchMetricPlan({"train_loss": "sum_count", "accuracy": "sum_count"})

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        config = context.stage.runtime.algorithm_config.get("model", {})
        classes = (
            int(config.get("num_classes", 2)) if isinstance(config, Mapping) else 2
        )
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {"name": "node_id", "dtype": "int64", "shape": ("batch",)},
            ),
            output_signature=(
                {"name": "output", "dtype": "float32", "shape": ("batch", classes + 1)},
            ),
            targets=(
                {
                    "name": "graph-model",
                    "format": "safetensors",
                    "exporter_id": "torch-safetensors-v1",
                },
                {
                    "name": "graph-inference",
                    "format": "onnx",
                    "exporter_id": "torch-onnx-v1",
                    "options": {"dynamo": False},
                },
            ),
            roles={"model": "graph-model", "inference": "graph-inference"},
        )

    @contextmanager
    def open_export_source(
        self, checkpoint_ref: TorchCheckpointRef, artifact_context: TorchArtifactContext
    ) -> Any:
        import torch
        import torch_geometric
        from tributo.exporting.models import (
            CheckpointField,
            ExportCheckpointV1,
            ExportSource,
        )

        checkpoint = checkpoint_ref.checkpoint
        opener = getattr(checkpoint, "as_directory", None)
        if not callable(opener):
            raise AlgorithmExecutionError("graph checkpoint cannot be opened")
        with opener() as directory:
            root = Path(directory)
            required_payloads = (
                "model_config.json",
                "model.pt",
                "inference_logits.pt",
            )
            missing_payloads = [
                name
                for name in required_payloads
                if not (root / name).is_file() or (root / name).is_symlink()
            ]
            if missing_payloads:
                raise AlgorithmExecutionError(
                    f"graph checkpoint is missing payloads: {missing_payloads}"
                )
            model_config = json.loads(
                (root / "model_config.json").read_text(encoding="utf-8")
            )
            model = cast(
                torch.nn.Module,
                _build_model(
                    int(model_config["input_features"]),
                    int(model_config["hidden_features"]),
                    int(model_config["num_classes"]),
                    model_kind=str(model_config.get("model_kind", self.model_kind)),
                    num_relations=int(model_config.get("num_relations", 1)),
                ),
            )
            try:
                model.load_state_dict(
                    torch.load(root / "model.pt", map_location="cpu", weights_only=True)
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise AlgorithmExecutionError(
                    "graph model payload is incompatible"
                ) from exc
            model.eval()
            inference_logits = torch.load(
                root / "inference_logits.pt", map_location="cpu", weights_only=True
            )
            if (
                not isinstance(inference_logits, torch.Tensor)
                or inference_logits.ndim != 2
                or inference_logits.shape[0] < 1
                or inference_logits.shape[1] != int(model_config["num_classes"])
                or not bool(torch.isfinite(inference_logits).all())
            ):
                raise AlgorithmExecutionError(
                    "graph inference logits do not match the finite class contract"
                )
            if model_config.get("graph_identity_version") != 1:
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
                raise AlgorithmExecutionError(
                    "graph checkpoint node features are invalid"
                )
            if graph_identity_digest != _graph_identity_digest(
                topology_digest=topology_digest,
                node_feature_digest=node_feature_digest,
                node_feature_names=tuple(node_feature_names),
            ):
                raise AlgorithmExecutionError("graph identity digest mismatched")
            expected_logits_digest = _require_sha256(
                model_config.get("inference_logits_digest"),
                name="inference_logits_digest",
            )
            actual_logits_digest = _tensor_digest(inference_logits)
            if expected_logits_digest != actual_logits_digest:
                raise AlgorithmExecutionError(
                    "graph inference logits digest mismatched"
                )
            model_digest = _state_digest(cast(Mapping[str, object], model.state_dict()))
            yield ExportSource(
                source_kind="torch_module",
                model_object=_node_lookup_model(model, inference_logits),
                architecture_id=artifact_context.stage.runtime.implementation_id,
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
                    "output_shape": ["batch", int(model_config["num_classes"]) + 1],
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
                    "topology_kind": "relational"
                    if self.model_kind == "rgcn"
                    else "homogeneous",
                    "producer_distribution": "tributo-algorithms-graph-pyg",
                },
                source_fingerprint=_graph_source_fingerprint(
                    model_digest=model_digest,
                    graph_identity_digest=graph_identity_digest,
                    inference_logits_digest=actual_logits_digest,
                ),
                checkpoint_contract=ExportCheckpointV1(
                    trainer_type=str(model_config.get("model_kind", self.model_kind)),
                    architecture_id=artifact_context.stage.runtime.implementation_id,
                    input_schema=(
                        CheckpointField(
                            name="node_id", dtype="int64", shape=("batch",)
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


@PublicAPI(stability="alpha")
class DistributedRGCN(DistributedGraphSAGE):
    """Train a relation-aware graph through PyG RGCNConv."""

    model_kind = "rgcn"


def create_algorithm(
    *,
    plan: object | None = None,
    implementation: object | None = None,
    artifacts: tuple[object, ...] = (),
) -> RayTorchAdapter:
    del plan, artifacts
    if implementation is DistributedRGCN:
        return DistributedRGCN()
    if implementation is DistributedGraphSAGE:
        return DistributedGraphSAGE()
    raise AlgorithmConfigurationError("graph implementation drifted")


__all__ = ["DistributedGraphSAGE", "DistributedRGCN", "create_algorithm"]
