"""Distributed jagged-history recommendation with explicit All-to-All routing."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import FrameworkNativeAlgorithm


def _model(
    *,
    user_count: int,
    item_count: int,
    embedding_dim: int,
) -> object:
    import torch

    class JaggedRankingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.user_embedding = torch.nn.Embedding(user_count, embedding_dim)
            self.history_embedding = torch.nn.EmbeddingBag(
                item_count,
                embedding_dim,
                mode="mean",
                include_last_offset=False,
            )
            self.candidate_embedding = torch.nn.Embedding(item_count, embedding_dim)
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(
            self,
            user_ids: object,
            history_values: object,
            history_offsets: object,
            candidate_ids: object,
        ) -> object:
            users = cast(Any, user_ids).long()
            values = cast(Any, history_values).long()
            offsets = cast(Any, history_offsets).long()
            candidates = cast(Any, candidate_ids).long()
            context = self.user_embedding(users) + self.history_embedding(
                values, offsets
            )
            candidate = self.candidate_embedding(candidates)
            return (context * candidate).sum(dim=1, keepdim=True) + self.bias

    return JaggedRankingModel()


def _state_digest(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = cast(Any, state[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _jagged_tensors(
    frame: object,
    *,
    user_col: str,
    history_col: str,
    candidate_col: str,
    label_col: str,
    item_count: int,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any, int]:
    import numpy as np
    import torch

    data = cast(Any, frame)
    histories: list[list[int]] = []
    for raw in data[history_col].tolist():
        if isinstance(raw, np.ndarray):
            values = raw.tolist()
        elif isinstance(raw, (list, tuple)):
            values = list(raw)
        else:
            raise AlgorithmExecutionError("jagged history values must be sequences")
        history = [int(value) for value in values]
        if not history or any(value < 0 or value >= item_count for value in history):
            raise AlgorithmExecutionError(
                "jagged histories must contain valid non-empty item IDs"
            )
        histories.append(history)
    if not histories:
        raise AlgorithmExecutionError("jagged recommendation shard is empty")
    offsets: list[int] = []
    flattened: list[int] = []
    for history in histories:
        offsets.append(len(flattened))
        flattened.extend(history)
    user_ids = torch.as_tensor(
        data[user_col].to_numpy(copy=True), dtype=torch.long, device=device
    )
    history_values = torch.as_tensor(flattened, dtype=torch.long, device=device)
    history_offsets = torch.as_tensor(offsets, dtype=torch.long, device=device)
    candidate_ids = torch.as_tensor(
        data[candidate_col].to_numpy(copy=True), dtype=torch.long, device=device
    )
    labels = torch.as_tensor(
        data[label_col].to_numpy(copy=True), dtype=torch.float32, device=device
    ).reshape(-1, 1)
    return (
        user_ids,
        history_values,
        history_offsets,
        candidate_ids,
        labels,
        len(flattened),
    )


def _route_sparse_keys(
    values: object,
    *,
    rank: int,
    world_size: int,
) -> int:
    import torch
    import torch.distributed as dist

    tokens = cast(torch.Tensor, values)
    if not dist.is_initialized() or world_size == 1:
        return int(tokens.numel())
    owners = torch.remainder(tokens, world_size)
    send_counts_tensor = torch.bincount(owners, minlength=world_size).to(
        dtype=torch.int64
    )
    gathered_counts = [torch.zeros_like(send_counts_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_counts, send_counts_tensor)
    send_counts = [int(value) for value in send_counts_tensor.cpu().tolist()]
    receive_counts = [
        int(gathered_counts[source][rank].cpu()) for source in range(world_size)
    ]
    ordered = tokens[torch.argsort(owners)].contiguous()
    received = torch.empty(
        sum(receive_counts),
        dtype=tokens.dtype,
        device=tokens.device,
    )
    dist.all_to_all_single(
        received,
        ordered,
        output_split_sizes=receive_counts,
        input_split_sizes=send_counts,
    )
    if received.numel() and not bool(
        torch.remainder(received, world_size).eq(rank).all()
    ):
        raise AlgorithmExecutionError("All-to-All sparse-key ownership drifted")
    return int(received.numel())


def _train_loop(config: dict[str, Any]) -> None:
    import pandas as pd
    import ray
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import get_device, prepare_model

    shard = train.get_dataset_shard("train")
    frames = [
        cast(pd.DataFrame, batch)
        for batch in shard.iter_batches(batch_format="pandas", batch_size=256)
    ]
    if not frames:
        raise AlgorithmExecutionError("jagged recommendation shard is empty")
    frame = pd.concat(frames, ignore_index=True)
    model_config = cast(Mapping[str, Any], config["model"])
    data_config = cast(Mapping[str, Any], config["data"])
    training = cast(Mapping[str, Any], config["training"])
    device = get_device()
    raw_model = cast(
        torch.nn.Module,
        _model(
            user_count=int(model_config["user_count"]),
            item_count=int(model_config["item_count"]),
            embedding_dim=int(model_config["embedding_dim"]),
        ),
    )
    model = prepare_model(raw_model)
    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    (
        user_ids,
        history_values,
        history_offsets,
        candidate_ids,
        labels,
        history_token_count,
    ) = _jagged_tensors(
        frame,
        user_col=str(data_config["user_col"]),
        history_col=str(data_config["history_col"]),
        candidate_col=str(data_config["candidate_col"]),
        label_col=str(data_config["label_col"]),
        item_count=int(model_config["item_count"]),
        device=device,
    )
    routed_token_count = _route_sparse_keys(
        history_values,
        rank=rank,
        world_size=world_size,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 0.01))
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(user_ids, history_values, history_offsets, candidate_ids)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()
    if dist.is_initialized():
        dist.barrier()
    unwrapped = cast(
        torch.nn.Module, model.module if hasattr(model, "module") else model
    )
    state = cast(Mapping[str, object], unwrapped.state_dict())
    model_digest = _state_digest(state)
    runtime = ray.get_runtime_context()
    assigned = runtime.get_assigned_resources()
    rows = int(labels.shape[0])
    worker = {
        "worker_id": str(runtime.get_worker_id()),
        "node_id": str(runtime.get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "shard_id": hashlib.sha256(
            f"{config['binding_digest']}:jagged:{rank}/{world_size}".encode("ascii")
        ).hexdigest(),
        "rows_processed": rows,
        "input_rows": {
            "train": rows,
            "coverage.history_tokens": history_token_count,
            "coverage.routed_owned_tokens": routed_token_count,
            "coverage.positive_pairs": int((labels == 1).sum()),
            "coverage.negative_pairs": int((labels == 0).sum()),
        },
        "batch_count": 1,
        "collective_steps": 2,
        "model_state_digest": model_digest,
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
    workers: list[dict[str, object] | None] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(workers, worker)
    else:
        workers[0] = worker
    if any(item is None for item in workers):
        raise AlgorithmExecutionError("jagged Worker evidence is incomplete")
    checkpoint = None
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix="tributo-jagged-") as directory:
            root = Path(directory)
            torch.save(dict(state), root / "model.pt")
            (root / "model_config.json").write_text(
                json.dumps(
                    {
                        "model": dict(model_config),
                        "data": dict(data_config),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            checkpoint = Checkpoint.from_directory(root)
            train.report(
                {
                    "loss": float(loss.detach().cpu()),
                    "execution_workers": cast(list[dict[str, object]], workers),
                    "model_state_digest": model_digest,
                },
                checkpoint=checkpoint,
            )
    else:
        train.report(
            {
                "loss": float(loss.detach().cpu()),
                "execution_workers": cast(list[dict[str, object]], workers),
                "model_state_digest": model_digest,
            }
        )


@dataclass(frozen=True)
class JaggedResult:
    checkpoint: object
    metrics: Mapping[str, object]
    evidence: Mapping[str, object]


class _JaggedDriver:
    def __init__(
        self,
        *,
        plan: ResolvedAlgorithmPlan,
        datasets: Mapping[str, object],
    ) -> None:
        self.plan = plan
        self.datasets = dict(datasets)

    def fit(self) -> JaggedResult:
        from ray.train import DataConfig, RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer

        config = self.plan.algorithm_config
        ray_config = cast(Mapping[str, Any], config["ray"])
        trainer = TorchTrainer(
            train_loop_per_worker=_train_loop,
            train_loop_config={
                "model": dict(cast(Mapping[str, Any], config["model"])),
                "data": dict(cast(Mapping[str, Any], config["data"])),
                "training": dict(cast(Mapping[str, Any], config["training"])),
                "binding_digest": (self.plan.primary_input_descriptor.binding_digest),
            },
            scaling_config=ScalingConfig(
                num_workers=self.plan.runtime.worker_count,
                use_gpu=self.plan.runtime.num_gpus > 0,
                resources_per_worker={
                    "CPU": self.plan.runtime.num_cpus,
                    "GPU": self.plan.runtime.num_gpus,
                    **dict(self.plan.runtime.custom_resources),
                },
                placement_strategy="SPREAD",
            ),
            datasets=cast(dict[str, Any], self.datasets),
            dataset_config=DataConfig(datasets_to_split=["train"]),
            run_config=RunConfig(
                name="tributo-jagged-embedding",
                storage_path=str(ray_config["storage_path"]),
            ),
        )
        result = cast(Any, trainer.fit())
        if result.checkpoint is None:
            raise AlgorithmExecutionError("jagged training did not checkpoint")
        workers = result.metrics.get("execution_workers")
        digest = result.metrics.get("model_state_digest")
        if not isinstance(workers, list) or not isinstance(digest, str):
            raise AlgorithmExecutionError("jagged Worker evidence is malformed")
        return JaggedResult(
            checkpoint=result.checkpoint,
            metrics=result.metrics,
            evidence={
                "workers": workers,
                "state": {
                    "coordination": "framework_native",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": digest,
                    "details": {
                        "framework": "pytorch-ddp",
                        "jagged": True,
                        "routing": "all_to_all_single_owner_mod",
                    },
                },
                "input_complete": True,
            },
        )


class DistributedJaggedEmbedding(FrameworkNativeAlgorithm):
    """Train one DDP recommender while routing sparse keys across workers."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan

    def validate_environment(self) -> None:
        try:
            import torch
            from ray.train.torch import TorchTrainer
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "jagged recommendation requires Ray Train and PyTorch"
            ) from exc
        if not torch.__version__ or TorchTrainer is None:
            raise AlgorithmConfigurationError(
                "jagged recommendation environment is invalid"
            )

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        if set(datasets) != {"train"}:
            raise AlgorithmConfigurationError(
                "jagged recommendation requires train Dataset"
            )
        data = cast(Mapping[str, Any], self.plan.algorithm_config["data"])
        columns = tuple(
            str(data[name])
            for name in ("user_col", "history_col", "candidate_col", "label_col")
        )
        binding = self.plan.primary_input_binding
        if (
            set(binding.feature_names) != set(columns[:-1])
            or binding.label_name != columns[-1]
        ):
            raise AlgorithmConfigurationError(
                "jagged recommendation InputBinding columns drifted"
            )
        return {"train": cast(Any, datasets["train"]).select_columns(list(columns))}

    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        del config
        return _JaggedDriver(plan=self.plan, datasets=datasets)

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        if not isinstance(result, JaggedResult):
            raise AlgorithmExecutionError("jagged training returned an invalid result")
        return result.evidence

    def checkpoint_source(self, result: object) -> object:
        if not isinstance(result, JaggedResult):
            raise AlgorithmExecutionError("jagged checkpoint result is invalid")
        return result.checkpoint


def create_jagged_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedJaggedEmbedding:
    del artifacts
    if implementation is not DistributedJaggedEmbedding:
        raise AlgorithmConfigurationError("jagged implementation drifted")
    return DistributedJaggedEmbedding(plan)


def export_jagged_result(
    *,
    result: object,
    checkpoint: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    import importlib.metadata

    import torch
    from tributo.exporting.models import BundleOutputConfig, ExportSource, ExportTarget
    from tributo.exporting.service import BundleExportService

    if not isinstance(result, JaggedResult):
        raise AlgorithmExecutionError("jagged export result is invalid")
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError("jagged output.bundle_uri is required")
    with cast(Any, checkpoint).as_directory() as directory:
        root = Path(directory)
        model_config = json.loads(
            (root / "model_config.json").read_text(encoding="utf-8")
        )
        model_values = cast(Mapping[str, Any], model_config["model"])
        model = cast(
            torch.nn.Module,
            _model(
                user_count=int(model_values["user_count"]),
                item_count=int(model_values["item_count"]),
                embedding_dim=int(model_values["embedding_dim"]),
            ),
        )
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        fingerprint = _state_digest(cast(Mapping[str, object], model.state_dict()))
        source = ExportSource(
            source_kind="torch_module",
            model_object=model,
            architecture_id=plan.resolution.implementation_id,
            model_config_data=model_config,
            feature_schema={
                "feature_names": list(plan.primary_input_binding.feature_names),
                "jagged_feature": model_config["data"]["history_col"],
            },
            metadata={
                "framework": "pytorch",
                "task_type": "ranking",
                "sparse_routing": "all_to_all_single_owner_mod",
                "producer_distribution": "tributo-algorithms-recsys-torch",
            },
            source_fingerprint=fingerprint,
        )
        bundle = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=str(output["bundle_uri"]),
                request_id=run_id,
                run_id=run_id,
                targets=[
                    ExportTarget(
                        name="jagged-ranking-model",
                        format="safetensors",
                        exporter_id="torch-safetensors-v1",
                    )
                ],
                roles={"model": "jagged-ranking-model"},
            ),
            tributo_version=importlib.metadata.version("tributo"),
        )
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics={"train_loss": float(cast(Any, result.metrics.get("loss", 0.0)))},
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
        },
    )


__all__ = [
    "DistributedJaggedEmbedding",
    "JaggedResult",
    "create_jagged_algorithm",
    "export_jagged_result",
]
