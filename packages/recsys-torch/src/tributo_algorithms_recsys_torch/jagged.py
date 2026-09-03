"""RayTorchAdapter for jagged-history recommendation with All-to-All routing."""

from __future__ import annotations

import hashlib
import json
import os
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
    TorchMetricPlan,
    TorchRuntimeContext,
    TorchStageContext,
    TorchWorkerCheckpointContext,
)
from tributo.util.annotations import PublicAPI


def _model(*, user_count: int, item_count: int, embedding_dim: int) -> object:
    import torch

    if min(user_count, item_count, embedding_dim) < 1:
        raise ValueError("jagged model dimensions must be positive")

    class JaggedRankingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.user_embedding = torch.nn.Embedding(user_count, embedding_dim)
            self.history_embedding = torch.nn.EmbeddingBag(
                item_count, embedding_dim, mode="mean", include_last_offset=False
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
            users = cast(torch.Tensor, user_ids).long()
            values = cast(torch.Tensor, history_values).long()
            offsets = cast(torch.Tensor, history_offsets).long()
            candidates = cast(torch.Tensor, candidate_ids).long()
            return (
                self.user_embedding(users) + self.history_embedding(values, offsets)
            ).mul(self.candidate_embedding(candidates)).sum(
                dim=1, keepdim=True
            ) + self.bias

    return JaggedRankingModel()


def _padded_inference_model(model: object) -> object:
    import torch

    base = cast(torch.nn.Module, model)

    class PaddedJaggedRankingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.user_embedding = cast(Any, base).user_embedding
            self.history_embedding = cast(Any, base).history_embedding
            self.candidate_embedding = cast(Any, base).candidate_embedding
            self.bias = cast(Any, base).bias

        def forward(
            self, user_ids: object, item_history: object, candidate_ids: object
        ) -> object:
            users = cast(torch.Tensor, user_ids).long()
            history = cast(torch.Tensor, item_history).long()
            candidates = cast(torch.Tensor, candidate_ids).long()
            user_count = self.user_embedding.num_embeddings
            item_count = self.candidate_embedding.num_embeddings
            user_valid = users.ge(0).logical_and(users.lt(user_count))
            candidate_valid = candidates.ge(0).logical_and(candidates.lt(item_count))
            history_valid = history.eq(-1).logical_or(
                history.ge(0).logical_and(history.lt(item_count))
            )
            valid = user_valid.logical_and(candidate_valid).logical_and(
                history_valid.all(dim=1)
            )
            safe_users = users.clamp(min=0, max=user_count - 1)
            safe_candidates = candidates.clamp(min=0, max=item_count - 1)
            mask = history.ge(0).logical_and(history_valid)
            safe_history = history.clamp(min=0, max=item_count - 1)
            embeddings = torch.nn.functional.embedding(
                safe_history, self.history_embedding.weight
            )
            weights = mask.unsqueeze(-1).to(dtype=embeddings.dtype)
            history_mean = (embeddings * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1.0)
            score = (self.user_embedding(safe_users) + history_mean).mul(
                self.candidate_embedding(safe_candidates)
            ).sum(dim=1, keepdim=True) + self.bias
            output = torch.where(valid.unsqueeze(-1), score, torch.zeros_like(score))
            return torch.cat(
                (output, valid.unsqueeze(-1).to(dtype=output.dtype)), dim=1
            )

    return PaddedJaggedRankingModel()


def _state_digest(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = cast(Any, state[name]).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
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
    user_count: int | None = None,
) -> tuple[Any, Any, Any, Any, Any, int]:
    import numpy as np
    import torch

    data = cast(Any, frame)
    histories: list[list[int]] = []
    for raw in data[history_col].tolist():
        values = (
            raw.tolist()
            if isinstance(raw, np.ndarray)
            else list(raw)
            if isinstance(raw, (list, tuple))
            else None
        )
        if not values:
            raise AlgorithmExecutionError(
                "jagged histories must be non-empty sequences"
            )
        history = [int(value) for value in values]
        if any(value < 0 or value >= item_count for value in history):
            raise AlgorithmExecutionError("jagged history item ID is out of range")
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
    if user_count is not None and (
        bool((user_ids < 0).any()) or bool((user_ids >= user_count).any())
    ):
        raise AlgorithmExecutionError("jagged user ID is out of range")
    if bool((candidate_ids < 0).any()) or bool((candidate_ids >= item_count).any()):
        raise AlgorithmExecutionError("jagged candidate item ID is out of range")
    if user_ids.shape != candidate_ids.shape or user_ids.shape[0] != labels.shape[0]:
        raise AlgorithmExecutionError("jagged interaction columns have mismatched rows")
    if not torch.isfinite(labels).all() or not bool(
        ((labels == 0) | (labels == 1)).all()
    ):
        raise AlgorithmExecutionError("jagged labels must be finite 0 or 1")
    return (
        user_ids,
        history_values,
        history_offsets,
        candidate_ids,
        labels,
        len(flattened),
    )


def _route_sparse_keys(values: object, *, rank: int, world_size: int) -> int:
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
        sum(receive_counts), dtype=tokens.dtype, device=tokens.device
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


def _load_retry_state(
    checkpoint_context: TorchWorkerCheckpointContext,
    model: object,
    optimizer: object,
    *,
    rank: int,
) -> int:
    if checkpoint_context.checkpoint is None:
        return 0
    if checkpoint_context.source != "ray_failure_retry":
        raise AlgorithmExecutionError(
            "jagged adapter accepts only Ray failure retry Checkpoints"
        )
    import torch

    opener = getattr(checkpoint_context.checkpoint.checkpoint, "as_directory", None)
    if not callable(opener):
        raise AlgorithmExecutionError("jagged retry checkpoint cannot be opened")
    with opener() as directory:
        root = Path(directory)
        model_path = root / "model.pt"
        optimizer_path = root / "optimizer.pt"
        if not model_path.is_file() or not optimizer_path.is_file():
            raise AlgorithmExecutionError(
                "jagged retry checkpoint is missing model/optimizer state"
            )
        target_model = cast(torch.nn.Module, getattr(model, "module", model))
        target_model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        cast(Any, optimizer).load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=True)
        )
        rng_path = root / "rng_state.pt"
        if rng_path.is_file():
            payload = torch.load(rng_path, map_location="cpu", weights_only=True)
            states = payload.get("states") if isinstance(payload, Mapping) else None
            raw_state = (
                states[rank]
                if isinstance(states, list) and rank < len(states)
                else payload
            )
            if isinstance(raw_state, bytes):
                torch.set_rng_state(
                    torch.frombuffer(bytearray(raw_state), dtype=torch.uint8).clone()
                )
            elif isinstance(raw_state, torch.Tensor):
                torch.set_rng_state(raw_state.to(dtype=torch.uint8).cpu())
            else:
                raise AlgorithmExecutionError("jagged retry RNG state is malformed")
    descriptor = checkpoint_context.checkpoint.descriptor
    return int(descriptor.completed_step) if descriptor is not None else 0


def _train_loop(
    config: Mapping[str, Any], checkpoint_context: TorchWorkerCheckpointContext
) -> None:
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
    model_config = config.get("model")
    data_config = config.get("data")
    training = config.get("training")
    if (
        not isinstance(model_config, Mapping)
        or not isinstance(data_config, Mapping)
        or not isinstance(training, Mapping)
    ):
        raise AlgorithmConfigurationError(
            "jagged model/data/training config is invalid"
        )
    device = get_device()
    model = prepare_model(
        cast(
            torch.nn.Module,
            _model(
                user_count=int(model_config["user_count"]),
                item_count=int(model_config["item_count"]),
                embedding_dim=int(model_config["embedding_dim"]),
            ),
        )
    )
    context = train.get_context()
    rank, world_size = context.get_world_rank(), context.get_world_size()
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
        user_count=int(model_config["user_count"]),
        item_count=int(model_config["item_count"]),
        device=device,
    )
    routed_token_count = _route_sparse_keys(
        history_values, rank=rank, world_size=world_size
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 0.01))
    )
    restored_step = _load_retry_state(checkpoint_context, model, optimizer, rank=rank)
    logits = model(user_ids, history_values, history_offsets, candidate_ids)
    numerator = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="sum"
    )
    rows = int(labels.shape[0])

    def reduce_normalizer(value: float) -> float:
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def finalize(scale: float) -> None:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training.get("max_gradient_norm", 1.0))
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    apply_torch_loss_backward(
        TorchLossContribution(numerator, rows),
        TorchAccumulationWindow(0, 1),
        TorchBackwardContext(
            world_size=world_size,
            backward=lambda value: value.backward(),
            reduce_normalizer=reduce_normalizer,
            finalize_window=finalize,
        ),
    )

    def reduce_metric(
        name: str, contribution: TorchMetricContribution, reducer: str
    ) -> float:
        del name, reducer
        state = torch.tensor(
            [contribution.numerator, contribution.normalizer],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_initialized():
            dist.all_reduce(state, op=dist.ReduceOp.SUM)
        return float(state[0].item() / state[1].item()) if state[1].item() > 0 else 0.0

    train_loss = reduce_torch_metrics(
        {"train_loss": TorchMetricContribution(float(numerator.detach().item()), rows)},
        TorchMetricPolicy({"train_loss": "sum_count"}),
        TorchMetricReductionContext(reduce_metric),
    ).values["train_loss"]
    unwrapped = cast(
        torch.nn.Module, model.module if hasattr(model, "module") else model
    )
    state = cast(Mapping[str, object], unwrapped.state_dict())
    model_digest = _state_digest(state)
    worker = {
        "worker_id": str(ray.get_runtime_context().get_worker_id()),
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "shard_id": f"jagged-{rank}",
        "rows_processed": rows,
        "input_rows": {
            "train": rows,
            "coverage.history_tokens": history_token_count,
            "coverage.routed_owned_tokens": routed_token_count,
            "coverage.positive_pairs": int((labels == 1).sum().item()),
            "coverage.negative_pairs": int((labels == 0).sum().item()),
        },
        "batch_count": 1,
        "collective_steps": 2,
        "model_state_digest": model_digest,
    }
    workers: list[object] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(workers, worker)
    else:
        workers[0] = worker
    with tempfile.TemporaryDirectory(prefix="tributo-jagged-") as directory:
        root = Path(directory)
        torch.save(dict(state), root / "model.pt")
        torch.save(optimizer.state_dict(), root / "optimizer.pt")
        rng_state = torch.get_rng_state().cpu().numpy().tobytes()
        rng_states: list[object] = [rng_state] * world_size
        if dist.is_initialized():
            dist.all_gather_object(rng_states, rng_state)
        torch.save(
            {"world_size": world_size, "states": rng_states}, root / "rng_state.pt"
        )
        (root / "model_config.json").write_text(
            json.dumps(
                {"model": dict(model_config), "data": dict(data_config)}, sort_keys=True
            ),
            encoding="utf-8",
        )

        class Draft:
            checkpoint_dir: str | os.PathLike[str] = str(root)

            def report(
                self,
                *,
                metrics: Mapping[str, object],
                stage_context: object,
                completed_step: int,
            ) -> None:
                del stage_context, completed_step
                checkpoint = Checkpoint.from_directory(str(root)) if rank == 0 else None
                train.report(dict(metrics), checkpoint=checkpoint)

        report_torch_checkpoint(
            {
                "train_loss": train_loss,
                "execution_workers": workers,
                "model_state_digest": model_digest,
            },
            Draft(),
            checkpoint_context.stage,
            restored_step + 1,
        )


@PublicAPI(stability="alpha")
class DistributedJaggedEmbedding(RayTorchAdapter):
    """Train a DDP recommender while routing sparse history keys across workers."""

    def validate_environment(self, context: TorchRuntimeContext) -> None:
        del context
        try:
            import torch
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "jagged recommendation requires PyTorch"
            ) from exc
        if not torch.__version__:
            raise AlgorithmConfigurationError(
                "jagged recommendation environment is invalid"
            )

    def bind_datasets(
        self, datasets: Mapping[str, object], context: TorchStageContext
    ) -> Mapping[str, object]:
        del context
        if set(datasets) != {"train"}:
            raise AlgorithmConfigurationError(
                "jagged recommendation requires train Dataset"
            )
        return dict(datasets)

    def worker_config(self, context: TorchStageContext) -> Mapping[str, object]:
        config = {
            key: value
            for key, value in context.runtime.algorithm_config.items()
            if key not in {"ray", "output"}
        }
        binding = context.runtime.input_bindings.get("train")
        if not isinstance(binding, Mapping):
            raise AlgorithmConfigurationError(
                "jagged Stage context is missing the train binding"
            )
        feature_names = binding.get("feature_names")
        label_name = binding.get("label_name")
        if (
            not isinstance(feature_names, (list, tuple))
            or len(feature_names) != 3
            or not all(isinstance(name, str) and name for name in feature_names)
            or not isinstance(label_name, str)
            or not label_name
            or binding.get("sample_weight_name") is not None
        ):
            raise AlgorithmConfigurationError(
                "jagged train binding has an invalid typed layout"
            )
        data = config.get("data")
        if not isinstance(data, Mapping):
            raise AlgorithmConfigurationError("jagged data config is required")
        normalized_data = dict(data)
        normalized_data.update(
            {
                "user_col": feature_names[0],
                "history_col": feature_names[1],
                "candidate_col": feature_names[2],
                "label_col": label_name,
            }
        )
        config["data"] = normalized_data
        return config

    def train_loop_per_worker(
        self,
        worker_config: Mapping[str, object],
        checkpoint_context: TorchWorkerCheckpointContext,
    ) -> None:
        _train_loop(worker_config, checkpoint_context)

    def checkpoint_source(
        self, result: object, context: TorchCheckpointContext
    ) -> object:
        del context
        checkpoint = getattr(result, "checkpoint", None)
        if checkpoint is None:
            raise AlgorithmExecutionError("jagged training result has no checkpoint")
        return checkpoint

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"train_loss": "sum_count"})

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        config = context.stage.runtime.algorithm_config.get("data", {})
        width = (
            int(config.get("inference_history_width", 1))
            if isinstance(config, Mapping)
            else 1
        )
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {"name": "user_id", "dtype": "int64", "shape": ("batch",)},
                {"name": "item_history", "dtype": "int64", "shape": ("batch", width)},
                {"name": "item_id", "dtype": "int64", "shape": ("batch",)},
            ),
            output_signature=(
                {"name": "output", "dtype": "float32", "shape": ("batch", 2)},
            ),
            targets=(
                {
                    "name": "jagged-ranking-model",
                    "format": "safetensors",
                    "exporter_id": "torch-safetensors-v1",
                },
                {
                    "name": "jagged-ranking-inference",
                    "format": "onnx",
                    "exporter_id": "torch-onnx-v1",
                    "options": {"dynamo": False},
                },
            ),
            roles={
                "model": "jagged-ranking-model",
                "inference": "jagged-ranking-inference",
            },
        )

    @contextmanager
    def open_export_source(
        self, checkpoint_ref: TorchCheckpointRef, artifact_context: TorchArtifactContext
    ) -> Any:
        import torch
        from tributo.exporting.models import (
            CheckpointField,
            ExportCheckpointV1,
            ExportSource,
        )

        opener = getattr(checkpoint_ref.checkpoint, "as_directory", None)
        if not callable(opener):
            raise AlgorithmExecutionError("jagged checkpoint cannot be opened")
        with opener() as directory:
            root = Path(directory)
            required_payloads = ("model_config.json", "model.pt")
            missing_payloads = [
                name
                for name in required_payloads
                if not (root / name).is_file() or (root / name).is_symlink()
            ]
            if missing_payloads:
                raise AlgorithmExecutionError(
                    f"jagged checkpoint is missing payloads: {missing_payloads}"
                )
            metadata = json.loads(
                (root / "model_config.json").read_text(encoding="utf-8")
            )
            model_config = cast(Mapping[str, Any], metadata["model"])
            data_config = cast(Mapping[str, Any], metadata["data"])
            model = cast(
                torch.nn.Module,
                _model(
                    user_count=int(model_config["user_count"]),
                    item_count=int(model_config["item_count"]),
                    embedding_dim=int(model_config["embedding_dim"]),
                ),
            )
            try:
                model.load_state_dict(
                    torch.load(root / "model.pt", map_location="cpu", weights_only=True)
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise AlgorithmExecutionError(
                    "jagged model payload is incompatible"
                ) from exc
            model.eval()
            yield ExportSource(
                source_kind="torch_module",
                model_object=_padded_inference_model(model),
                architecture_id=artifact_context.stage.runtime.implementation_id,
                model_config_data=metadata,
                feature_schema={
                    "input_names": ["user_id", "item_history", "item_id"],
                    "raw_feature_names": [
                        str(data_config["user_col"]),
                        str(data_config["history_col"]),
                        str(data_config["candidate_col"]),
                    ],
                    "jagged_feature": str(data_config["history_col"]),
                    "user_id_min": 0,
                    "user_id_max_exclusive": int(model_config["user_count"]),
                    "item_id_min": 0,
                    "item_id_max_exclusive": int(model_config["item_count"]),
                    "output_layout": {"score": 0, "valid": 1},
                    "history_width": int(data_config["inference_history_width"]),
                },
                sample_inputs={
                    "user_id": torch.tensor([0, 0], dtype=torch.int64),
                    "item_history": torch.tensor(
                        [[0] + [-1] * (int(data_config["inference_history_width"]) - 1)]
                        * 2,
                        dtype=torch.int64,
                    ),
                    "item_id": torch.tensor([0, 0], dtype=torch.int64),
                },
                metadata={
                    "framework": "pytorch",
                    "task_type": "ranking",
                    "sparse_routing": "all_to_all_single_owner_mod",
                    "producer_distribution": "tributo-algorithms-recsys-torch",
                },
                source_fingerprint=_state_digest(
                    cast(Mapping[str, object], model.state_dict())
                ),
                checkpoint_contract=ExportCheckpointV1(
                    trainer_type="jagged_embedding_recommender",
                    architecture_id=artifact_context.stage.runtime.implementation_id,
                    input_schema=(
                        CheckpointField(
                            name="user_id", dtype="int64", shape=("batch",)
                        ),
                        CheckpointField(
                            name="item_history",
                            dtype="int64",
                            shape=(
                                "batch",
                                int(data_config["inference_history_width"]),
                            ),
                        ),
                        CheckpointField(
                            name="item_id", dtype="int64", shape=("batch",)
                        ),
                    ),
                    output_schema=(
                        CheckpointField(
                            name="output", dtype="float32", shape=("batch", 2)
                        ),
                    ),
                    preprocessing={
                        "type": "padded_jagged_history",
                        "padding_value": -1,
                        "history_width": int(data_config["inference_history_width"]),
                        "invalid_id_policy": "zero_output_with_valid_false",
                    },
                    task_type="ranking",
                    framework="pytorch",
                    framework_version=torch.__version__,
                ),
            )


def create_jagged_algorithm(
    *,
    plan: object | None = None,
    implementation: object | None = None,
    artifacts: tuple[object, ...] = (),
) -> DistributedJaggedEmbedding:
    del plan, artifacts
    if implementation is not DistributedJaggedEmbedding:
        raise AlgorithmConfigurationError("jagged implementation drifted")
    return DistributedJaggedEmbedding()


__all__ = ["DistributedJaggedEmbedding", "create_jagged_algorithm"]
