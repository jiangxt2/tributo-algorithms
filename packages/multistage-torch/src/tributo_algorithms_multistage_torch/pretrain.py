"""RayTorchAdapter for pretrain-then-finetune classification."""

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

STAGES = ("pretrain", "finetune")


def _encoder(input_features: int, hidden_features: int) -> Any:
    import torch

    if input_features < 1 or hidden_features < 1:
        raise ValueError("pretrain model dimensions must be positive")
    return torch.nn.Sequential(
        torch.nn.Linear(input_features, hidden_features), torch.nn.ReLU()
    )


def _pretrain_model(input_features: int, hidden_features: int) -> object:
    import torch

    class PretrainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _encoder(input_features, hidden_features)
            self.decoder = torch.nn.Linear(hidden_features, input_features)

        def forward(self, features: object) -> object:
            return self.decoder(self.encoder(cast(torch.Tensor, features)))

    return PretrainModel()


def _finetune_model(input_features: int, hidden_features: int) -> object:
    import torch

    class FinetuneModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _encoder(input_features, hidden_features)
            self.classifier = torch.nn.Linear(hidden_features, 1)

        def forward(self, features: object) -> object:
            return self.classifier(self.encoder(cast(torch.Tensor, features)))

    return FinetuneModel()


def _state_digest(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = cast(Any, state[name]).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


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
            "pretrain-finetune adapter accepts only Ray failure retry Checkpoints"
        )
    import torch

    opener = getattr(checkpoint_context.checkpoint.checkpoint, "as_directory", None)
    if not callable(opener):
        raise AlgorithmExecutionError(
            "pretrain-finetune retry checkpoint cannot be opened"
        )
    with opener() as directory:
        root = Path(directory)
        model_path = root / "model.pt"
        optimizer_path = root / "optimizer.pt"
        if not model_path.is_file() or not optimizer_path.is_file():
            raise AlgorithmExecutionError(
                "pretrain-finetune retry checkpoint is missing model/optimizer state"
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
                raise AlgorithmExecutionError(
                    "pretrain-finetune retry RNG state is malformed"
                )
    descriptor = checkpoint_context.checkpoint.descriptor
    return int(descriptor.completed_step) if descriptor is not None else 0


def _load_encoder_checkpoint(
    checkpoint_context: TorchWorkerCheckpointContext, model: object
) -> None:
    if checkpoint_context.checkpoint is None:
        raise AlgorithmExecutionError("finetune stage requires pretrain Checkpoint")
    import torch

    opener = getattr(checkpoint_context.checkpoint.checkpoint, "as_directory", None)
    if not callable(opener):
        raise AlgorithmExecutionError("pretrain Checkpoint cannot be opened")
    with opener() as directory:
        path = Path(directory) / "encoder.pt"
        if not path.is_file():
            raise AlgorithmExecutionError("pretrain Checkpoint is missing encoder.pt")
        target = cast(torch.nn.Module, getattr(model, "module", model))
        cast(torch.nn.Module, target.encoder).load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )


def _stage_train_loop(
    config: Mapping[str, Any], checkpoint_context: TorchWorkerCheckpointContext
) -> None:
    import ray
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import get_device, prepare_model

    stage = str(config.get("stage", "pretrain"))
    if stage not in STAGES:
        raise AlgorithmConfigurationError("pretrain-finetune stage is invalid")
    feature_names = tuple(str(name) for name in config.get("feature_names", ()))
    label_name = str(config.get("label_name", "label"))
    if not feature_names:
        raise AlgorithmConfigurationError(
            "pretrain-finetune feature_names are required"
        )
    shard = train.get_dataset_shard("train")
    raw_batches = list(
        shard.iter_torch_batches(
            batch_size=int(config.get("batch_size", 32)), drop_last=False
        )
    )
    if not raw_batches:
        raise AlgorithmExecutionError(f"{stage} shard is empty")
    first_batch = raw_batches[0]
    if not isinstance(first_batch, Mapping):
        raise AlgorithmExecutionError("pretrain-finetune batch is not columnar")
    if not feature_names:
        feature_names = tuple(name for name in first_batch if name != label_name)
    batches: list[tuple[object, object]] = []
    for batch in raw_batches:
        if not isinstance(batch, Mapping):
            raise AlgorithmExecutionError("pretrain-finetune batch is not columnar")
        features = torch.stack([batch[name].float() for name in feature_names], dim=1)
        labels = batch[label_name].float().reshape(-1, 1)
        if (
            not torch.isfinite(features).all()
            or not torch.isfinite(labels).all()
            or not bool(((labels == 0) | (labels == 1)).all())
        ):
            raise AlgorithmExecutionError(
                "pretrain-finetune batch contains invalid values"
            )
        batches.append((features, labels))
    if not batches:
        raise AlgorithmExecutionError(f"{stage} shard is empty")
    model_config = config.get("model", {})
    training = config.get("training", {})
    if not isinstance(model_config, Mapping) or not isinstance(training, Mapping):
        raise AlgorithmConfigurationError(
            "pretrain-finetune model/training config is invalid"
        )
    input_features = int(model_config["input_features"])
    hidden_features = int(model_config["hidden_features"])
    if len(feature_names) != input_features:
        raise AlgorithmConfigurationError(
            "pretrain-finetune feature binding does not match model.input_features"
        )
    device: Any = get_device()
    if stage == "pretrain":
        model = cast(torch.nn.Module, _pretrain_model(input_features, hidden_features))
    else:
        model = cast(torch.nn.Module, _finetune_model(input_features, hidden_features))
    model = prepare_model(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 0.01))
    )
    epochs = int(
        training.get("pretrain_epochs" if stage == "pretrain" else "finetune_epochs", 1)
    )
    rank, world_size = (
        train.get_context().get_world_rank(),
        train.get_context().get_world_size(),
    )
    restored_step = 0
    if checkpoint_context.checkpoint is not None:
        if checkpoint_context.source == "stage_dependency":
            if stage != "finetune":
                raise AlgorithmExecutionError(
                    "pretrain stage cannot consume a Stage dependency Checkpoint"
                )
            _load_encoder_checkpoint(checkpoint_context, model)
        elif checkpoint_context.source == "ray_failure_retry":
            restored_step = _load_retry_state(
                checkpoint_context, model, optimizer, rank=rank
            )
        else:
            raise AlgorithmExecutionError(
                f"pretrain-finetune checkpoint source {checkpoint_context.source!r} is invalid"
            )
    elif stage == "finetune":
        raise AlgorithmExecutionError("finetune stage requires pretrain Checkpoint")
    start_epoch = min(restored_step, epochs)
    last_numerator = 0.0
    last_rows = 0
    for epoch in range(start_epoch, epochs):
        optimizer.zero_grad(set_to_none=True)
        numerator_total: torch.Tensor | None = None
        rows = 0
        for raw_features, raw_labels in batches:
            features = cast(torch.Tensor, cast(Any, raw_features).to(device))
            labels = cast(torch.Tensor, cast(Any, raw_labels).to(device))
            output = cast(torch.Tensor, model(features))
            numerator: torch.Tensor
            if stage == "pretrain":
                squared = (output - features) ** 2
                numerator = cast(torch.Tensor, squared.sum())
                normalizer = int(squared.numel())
            else:
                numerator = cast(
                    torch.Tensor,
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        output, labels, reduction="sum"
                    ),
                )
                normalizer = int(labels.shape[0])
            numerator_total = (
                numerator if numerator_total is None else numerator_total + numerator
            )
            rows += normalizer
        if numerator_total is None:
            raise AlgorithmExecutionError("pretrain-finetune produced no loss")

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
            TorchLossContribution(numerator_total, rows),
            TorchAccumulationWindow(epoch, 1),
            TorchBackwardContext(
                world_size=world_size,
                backward=lambda value: value.backward(),
                reduce_normalizer=reduce_normalizer,
                finalize_window=finalize,
            ),
        )
        last_numerator, last_rows = float(numerator_total.detach().item()), rows

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

    reduced_loss = reduce_torch_metrics(
        {"train_loss": TorchMetricContribution(last_numerator, last_rows)},
        TorchMetricPolicy({"train_loss": "sum_count"}),
        TorchMetricReductionContext(reduce_metric),
    ).values["train_loss"]
    unwrapped = cast(
        torch.nn.Module, model.module if hasattr(model, "module") else model
    )
    digest = _state_digest(cast(Mapping[str, object], unwrapped.state_dict()))
    train_rows = sum(
        int(cast(torch.Tensor, features).shape[0]) for features, _ in batches
    )
    worker = {
        "worker_id": f"pretrain-{stage}-{rank}",
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "shard_id": f"{stage}-{rank}",
        "rows_processed": train_rows,
        "input_rows": {"train": train_rows},
        "batch_count": max(epochs - start_epoch, 0),
        "collective_steps": max(epochs - start_epoch, 0),
        "model_state_digest": digest,
        "stage": stage,
    }
    workers: list[object] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(workers, worker)
    else:
        workers[0] = worker
    with tempfile.TemporaryDirectory(prefix=f"tributo-pretrain-{stage}-") as directory:
        root = Path(directory)
        if stage == "pretrain":
            torch.save(cast(Any, unwrapped).encoder.state_dict(), root / "encoder.pt")
        torch.save(unwrapped.state_dict(), root / "model.pt")
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
                {
                    "input_features": input_features,
                    "hidden_features": hidden_features,
                    "feature_names": list(feature_names),
                    "stage": stage,
                },
                sort_keys=True,
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

        metric_name = "pretrain_loss" if stage == "pretrain" else "finetune_loss"
        report_torch_checkpoint(
            {
                "stage": stage,
                "train_loss": reduced_loss,
                metric_name: reduced_loss,
                "execution_workers": workers,
                "model_state_digest": digest,
            },
            Draft(),
            checkpoint_context.stage,
            epochs,
        )


@PublicAPI(stability="alpha")
class DistributedPretrainFinetune(RayTorchAdapter):
    """Execute pretraining then finetuning through Core's component plan."""

    def validate_environment(self, context: TorchRuntimeContext) -> None:
        del context
        try:
            import torch
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "pretrain-finetune requires PyTorch"
            ) from exc
        if not torch.__version__:
            raise AlgorithmConfigurationError(
                "pretrain-finetune environment is invalid"
            )

    def bind_datasets(
        self, datasets: Mapping[str, object], context: TorchStageContext
    ) -> Mapping[str, object]:
        del context
        if set(datasets) != {"train"}:
            raise AlgorithmConfigurationError(
                "pretrain-finetune requires train Dataset"
            )
        return dict(datasets)

    def worker_config(self, context: TorchStageContext) -> Mapping[str, object]:
        config = {
            key: value
            for key, value in context.runtime.algorithm_config.items()
            if key not in {"ray", "output"}
        }
        config["stage"] = context.stage_id
        binding = context.runtime.input_bindings.get("train")
        if not isinstance(binding, Mapping):
            raise AlgorithmConfigurationError(
                "pretrain-finetune Stage context is missing the train binding"
            )
        feature_names = binding.get("feature_names")
        label_name = binding.get("label_name")
        if (
            not isinstance(feature_names, (list, tuple))
            or not feature_names
            or not all(isinstance(name, str) and name for name in feature_names)
            or not isinstance(label_name, str)
            or not label_name
        ):
            raise AlgorithmConfigurationError(
                "pretrain-finetune train binding has an invalid typed layout"
            )
        config["feature_names"] = tuple(feature_names)
        config["label_name"] = label_name
        return config

    def train_loop_per_worker(
        self,
        worker_config: Mapping[str, object],
        checkpoint_context: TorchWorkerCheckpointContext,
    ) -> None:
        _stage_train_loop(worker_config, checkpoint_context)

    def checkpoint_source(
        self, result: object, context: TorchCheckpointContext
    ) -> object:
        del context
        checkpoint = getattr(result, "checkpoint", None)
        if checkpoint is None:
            raise AlgorithmExecutionError("pretrain-finetune result has no checkpoint")
        return checkpoint

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan(
            {
                "train_loss": "sum_count",
                "pretrain_loss": "sum_count",
                "finetune_loss": "sum_count",
            }
        )

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        config = context.stage.runtime.algorithm_config.get("model", {})
        features = (
            int(config.get("input_features", 1)) if isinstance(config, Mapping) else 1
        )
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {
                    "name": "float_input",
                    "dtype": "float32",
                    "shape": ("batch", features),
                },
            ),
            output_signature=(
                {"name": "output", "dtype": "float32", "shape": ("batch", 1)},
            ),
            targets=(
                {
                    "name": "finetuned-model",
                    "format": "safetensors",
                    "exporter_id": "torch-safetensors-v1",
                },
                {
                    "name": "finetuned-inference",
                    "format": "onnx",
                    "exporter_id": "torch-onnx-v1",
                    "options": {"dynamo": False},
                },
            ),
            roles={"model": "finetuned-model", "inference": "finetuned-inference"},
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
            raise AlgorithmExecutionError(
                "pretrain-finetune checkpoint cannot be opened"
            )
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
                    "pretrain-finetune checkpoint is missing payloads: "
                    f"{missing_payloads}"
                )
            config = json.loads(
                (root / "model_config.json").read_text(encoding="utf-8")
            )
            model = cast(
                torch.nn.Module,
                _finetune_model(
                    int(config["input_features"]), int(config["hidden_features"])
                ),
            )
            try:
                model.load_state_dict(
                    torch.load(root / "model.pt", map_location="cpu", weights_only=True)
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise AlgorithmExecutionError(
                    "pretrain-finetune model payload is incompatible"
                ) from exc
            model.eval()
            yield ExportSource(
                source_kind="torch_module",
                model_object=model,
                architecture_id=artifact_context.stage.runtime.implementation_id,
                model_config_data=config,
                feature_schema={"input_names": ["float_input"]},
                sample_inputs={
                    "float_input": torch.zeros(
                        (2, int(config["input_features"])), dtype=torch.float32
                    )
                },
                metadata={
                    "framework": "pytorch",
                    "task_type": "binary_classification",
                    "component_stages": list(STAGES),
                    "producer_distribution": "tributo-algorithms-multistage-torch",
                },
                source_fingerprint=_state_digest(
                    cast(Mapping[str, object], model.state_dict())
                ),
                checkpoint_contract=ExportCheckpointV1(
                    trainer_type="pretrain_finetune_classifier",
                    architecture_id=artifact_context.stage.runtime.implementation_id,
                    input_schema=(
                        CheckpointField(
                            name="float_input",
                            dtype="float32",
                            shape=("batch", int(config["input_features"])),
                        ),
                    ),
                    output_schema=(
                        CheckpointField(
                            name="output", dtype="float32", shape=("batch", 1)
                        ),
                    ),
                    preprocessing={"type": "ordered_features"},
                    task_type="binary_classification",
                    framework="pytorch",
                    framework_version=torch.__version__,
                ),
            )


def create_pretrain_finetune_algorithm(
    *,
    plan: object | None = None,
    implementation: object | None = None,
    artifacts: tuple[object, ...] = (),
) -> DistributedPretrainFinetune:
    del plan, artifacts
    if implementation is not DistributedPretrainFinetune:
        raise AlgorithmConfigurationError("pretrain-finetune implementation drifted")
    return DistributedPretrainFinetune()


__all__ = [
    "DistributedPretrainFinetune",
    "STAGES",
    "create_pretrain_finetune_algorithm",
]
