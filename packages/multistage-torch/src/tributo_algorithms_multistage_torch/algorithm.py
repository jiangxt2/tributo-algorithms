"""RayTorchAdapter implementation for Teacher-to-Student distillation."""

from __future__ import annotations

import hashlib
import json
import math
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
    TorchMetricPlan,
    TorchRuntimeContext,
    TorchStageContext,
    TorchWorkerCheckpointContext,
)
from tributo.util.annotations import PublicAPI


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


def _model(input_features: int, hidden: int) -> object:
    import torch

    if input_features < 1 or hidden < 1:
        raise ValueError("distillation model dimensions must be positive")
    return torch.nn.Sequential(
        torch.nn.Linear(input_features, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )


def _state_digest(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = cast(Any, state[name]).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_teacher_checkpoint(
    checkpoint_context: TorchWorkerCheckpointContext, teacher: object
) -> None:
    if checkpoint_context.checkpoint is None:
        raise AlgorithmExecutionError("student stage requires Teacher Checkpoint")
    import torch

    opener = getattr(checkpoint_context.checkpoint.checkpoint, "as_directory", None)
    if not callable(opener):
        raise AlgorithmExecutionError("Teacher Checkpoint cannot be opened")
    with opener() as directory:
        path = Path(directory) / "model.pt"
        if not path.is_file():
            raise AlgorithmExecutionError("Teacher Checkpoint is missing model.pt")
        cast(torch.nn.Module, teacher).load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )


def _stage_train_loop(
    config: Mapping[str, Any], checkpoint_context: TorchWorkerCheckpointContext
) -> None:
    import ray
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train.torch import get_device, prepare_model

    stage = str(config.get("stage", "teacher"))
    if stage not in {"teacher", "student"}:
        raise AlgorithmConfigurationError("distillation stage is invalid")
    feature_names = tuple(str(name) for name in config.get("feature_names", ()))
    label_name = str(config.get("label_name", "label"))
    if not feature_names:
        raise AlgorithmConfigurationError("distillation feature_names are required")
    shard = train.get_dataset_shard("train")
    raw_batches = list(
        shard.iter_torch_batches(
            batch_size=int(config.get("batch_size", 32)), drop_last=False
        )
    )
    if not raw_batches:
        raise AlgorithmExecutionError(f"distillation {stage} shard is empty")
    first_batch = raw_batches[0]
    if not isinstance(first_batch, Mapping):
        raise AlgorithmExecutionError("distillation batch is not columnar")
    if not feature_names:
        feature_names = tuple(name for name in first_batch if name != label_name)
    batches: list[tuple[object, object]] = []
    for batch in raw_batches:
        if not isinstance(batch, Mapping):
            raise AlgorithmExecutionError("distillation batch is not columnar")
        features = torch.stack([batch[name].float() for name in feature_names], dim=1)
        labels = batch[label_name].float().reshape(-1, 1)
        if (
            not torch.isfinite(features).all()
            or not torch.isfinite(labels).all()
            or not bool(((labels == 0) | (labels == 1)).all())
        ):
            raise AlgorithmExecutionError("distillation batch contains invalid values")
        batches.append((features, labels))
    if not batches:
        raise AlgorithmExecutionError(f"distillation {stage} shard is empty")
    device: Any = get_device()
    model_config = config.get("model", {})
    training = config.get("training", {})
    if not isinstance(model_config, Mapping) or not isinstance(training, Mapping):
        raise AlgorithmConfigurationError(
            "distillation model/training config is invalid"
        )
    input_features = int(model_config["input_features"])
    if len(feature_names) != input_features:
        raise AlgorithmConfigurationError(
            "distillation feature binding does not match model.input_features"
        )
    hidden = int(
        model_config["teacher_hidden"]
        if stage == "teacher"
        else model_config["student_hidden"]
    )
    model = prepare_model(cast(torch.nn.Module, _model(input_features, hidden)))
    teacher = (
        cast(
            torch.nn.Module,
            _model(input_features, int(model_config["teacher_hidden"])),
        ).to(device)
        if stage == "student"
        else None
    )
    rank, world_size = (
        train.get_context().get_world_rank(),
        train.get_context().get_world_size(),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning_rate", 0.01))
    )
    max_gradient_norm = _gradient_clip_norm(training)
    if checkpoint_context.checkpoint is not None:
        if checkpoint_context.source == "stage_dependency":
            if stage != "student" or teacher is None:
                raise AlgorithmExecutionError(
                    "distillation stage dependency targets only the student stage"
                )
            _load_teacher_checkpoint(checkpoint_context, teacher)
        else:
            raise AlgorithmExecutionError(
                f"distillation checkpoint source {checkpoint_context.source!r} is invalid"
            )
    elif stage == "student":
        raise AlgorithmExecutionError("student stage requires Teacher Checkpoint")
    if teacher is not None:
        teacher.eval()
    epochs = int(training.get("epochs", 1))
    alpha = float(training.get("supervised_weight", 0.5))
    if not 0 <= alpha <= 1:
        raise AlgorithmConfigurationError(
            "distillation supervised_weight must be in [0, 1]"
        )
    last_numerator = 0.0
    last_rows = 0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        numerator_total: torch.Tensor | None = None
        rows = 0
        for raw_features, raw_labels in batches:
            features = cast(torch.Tensor, cast(Any, raw_features).to(device))
            labels = cast(torch.Tensor, cast(Any, raw_labels).to(device))
            logits = cast(torch.Tensor, model(features))
            supervised = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="sum"
            )
            numerator: torch.Tensor
            if teacher is None:
                numerator = cast(torch.Tensor, supervised)
            else:
                with torch.no_grad():
                    teacher_probability = torch.sigmoid(teacher(features))
                soft = torch.nn.functional.mse_loss(
                    torch.sigmoid(logits), teacher_probability, reduction="sum"
                )
                numerator = cast(
                    torch.Tensor, alpha * supervised + (1.0 - alpha) * soft
                )
            numerator_total = (
                numerator if numerator_total is None else numerator_total + numerator
            )
            rows += int(labels.shape[0])
        if numerator_total is None:
            raise AlgorithmExecutionError("distillation produced no loss")

        def reduce_normalizer(value: float) -> float:
            tensor = torch.tensor(value, dtype=torch.float64, device=device)
            if dist.is_initialized():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return float(tensor.item())

        def finalize(scale: float) -> None:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
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
    worker = {
        "worker_id": f"distillation-{rank}",
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "shard_id": f"{stage}-{rank}",
        "rows_processed": last_rows,
        "input_rows": {"train": last_rows},
        "batch_count": epochs,
        "collective_steps": epochs,
        "model_state_digest": digest,
        "stage": stage,
    }
    workers: list[object] = [None] * world_size
    if dist.is_initialized():
        dist.all_gather_object(workers, worker)
    else:
        workers[0] = worker
    with tempfile.TemporaryDirectory(
        prefix=f"tributo-distillation-{stage}-"
    ) as directory:
        root = Path(directory)
        torch.save(unwrapped.state_dict(), root / "model.pt")
        (root / "model_config.json").write_text(
            json.dumps(
                {
                    "input_features": int(model_config["input_features"]),
                    "teacher_hidden": int(model_config["teacher_hidden"]),
                    "student_hidden": int(model_config["student_hidden"]),
                    "feature_names": list(feature_names),
                    "stage": stage,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        report_torch_checkpoint(
            {
                "train_loss": reduced_loss,
                f"{stage}_loss": reduced_loss,
                "execution_workers": workers,
                "model_state_digest": digest,
            },
            TorchCheckpointPayloadDraft(root),
            checkpoint_context.stage,
            epochs,
        )


@PublicAPI(stability="alpha")
class DistributedDistillation(RayTorchAdapter):
    """Execute Teacher then Student through Core's component Stage plan."""

    def validate_environment(self, context: TorchRuntimeContext) -> None:
        del context
        try:
            import torch
        except ImportError as exc:
            raise AlgorithmConfigurationError("distillation requires PyTorch") from exc
        if not torch.__version__:
            raise AlgorithmConfigurationError("distillation environment is invalid")

    def bind_datasets(
        self, datasets: Mapping[str, object], context: TorchStageContext
    ) -> Mapping[str, object]:
        del context
        if set(datasets) != {"train"}:
            raise AlgorithmConfigurationError("distillation requires train Dataset")
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
                "distillation Stage context is missing the train binding"
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
                "distillation train binding has an invalid typed layout"
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
            raise AlgorithmExecutionError("distillation result has no checkpoint")
        return checkpoint

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan(
            {
                "train_loss": "sum_count",
                "teacher_loss": "sum_count",
                "student_loss": "sum_count",
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
                    "name": "student-model",
                    "format": "safetensors",
                    "exporter_id": "torch-safetensors-v1",
                },
                {
                    "name": "student-inference",
                    "format": "onnx",
                    "exporter_id": "torch-onnx-v1",
                    "options": {"dynamo": False},
                },
            ),
            roles={"model": "student-model", "inference": "student-inference"},
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
            raise AlgorithmExecutionError("distillation checkpoint cannot be opened")
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
                    f"distillation checkpoint is missing payloads: {missing_payloads}"
                )
            config = json.loads(
                (root / "model_config.json").read_text(encoding="utf-8")
            )
            model = cast(
                torch.nn.Module,
                _model(int(config["input_features"]), int(config["student_hidden"])),
            )
            try:
                model.load_state_dict(
                    torch.load(root / "model.pt", map_location="cpu", weights_only=True)
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise AlgorithmExecutionError(
                    "distillation model payload is incompatible"
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
                    "component_stages": ["teacher", "student"],
                    "producer_distribution": "tributo-algorithms-multistage-torch",
                },
                source_fingerprint=_state_digest(
                    cast(Mapping[str, object], model.state_dict())
                ),
                checkpoint_contract=ExportCheckpointV1(
                    trainer_type="teacher_student_distillation",
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


def create_algorithm(
    *,
    plan: object | None = None,
    implementation: object | None = None,
    artifacts: tuple[object, ...] = (),
) -> DistributedDistillation:
    del plan, artifacts
    if implementation is not DistributedDistillation:
        raise AlgorithmConfigurationError("distillation implementation drifted")
    return DistributedDistillation()


__all__ = ["DistributedDistillation", "create_algorithm"]
