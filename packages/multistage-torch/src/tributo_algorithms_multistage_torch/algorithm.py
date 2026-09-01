"""Framework-native finite Teacher-to-Student distillation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
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


def _model(input_features: int, hidden: int) -> object:
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_features, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )


def _state_digest(state: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = cast(Any, state[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _stage_train_loop(config: dict[str, Any]) -> None:
    import ray
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import get_device, prepare_model

    stage = str(config["stage"])
    feature_names = tuple(str(name) for name in config["feature_names"])
    label_name = str(config["label_name"])
    shard = train.get_dataset_shard("train")
    feature_batches = []
    label_batches = []
    for batch in shard.iter_torch_batches(
        batch_size=int(config["batch_size"]),
        prefetch_batches=0,
        dtypes=torch.float32,
    ):
        feature_batches.append(
            torch.stack([batch[name].float() for name in feature_names], dim=1)
        )
        label_batches.append(batch[label_name].float().reshape(-1, 1))
    if not feature_batches:
        raise AlgorithmExecutionError(f"distillation {stage} shard is empty")
    device = get_device()
    features = torch.cat(feature_batches, dim=0).to(device)
    labels = torch.cat(label_batches, dim=0).to(device)
    input_features = int(config["input_features"])
    hidden = int(
        config["teacher_hidden"] if stage == "teacher" else config["student_hidden"]
    )
    raw_model = cast(torch.nn.Module, _model(input_features, hidden))
    teacher: torch.nn.Module | None = None
    if stage == "student":
        checkpoint = config.get("teacher_checkpoint")
        if checkpoint is None:
            raise AlgorithmExecutionError("student stage requires Teacher checkpoint")
        teacher = cast(
            torch.nn.Module,
            _model(input_features, int(config["teacher_hidden"])),
        ).to(device)
        with checkpoint.as_directory() as directory:
            teacher.load_state_dict(
                torch.load(
                    Path(directory) / "teacher.pt",
                    map_location=device,
                    weights_only=True,
                )
            )
        teacher.eval()
    model = prepare_model(raw_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    epochs = int(config["epochs"])
    last_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        supervised = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels
        )
        if teacher is None:
            loss = supervised
        else:
            with torch.no_grad():
                teacher_probability = torch.sigmoid(teacher(features))
            soft_loss = torch.nn.functional.mse_loss(
                torch.sigmoid(logits), teacher_probability
            )
            alpha = float(config["supervised_weight"])
            loss = alpha * supervised + (1.0 - alpha) * soft_loss
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    if dist.is_initialized():
        dist.barrier()
    unwrapped = cast(
        torch.nn.Module,
        model.module if hasattr(model, "module") else model,
    )
    state = cast(Mapping[str, object], unwrapped.state_dict())
    model_digest = _state_digest(state)
    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    runtime = ray.get_runtime_context()
    assigned = runtime.get_assigned_resources()
    worker = {
        "worker_id": str(runtime.get_worker_id()),
        "node_id": str(runtime.get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "shard_id": hashlib.sha256(
            f"{config['binding_digest']}:{stage}:{rank}/{world_size}".encode("ascii")
        ).hexdigest(),
        "rows_processed": int(labels.shape[0]),
        "input_rows": {"train": int(labels.shape[0])},
        "batch_count": epochs,
        "collective_steps": epochs,
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
        raise AlgorithmExecutionError("distillation Worker evidence is incomplete")
    checkpoint_value = None
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix=f"tributo-{stage}-") as directory:
            root = Path(directory)
            torch.save(
                dict(state), root / ("teacher.pt" if stage == "teacher" else "model.pt")
            )
            (root / "model_config.json").write_text(
                json.dumps(
                    {
                        "input_features": input_features,
                        "teacher_hidden": int(config["teacher_hidden"]),
                        "student_hidden": int(config["student_hidden"]),
                        "feature_names": list(feature_names),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            checkpoint_value = Checkpoint.from_directory(root)
            train.report(
                {
                    "stage": stage,
                    "loss": last_loss,
                    "execution_workers": cast(list[dict[str, object]], workers),
                    "model_state_digest": model_digest,
                },
                checkpoint=checkpoint_value,
            )
    else:
        train.report(
            {
                "stage": stage,
                "loss": last_loss,
                "execution_workers": cast(list[dict[str, object]], workers),
                "model_state_digest": model_digest,
            }
        )


@dataclass(frozen=True)
class DistillationResult:
    checkpoint: object
    metrics: Mapping[str, object]
    stages: Mapping[str, Mapping[str, object]]
    composition_digest: str


def _distillation_scaling_config(plan: ResolvedAlgorithmPlan) -> "ScalingConfig":
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


class _DistillationDriver:
    def __init__(
        self,
        *,
        plan: ResolvedAlgorithmPlan,
        datasets: Mapping[str, object],
    ) -> None:
        self.plan = plan
        self.datasets = dict(datasets)

    def _trainer(self, stage: str, *, resume: object | None = None) -> object:
        from ray.train import DataConfig, RunConfig
        from ray.train.torch import TorchTrainer

        config = self.plan.algorithm_config
        model = cast(Mapping[str, Any], config.get("model", {}))
        training = cast(Mapping[str, Any], config.get("training", {}))
        ray_config = cast(Mapping[str, Any], config.get("ray", {}))
        binding = self.plan.primary_input_binding
        return TorchTrainer(
            train_loop_per_worker=_stage_train_loop,
            train_loop_config={
                "stage": stage,
                "feature_names": list(binding.feature_names),
                "label_name": binding.label_name,
                "binding_digest": self.plan.primary_input_descriptor.binding_digest,
                "input_features": int(model["input_features"]),
                "teacher_hidden": int(model["teacher_hidden"]),
                "student_hidden": int(model["student_hidden"]),
                "epochs": int(training.get("epochs", 1)),
                "batch_size": int(training.get("batch_size", 32)),
                "learning_rate": float(training.get("learning_rate", 0.01)),
                "supervised_weight": float(training.get("supervised_weight", 0.5)),
                **(
                    {"teacher_checkpoint": resume}
                    if stage == "student" and resume is not None
                    else {}
                ),
            },
            scaling_config=_distillation_scaling_config(self.plan),
            datasets=cast(dict[str, Any], self.datasets),
            dataset_config=DataConfig(datasets_to_split=["train"]),
            run_config=RunConfig(
                name=f"tributo-distillation-{stage}",
                storage_path=str(ray_config["storage_path"]),
            ),
        )

    def fit(self) -> DistillationResult:
        train_dataset = self.datasets["train"]
        rows = int(cast(Any, train_dataset).count())
        teacher_result = cast(Any, self._trainer("teacher")).fit()
        if teacher_result.checkpoint is None:
            raise AlgorithmExecutionError("Teacher stage did not checkpoint")
        student_result = cast(
            Any,
            self._trainer("student", resume=teacher_result.checkpoint),
        ).fit()
        if student_result.checkpoint is None:
            raise AlgorithmExecutionError("Student stage did not checkpoint")
        stage_evidence: dict[str, Mapping[str, object]] = {}
        digests = {}
        for stage, result in (
            ("teacher", teacher_result),
            ("student", student_result),
        ):
            metrics = result.metrics
            workers = metrics.get("execution_workers")
            digest = metrics.get("model_state_digest")
            if not isinstance(workers, list) or not isinstance(digest, str):
                raise AlgorithmExecutionError(f"{stage} stage evidence is malformed")
            digests[stage] = digest
            stage_evidence[stage] = {
                "workers": workers,
                "state": {
                    "coordination": "framework_native",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": digest,
                    "details": {"framework": "pytorch-ddp", "stage": stage},
                },
                "input_complete": True,
                "expected_training_rows": rows,
            }
        composition_digest = hashlib.sha256(
            json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return DistillationResult(
            checkpoint=student_result.checkpoint,
            metrics=student_result.metrics,
            stages=stage_evidence,
            composition_digest=composition_digest,
        )


class DistributedDistillation(FrameworkNativeAlgorithm):
    """Execute Teacher and Student Ray Train stages in one bounded invocation."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        if plan.runtime.resume_from is not None:
            raise AlgorithmConfigurationError(
                "distillation resume requires a separate failure-recovery gate"
            )

    def validate_environment(self) -> None:
        try:
            import torch
            from ray.train.torch import TorchTrainer
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "distillation requires Ray Train and PyTorch"
            ) from exc
        if not torch.__version__ or TorchTrainer is None:
            raise AlgorithmConfigurationError("distillation environment is invalid")

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        if set(datasets) != {"train"}:
            raise AlgorithmConfigurationError("distillation requires train Dataset")
        return dict(datasets)

    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        del config
        return _DistillationDriver(plan=self.plan, datasets=datasets)

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        if not isinstance(result, DistillationResult):
            raise AlgorithmExecutionError("distillation returned an invalid result")
        return {
            "stages": dict(result.stages),
            "composition_digest": result.composition_digest,
        }

    def checkpoint_source(self, result: object) -> object:
        if not isinstance(result, DistillationResult):
            raise AlgorithmExecutionError("distillation checkpoint result is invalid")
        return result.checkpoint


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedDistillation:
    del artifacts
    if implementation is not DistributedDistillation:
        raise AlgorithmConfigurationError("distillation implementation drifted")
    return DistributedDistillation(plan)


def export_result(
    *,
    result: object,
    checkpoint: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    import importlib.metadata

    import torch
    from tributo.exporting.models import (
        BundleOutputConfig,
        CheckpointField,
        ExportCheckpointV1,
        ExportSource,
        ExportTarget,
    )
    from tributo.exporting.service import BundleExportService

    if not isinstance(result, DistillationResult):
        raise AlgorithmExecutionError("distillation export result is invalid")
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError("distillation output.bundle_uri is required")
    with cast(Any, checkpoint).as_directory() as directory:
        root = Path(directory)
        model_config = json.loads(
            (root / "model_config.json").read_text(encoding="utf-8")
        )
        model = cast(
            torch.nn.Module,
            _model(
                int(model_config["input_features"]),
                int(model_config["student_hidden"]),
            ),
        )
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        fingerprint = _state_digest(cast(Mapping[str, object], model.state_dict()))
        input_features = int(model_config["input_features"])
        source = ExportSource(
            source_kind="torch_module",
            model_object=model,
            architecture_id=plan.resolution.implementation_id,
            model_config_data=model_config,
            feature_schema={"input_names": ["float_input"]},
            sample_inputs={
                "float_input": torch.zeros((2, input_features), dtype=torch.float32)
            },
            metadata={
                "framework": "pytorch",
                "task_type": "binary_classification",
                "component_stages": ["teacher", "student"],
                "composition_digest": result.composition_digest,
                "producer_distribution": "tributo-algorithms-multistage-torch",
            },
            source_fingerprint=fingerprint,
            checkpoint_contract=ExportCheckpointV1(
                trainer_type="teacher_student_distillation",
                architecture_id=plan.resolution.implementation_id,
                input_schema=(
                    CheckpointField(
                        name="float_input",
                        dtype="float32",
                        shape=("batch", input_features),
                    ),
                ),
                output_schema=(
                    CheckpointField(
                        name="output",
                        dtype="float32",
                        shape=("batch", 1),
                    ),
                ),
                preprocessing={"type": "ordered_features"},
                task_type="binary_classification",
                framework="pytorch",
                framework_version=torch.__version__,
            ),
        )
        bundle = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=str(output["bundle_uri"]),
                request_id=run_id,
                run_id=run_id,
                targets=[
                    ExportTarget(
                        name="student-model",
                        format="safetensors",
                        exporter_id="torch-safetensors-v1",
                    ),
                    ExportTarget(
                        name="student-inference",
                        format="onnx",
                        exporter_id="torch-onnx-v1",
                        options={"dynamo": False},
                    ),
                ],
                roles={
                    "model": "student-model",
                    "inference": "student-inference",
                },
            ),
            tributo_version=importlib.metadata.version("tributo"),
        )
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics={
            "student_loss": float(cast(Any, result.metrics.get("loss", 0.0))),
            "stage_count": 2,
        },
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
            "composition_digest": result.composition_digest,
        },
    )


__all__ = [
    "DistributedDistillation",
    "DistillationResult",
    "create_algorithm",
    "export_result",
]
