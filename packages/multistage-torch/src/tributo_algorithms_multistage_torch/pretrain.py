"""Framework-native self-supervised pretraining followed by finetuning."""

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

STAGES = ("pretrain", "finetune")


def _encoder(input_features: int, hidden_features: int) -> Any:
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(input_features, hidden_features),
        torch.nn.ReLU(),
    )


def _pretrain_model(input_features: int, hidden_features: int) -> object:
    import torch

    class PretrainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _encoder(input_features, hidden_features)
            self.decoder = torch.nn.Linear(hidden_features, input_features)

        def forward(self, features: object) -> object:
            return self.decoder(self.encoder(cast(Any, features)))

    return PretrainModel()


def _finetune_model(input_features: int, hidden_features: int) -> object:
    import torch

    class FinetuneModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _encoder(input_features, hidden_features)
            self.classifier = torch.nn.Linear(hidden_features, 1)

        def forward(self, features: object) -> object:
            return self.classifier(self.encoder(cast(Any, features)))

    return FinetuneModel()


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
        raise AlgorithmExecutionError(f"{stage} shard is empty")
    device = get_device()
    features = torch.cat(feature_batches, dim=0).to(device)
    labels = torch.cat(label_batches, dim=0).to(device)
    input_features = int(config["input_features"])
    hidden_features = int(config["hidden_features"])
    if stage == "pretrain":
        raw_model = cast(
            torch.nn.Module,
            _pretrain_model(input_features, hidden_features),
        )
    else:
        raw_model = cast(
            torch.nn.Module,
            _finetune_model(input_features, hidden_features),
        )
        checkpoint = config.get("pretrain_checkpoint")
        if checkpoint is None:
            raise AlgorithmExecutionError(
                "finetune stage requires pretraining checkpoint"
            )
        with checkpoint.as_directory() as directory:
            cast(Any, raw_model).encoder.load_state_dict(
                torch.load(
                    Path(directory) / "encoder.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
    model = prepare_model(raw_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    epochs = int(
        config["pretrain_epochs"] if stage == "pretrain" else config["finetune_epochs"]
    )
    last_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss = (
            torch.nn.functional.mse_loss(output, features)
            if stage == "pretrain"
            else torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        )
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
    rows = int(labels.shape[0])
    worker = {
        "worker_id": str(runtime.get_worker_id()),
        "node_id": str(runtime.get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "shard_id": hashlib.sha256(
            f"{config['binding_digest']}:{stage}:{rank}/{world_size}".encode("ascii")
        ).hexdigest(),
        "rows_processed": rows,
        "input_rows": {"train": rows},
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
        raise AlgorithmExecutionError(f"{stage} Worker evidence is incomplete")
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix=f"tributo-{stage}-") as directory:
            root = Path(directory)
            if stage == "pretrain":
                torch.save(
                    cast(Any, unwrapped).encoder.state_dict(), root / "encoder.pt"
                )
            else:
                torch.save(dict(state), root / "model.pt")
            (root / "model_config.json").write_text(
                json.dumps(
                    {
                        "input_features": input_features,
                        "hidden_features": hidden_features,
                        "feature_names": list(feature_names),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            train.report(
                {
                    "stage": stage,
                    "loss": last_loss,
                    "execution_workers": cast(list[dict[str, object]], workers),
                    "model_state_digest": model_digest,
                },
                checkpoint=Checkpoint.from_directory(root),
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
class PretrainFinetuneResult:
    checkpoint: object
    metrics: Mapping[str, object]
    stages: Mapping[str, Mapping[str, object]]
    composition_digest: str


class _PretrainFinetuneDriver:
    def __init__(
        self,
        *,
        plan: ResolvedAlgorithmPlan,
        datasets: Mapping[str, object],
    ) -> None:
        self.plan = plan
        self.datasets = dict(datasets)

    def _trainer(self, stage: str, *, resume: object | None = None) -> object:
        from ray.train import DataConfig, RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer

        config = self.plan.algorithm_config
        model = cast(Mapping[str, Any], config["model"])
        training = cast(Mapping[str, Any], config["training"])
        ray_config = cast(Mapping[str, Any], config["ray"])
        binding = self.plan.primary_input_binding
        return TorchTrainer(
            train_loop_per_worker=_stage_train_loop,
            train_loop_config={
                "stage": stage,
                "feature_names": list(binding.feature_names),
                "label_name": binding.label_name,
                "binding_digest": self.plan.primary_input_descriptor.binding_digest,
                "input_features": int(model["input_features"]),
                "hidden_features": int(model["hidden_features"]),
                "pretrain_epochs": int(training.get("pretrain_epochs", 1)),
                "finetune_epochs": int(training.get("finetune_epochs", 1)),
                "batch_size": int(training.get("batch_size", 32)),
                "learning_rate": float(training.get("learning_rate", 0.01)),
                **(
                    {"pretrain_checkpoint": resume}
                    if stage == "finetune" and resume is not None
                    else {}
                ),
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
                name=f"tributo-pretrain-finetune-{stage}",
                storage_path=str(ray_config["storage_path"]),
            ),
        )

    def fit(self) -> PretrainFinetuneResult:
        rows = int(cast(Any, self.datasets["train"]).count())
        pretrain_result = cast(Any, self._trainer("pretrain")).fit()
        if pretrain_result.checkpoint is None:
            raise AlgorithmExecutionError("pretraining did not checkpoint")
        finetune_result = cast(
            Any,
            self._trainer(
                "finetune",
                resume=pretrain_result.checkpoint,
            ),
        ).fit()
        if finetune_result.checkpoint is None:
            raise AlgorithmExecutionError("finetuning did not checkpoint")
        evidence: dict[str, Mapping[str, object]] = {}
        digests: dict[str, str] = {}
        for stage, result in (
            ("pretrain", pretrain_result),
            ("finetune", finetune_result),
        ):
            workers = result.metrics.get("execution_workers")
            digest = result.metrics.get("model_state_digest")
            if not isinstance(workers, list) or not isinstance(digest, str):
                raise AlgorithmExecutionError(f"{stage} evidence is malformed")
            digests[stage] = digest
            evidence[stage] = {
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
        return PretrainFinetuneResult(
            checkpoint=finetune_result.checkpoint,
            metrics=finetune_result.metrics,
            stages=evidence,
            composition_digest=composition_digest,
        )


class DistributedPretrainFinetune(FrameworkNativeAlgorithm):
    """Execute self-supervised pretraining and supervised finetuning."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan

    def validate_environment(self) -> None:
        try:
            import torch
            from ray.train.torch import TorchTrainer
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "pretrain-finetune requires Ray Train and PyTorch"
            ) from exc
        if not torch.__version__ or TorchTrainer is None:
            raise AlgorithmConfigurationError(
                "pretrain-finetune environment is invalid"
            )

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        if set(datasets) != {"train"}:
            raise AlgorithmConfigurationError(
                "pretrain-finetune requires train Dataset"
            )
        return dict(datasets)

    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        del config
        return _PretrainFinetuneDriver(plan=self.plan, datasets=datasets)

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        if not isinstance(result, PretrainFinetuneResult):
            raise AlgorithmExecutionError("pretrain-finetune result is invalid")
        return {
            "stages": dict(result.stages),
            "composition_digest": result.composition_digest,
        }

    def checkpoint_source(self, result: object) -> object:
        if not isinstance(result, PretrainFinetuneResult):
            raise AlgorithmExecutionError("pretrain-finetune checkpoint is invalid")
        return result.checkpoint


def create_pretrain_finetune_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedPretrainFinetune:
    del artifacts
    if implementation is not DistributedPretrainFinetune:
        raise AlgorithmConfigurationError("pretrain-finetune implementation drifted")
    return DistributedPretrainFinetune(plan)


def export_pretrain_finetune_result(
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

    if not isinstance(result, PretrainFinetuneResult):
        raise AlgorithmExecutionError("pretrain-finetune export result is invalid")
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError(
            "pretrain-finetune output.bundle_uri is required"
        )
    with cast(Any, checkpoint).as_directory() as directory:
        root = Path(directory)
        model_config = json.loads(
            (root / "model_config.json").read_text(encoding="utf-8")
        )
        model = cast(
            torch.nn.Module,
            _finetune_model(
                int(model_config["input_features"]),
                int(model_config["hidden_features"]),
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
            feature_schema={"feature_names": model_config["feature_names"]},
            metadata={
                "framework": "pytorch",
                "task_type": "binary_classification",
                "component_stages": list(STAGES),
                "composition_digest": result.composition_digest,
                "producer_distribution": "tributo-algorithms-multistage-torch",
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
                        name="finetuned-model",
                        format="safetensors",
                        exporter_id="torch-safetensors-v1",
                    )
                ],
                roles={"model": "finetuned-model"},
            ),
            tributo_version=importlib.metadata.version("tributo"),
        )
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics={
            "finetune_loss": float(cast(Any, result.metrics.get("loss", 0.0))),
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
    "DistributedPretrainFinetune",
    "PretrainFinetuneResult",
    "STAGES",
    "create_pretrain_finetune_algorithm",
    "export_pretrain_finetune_result",
]
