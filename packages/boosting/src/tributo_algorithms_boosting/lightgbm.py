"""Ray Train framework-native distributed LightGBM implementation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import FrameworkNativeAlgorithm


class _EvidenceCollector:
    def __init__(self) -> None:
        self.records: dict[int, dict[str, object]] = {}

    def record(self, value: dict[str, object]) -> None:
        rank = value.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("LightGBM evidence rank must be an integer")
        if rank in self.records:
            raise ValueError(f"duplicate LightGBM evidence rank: {rank}")
        self.records[rank] = dict(value)

    def snapshot(self) -> list[dict[str, object]]:
        return [self.records[rank] for rank in sorted(self.records)]


def _model_task(model: Mapping[str, Any]) -> tuple[str, int]:
    """Resolve the bounded export task from the LightGBM model config."""
    requested = str(model.get("task", "")).lower()
    objective = str(model.get("objective", "")).lower()
    if requested not in {"", "classification", "regression"}:
        raise AlgorithmConfigurationError(
            "LightGBM model.task must be classification or regression"
        )
    objective_is_classification = objective.startswith(
        ("binary", "multiclass", "multiclassova", "multiclassova2")
    )
    if requested == "classification" and objective and not objective_is_classification:
        raise AlgorithmConfigurationError(
            "LightGBM classification task conflicts with model.objective"
        )
    if requested == "regression" and objective_is_classification:
        raise AlgorithmConfigurationError(
            "LightGBM regression task conflicts with model.objective"
        )
    raw_class_count = model.get("num_class")
    if objective.startswith("multiclass") and raw_class_count is None:
        raise AlgorithmConfigurationError(
            "multiclass LightGBM requires model.num_class"
        )
    if (
        objective == "binary"
        and raw_class_count is not None
        and int(raw_class_count) != 2
    ):
        raise AlgorithmConfigurationError("binary LightGBM requires model.num_class=2")
    classification = requested == "classification" or objective.startswith(
        ("binary", "multiclass", "multiclassova", "multiclassova2")
    )
    if not classification:
        return "regression", 1
    try:
        class_count = int(model.get("num_class", 2))
    except (TypeError, ValueError) as exc:
        raise AlgorithmConfigurationError(
            "LightGBM model.num_class must be an integer"
        ) from exc
    if class_count < 2:
        raise AlgorithmConfigurationError(
            "LightGBM classification requires at least two classes"
        )
    return "classification", class_count


def _training_params(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return LightGBM params without Tributo-only control keys."""
    params = dict(model)
    task, class_count = _model_task(model)
    params.pop("task", None)
    if task == "classification":
        params.setdefault("objective", "multiclass" if class_count > 2 else "binary")
        if class_count > 2:
            params.setdefault("num_class", class_count)
    else:
        params.setdefault("objective", "regression")
    params.setdefault("num_threads", 1)
    return params


def _train_loop(config: dict[str, Any]) -> None:
    import lightgbm
    import numpy as np
    import ray
    from ray import train
    from ray.train import Checkpoint
    from ray.train.lightgbm import get_network_params

    shard = train.get_dataset_shard("train")
    # The native LightGBM API consumes pandas, so this materializes one
    # complete Ray shard per worker; external-memory training is out of scope.
    frame = shard.materialize().to_pandas()
    feature_names = list(config["feature_names"])
    label_name = str(config["label_name"])
    if frame.empty:
        raise AlgorithmExecutionError("LightGBM worker shard is empty")
    train_set = lightgbm.Dataset(
        frame[feature_names],
        label=frame[label_name],
        free_raw_data=False,
    )
    params = dict(config["params"])
    params.update(
        {
            "tree_learner": "data_parallel",
            "pre_partition": True,
            **get_network_params(),
        }
    )
    booster = lightgbm.train(
        params,
        train_set,
        num_boost_round=int(config["num_boost_round"]),
        valid_sets=[train_set],
        valid_names=["train"],
    )
    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    probe = np.linspace(
        -1.0,
        1.0,
        8 * len(feature_names),
        dtype=np.float64,
    ).reshape(8, len(feature_names))
    predictions = np.asarray(booster.predict(probe, raw_score=True), dtype=np.float64)
    digest = hashlib.sha256(
        predictions.tobytes()
        + str(booster.current_iteration()).encode("ascii")
        + str(booster.num_feature()).encode("ascii")
    ).hexdigest()
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
        "rows_processed": int(frame.shape[0]),
        "input_rows": {"train": int(frame.shape[0])},
        "batch_count": int(config["num_boost_round"]),
        "collective_steps": int(config["num_boost_round"]),
        "model_state_digest": digest,
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
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix="tributo-lightgbm-") as directory:
            root = Path(directory)
            booster.save_model(str(root / "model.txt"))
            (root / "feature_names.json").write_text(
                json.dumps(feature_names), encoding="utf-8"
            )
            (root / "model_config.json").write_text(
                json.dumps(
                    {
                        "task": config["task"],
                        "class_count": config["class_count"],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            train.report(
                {"model_state_digest": digest},
                checkpoint=Checkpoint.from_directory(root),
            )
    else:
        train.report({"model_state_digest": digest})


class DistributedLightGBM(FrameworkNativeAlgorithm):
    """Train one synchronized LightGBM Booster through Ray Train."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        self.collector: Any | None = None
        if plan.runtime.resume_from is not None:
            raise AlgorithmConfigurationError(
                "official LightGBM resume requires a separate recovery gate"
            )

    def validate_environment(self) -> None:
        try:
            import lightgbm
            from ray.train.lightgbm import LightGBMTrainer
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "official LightGBM requires LightGBM and Ray Train"
            ) from exc
        if not lightgbm.__version__ or LightGBMTrainer is None:
            raise AlgorithmConfigurationError("LightGBM environment is invalid")

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        if len(datasets) != 1:
            raise AlgorithmConfigurationError("LightGBM requires one train Dataset")
        dataset = next(iter(datasets.values()))
        binding = self.plan.primary_input_binding
        data = cast(Mapping[str, Any], self.plan.algorithm_config.get("data", {}))
        configured_features = tuple(data.get("feature_columns", ()))
        features = configured_features or binding.feature_names
        if tuple(features) != binding.feature_names:
            raise AlgorithmConfigurationError(
                "LightGBM data.feature_columns must match InputBinding"
            )
        if data.get("label_col") != binding.label_name:
            raise AlgorithmConfigurationError(
                "LightGBM data.label_col must match InputBinding label"
            )
        return {
            "train": cast(Any, dataset).select_columns(
                [*features, cast(str, binding.label_name)]
            )
        }

    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        import ray
        from ray.train import RunConfig, ScalingConfig
        from ray.train.lightgbm import LightGBMTrainer

        data = cast(Mapping[str, Any], config.get("data", {}))
        model = dict(cast(Mapping[str, Any], config.get("model", {})))
        task, class_count = _model_task(model)
        params = _training_params(model)
        training = cast(Mapping[str, Any], config.get("training", {}))
        ray_config = cast(Mapping[str, Any], config.get("ray", {}))
        binding = self.plan.primary_input_binding
        collector_type = ray.remote(_EvidenceCollector).options(num_cpus=0)
        self.collector = collector_type.remote()
        return LightGBMTrainer(
            train_loop_per_worker=_train_loop,
            train_loop_config={
                "feature_names": list(binding.feature_names),
                "label_name": str(data["label_col"]),
                "params": params,
                "task": task,
                "class_count": class_count,
                "num_boost_round": int(training.get("num_rounds", 10)),
                "evidence_actor": self.collector,
                "binding_digest": self.plan.primary_input_descriptor.binding_digest,
            },
            scaling_config=ScalingConfig(
                num_workers=self.plan.runtime.worker_count,
                use_gpu=self.plan.runtime.num_gpus > 0,
                placement_strategy="SPREAD",
                resources_per_worker={
                    "CPU": self.plan.runtime.num_cpus,
                    "GPU": self.plan.runtime.num_gpus,
                    **dict(self.plan.runtime.custom_resources),
                },
            ),
            datasets=cast(dict[str, Any], dict(datasets)),
            run_config=RunConfig(
                name="tributo-official-lightgbm",
                storage_path=str(ray_config["storage_path"]),
            ),
        )

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        import ray

        del result
        if self.collector is None:
            raise AlgorithmExecutionError("LightGBM evidence collector is missing")
        workers = ray.get(self.collector.snapshot.remote())
        ray.kill(self.collector, no_restart=True)
        self.collector = None
        if len(workers) != self.plan.runtime.worker_count:
            raise AlgorithmExecutionError("LightGBM did not report every worker")
        digests = {str(item.get("model_state_digest")) for item in workers}
        if len(digests) != 1:
            raise AlgorithmExecutionError("LightGBM Booster state diverged")
        return {
            "workers": workers,
            "state": {
                "coordination": "framework_native",
                "synchronized": True,
                "bounded": True,
                "global_model_digest": next(iter(digests)),
                "details": {"framework": "lightgbm", "collective": "tcp"},
            },
            "input_complete": True,
        }

    def checkpoint_source(self, result: object) -> object:
        checkpoint = getattr(result, "checkpoint", None)
        if checkpoint is None:
            raise AlgorithmExecutionError("LightGBM result has no checkpoint")
        return checkpoint


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedLightGBM:
    """Construct the descriptor-selected LightGBM implementation."""
    del artifacts
    if implementation is not DistributedLightGBM:
        raise AlgorithmConfigurationError("LightGBM implementation drifted")
    return DistributedLightGBM(plan)


def export_result(
    *,
    result: object,
    checkpoint: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    """Export a LightGBM checkpoint to ONNX and a validated Bundle."""
    import lightgbm
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    from tributo.exporting.models import (
        BundleOutputConfig,
        CheckpointField,
        ExportCheckpointV1,
        ExportSource,
        ExportTarget,
    )
    from tributo.exporting.service import BundleExportService

    del result
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError("LightGBM output.bundle_uri is required")
    data = cast(Mapping[str, Any], plan.algorithm_config.get("data", {}))
    binding = plan.primary_input_binding
    with cast(Any, checkpoint).as_directory() as directory:
        root = Path(directory)
        booster = lightgbm.Booster(model_file=str(root / "model.txt"))
        try:
            raw_feature_names = json.loads(
                (root / "feature_names.json").read_text(encoding="utf-8")
            )
            model_config = json.loads(
                (root / "model_config.json").read_text(encoding="utf-8")
            )
            if (
                not isinstance(raw_feature_names, list)
                or not all(isinstance(name, str) and name for name in raw_feature_names)
                or not isinstance(model_config, Mapping)
            ):
                raise ValueError("LightGBM checkpoint metadata is invalid")
            feature_names = tuple(raw_feature_names)
            task = str(model_config.get("task", "regression"))
            class_count = int(model_config.get("class_count", 1))
            if task not in {"classification", "regression"}:
                raise ValueError("LightGBM checkpoint task is invalid")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AlgorithmExecutionError(
                "LightGBM checkpoint metadata is invalid"
            ) from exc
        try:
            export_model: Any = booster
            output_schema: tuple[Any, ...]
            if task == "classification":
                if class_count < 2:
                    raise AlgorithmConfigurationError(
                        "LightGBM checkpoint has an invalid class count"
                    )
                export_model = lightgbm.LGBMClassifier(
                    objective="multiclass" if class_count > 2 else "binary"
                )
                export_model._Booster = booster
                export_model.fitted_ = True
                export_model._classes = np.arange(class_count)
                export_model._n_classes = class_count
                export_model._objective = "multiclass" if class_count > 2 else "binary"
                export_model.n_features_in_ = len(feature_names)
                output_schema = (
                    CheckpointField(name="label", dtype="int64", shape=("batch",)),
                    CheckpointField(
                        name="probabilities",
                        dtype="float32",
                        shape=("batch", class_count),
                    ),
                )
            elif task == "regression":
                output_schema = (
                    CheckpointField(
                        name="variable", dtype="float32", shape=("batch", 1)
                    ),
                )
            else:
                raise AlgorithmExecutionError("LightGBM checkpoint task is invalid")
            converted = convert_lightgbm(
                export_model,
                initial_types=[
                    ("float_input", FloatTensorType([None, len(feature_names)]))
                ],
                zipmap=False,
                target_opset=15,
            )
            for graph_output in converted.graph.output:
                dimensions = graph_output.type.tensor_type.shape.dim
                if not dimensions:
                    raise AlgorithmExecutionError(
                        "LightGBM ONNX output omitted its batch dimension"
                    )
                batch_dimension = dimensions[0]
                batch_dimension.ClearField("dim_value")
                batch_dimension.dim_param = "batch"
            converted.graph.name = "tributo-lightgbm"
            converted.doc_string = ""
            for node in converted.graph.node:
                node.doc_string = ""
            payload = converted.SerializeToString()
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"LightGBM ONNX export failed: {type(exc).__name__}"
            ) from exc
    artifact = ArtifactDraft.from_payload(
        name="model", kind="model", format="application/onnx", payload=payload
    )
    source = ExportSource(
        source_kind="prebuilt_onnx",
        model_object=payload,
        feature_schema={"feature_names": list(feature_names)},
        metadata={
            "framework": "lightgbm",
            "framework_versions": {"lightgbm": lightgbm.__version__},
            "task_type": task,
            "producer_distribution": "tributo-algorithms-boosting",
        },
        source_fingerprint=artifact.sha256,
        checkpoint_contract=ExportCheckpointV1(
            trainer_type="lightgbm",
            architecture_id=plan.resolution.algorithm,
            input_schema=(
                CheckpointField(
                    name="float_input",
                    dtype="float32",
                    shape=("batch", len(feature_names)),
                ),
            ),
            output_schema=output_schema,
            task_type=task,
            framework="lightgbm",
            framework_version=lightgbm.__version__,
        ),
    )
    bundle = BundleExportService().export_bundle(
        source,
        BundleOutputConfig(
            bundle_uri=cast(str, output["bundle_uri"]),
            request_id=run_id,
            run_id=run_id,
            targets=[
                ExportTarget(
                    name="onnx-model",
                    format="onnx",
                    exporter_id="prebuilt-onnx-v1",
                )
            ],
            roles={"inference": "onnx-model"},
        ),
    )
    return AlgorithmExecutionResult(
        status="succeeded",
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
            "feature_names": list(binding.feature_names),
            "label_col": data.get("label_col"),
        },
        artifacts=(artifact,),
    )


__all__ = ["DistributedLightGBM", "create_algorithm", "export_result"]
