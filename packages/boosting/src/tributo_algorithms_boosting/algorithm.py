"""Public Ray Train/XGBoost distributed implementation."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import FrameworkNativeAlgorithm

from tributo_algorithms_boosting.data_config import CompleteCoverageDataConfig

_XGBOOST_BATCH_ROWS = 65_536


class _EvidenceCollector:
    def __init__(self) -> None:
        self.records: dict[int, dict[str, object]] = {}

    def record(self, value: dict[str, object]) -> None:
        rank = value.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("XGBoost evidence rank must be an integer")
        if rank in self.records:
            raise ValueError(f"duplicate XGBoost evidence rank: {rank}")
        self.records[rank] = dict(value)

    def snapshot(self) -> list[dict[str, object]]:
        return [self.records[rank] for rank in sorted(self.records)]


def _train_loop(config: dict[str, Any]) -> None:
    import gc
    import json

    import ray
    import xgboost
    from ray import train
    from ray.train import Checkpoint

    feature_names = list(config["feature_names"])
    label_name = str(config["label_name"])
    rounds = config["num_boost_round"]
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise AlgorithmExecutionError("XGBoost num_boost_round must be positive")

    class ShardDataIter(xgboost.DataIter):
        def __init__(self, cache_prefix: str) -> None:
            super().__init__(cache_prefix=cache_prefix, release_data=True)
            self.iterator: Any = None
            self.has_rows = False

        def reset(self) -> None:
            self.has_rows = False
            shard = train.get_dataset_shard("train")
            if shard is None:
                raise AlgorithmExecutionError("XGBoost train shard is missing")
            self.iterator = iter(
                shard.iter_batches(
                    batch_format="pandas",
                    batch_size=_XGBOOST_BATCH_ROWS,
                    prefetch_batches=1,
                )
            )

        def next(self, input_data: Any) -> bool:
            if self.iterator is None:
                self.reset()
            for frame in self.iterator:
                if not frame.empty:
                    self.has_rows = True
                    input_data(
                        data=frame[feature_names].to_numpy(),
                        label=frame[label_name].to_numpy(),
                    )
                    return True
            if not self.has_rows:
                raise AlgorithmExecutionError("XGBoost Worker shard is empty")
            return False

    context = train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    evidence_actor = config["evidence_actor"]

    with tempfile.TemporaryDirectory(prefix="tributo-xgboost-pages-") as cache_root:
        iterator = ShardDataIter(str(Path(cache_root) / "train"))
        dtrain = None
        try:
            dtrain = xgboost.DMatrix(iterator)
            rows_processed = int(dtrain.num_row())
            if rows_processed < 1:
                raise AlgorithmExecutionError("XGBoost Worker shard is empty")
            booster = xgboost.train(
                dict(config["params"]),
                dtrain,
                num_boost_round=rounds,
                evals=[(dtrain, "train")],
                verbose_eval=False,
            )
            actual_rounds = int(booster.num_boosted_rounds())
        finally:
            # DMatrix owns external-memory pages. Release it before its cache directory.
            del dtrain, iterator
            gc.collect()

    raw = bytes(booster.save_raw(raw_format="ubj"))
    digest = hashlib.sha256(raw).hexdigest()
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
        "rows_processed": rows_processed,
        "input_rows": {"train": rows_processed},
        "batch_count": actual_rounds,
        "collective_steps": actual_rounds,
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
    ray.get(evidence_actor.record.remote(evidence))
    metrics = {"model_state_digest": digest, "training_rounds": actual_rounds}
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix="tributo-xgboost-") as directory:
            root = Path(directory)
            booster.save_model(root / "model.ubj")
            (root / "feature_names.json").write_text(
                json.dumps(feature_names), encoding="utf-8"
            )
            train.report(metrics, checkpoint=Checkpoint.from_directory(root))
    else:
        train.report(metrics)


class DistributedXGBoost(FrameworkNativeAlgorithm):
    """Train one synchronized Booster through Ray's public XGBoostTrainer."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        self.collector: Any | None = None
        if plan.runtime.resume_from is not None:
            raise AlgorithmConfigurationError(
                "official XGBoost resume requires a separate failure-recovery gate"
            )

    def validate_environment(self) -> None:
        try:
            import xgboost
            from ray.train.xgboost import XGBoostTrainer
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "official XGBoost requires xgboost and Ray Train"
            ) from exc
        if not xgboost.__version__ or XGBoostTrainer is None:
            raise AlgorithmConfigurationError("XGBoost environment is invalid")

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        if len(datasets) != 1:
            raise AlgorithmConfigurationError("XGBoost requires one train Dataset")
        dataset = next(iter(datasets.values()))
        data = cast(Mapping[str, Any], self.plan.algorithm_config.get("data", {}))
        configured_features = tuple(data.get("feature_columns", ()))
        binding = self.plan.primary_input_binding
        feature_names = configured_features or binding.feature_names
        if tuple(feature_names) != binding.feature_names:
            raise AlgorithmConfigurationError(
                "XGBoost data.feature_columns must match InputBinding"
            )
        if data.get("label_col") != binding.label_name:
            raise AlgorithmConfigurationError(
                "XGBoost data.label_col must match InputBinding label"
            )
        return {
            "train": cast(Any, dataset).select_columns(
                [*feature_names, cast(str, binding.label_name)]
            )
        }

    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        import ray
        from ray.train import RunConfig, ScalingConfig
        from ray.train.xgboost import XGBoostTrainer

        data = cast(Mapping[str, Any], config.get("data", {}))
        model = dict(cast(Mapping[str, Any], config.get("model", {})))
        training = cast(Mapping[str, Any], config.get("training", {}))
        ray_config = cast(Mapping[str, Any], config.get("ray", {}))
        binding = self.plan.primary_input_binding
        collector_type = ray.remote(_EvidenceCollector).options(num_cpus=0)
        self.collector = collector_type.remote()
        return XGBoostTrainer(
            train_loop_per_worker=_train_loop,
            train_loop_config={
                "feature_names": list(binding.feature_names),
                "label_name": str(data["label_col"]),
                "params": model,
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
            dataset_config=CompleteCoverageDataConfig(datasets_to_split=["train"]),
            run_config=RunConfig(
                name="tributo-official-xgboost",
                storage_path=str(ray_config["storage_path"]),
            ),
        )

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        import ray

        del result
        if self.collector is None:
            raise AlgorithmExecutionError("XGBoost evidence collector is missing")
        workers = ray.get(self.collector.snapshot.remote())
        ray.kill(self.collector, no_restart=True)
        self.collector = None
        if len(workers) != self.plan.runtime.worker_count:
            raise AlgorithmExecutionError("XGBoost did not report every Worker")
        digests = {str(item.get("model_state_digest")) for item in workers}
        if len(digests) != 1:
            raise AlgorithmExecutionError("XGBoost Booster state diverged")
        return {
            "workers": workers,
            "state": {
                "coordination": "framework_native",
                "synchronized": True,
                "bounded": True,
                "global_model_digest": next(iter(digests)),
                "details": {"framework": "xgboost", "collective": "rabit"},
            },
            "input_complete": True,
        }

    def checkpoint_source(self, result: object) -> object:
        checkpoint = getattr(result, "checkpoint", None)
        if checkpoint is None:
            raise AlgorithmExecutionError("XGBoost result has no checkpoint")
        return checkpoint


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedXGBoost:
    del artifacts
    if implementation is not DistributedXGBoost:
        raise AlgorithmConfigurationError("XGBoost implementation drifted")
    return DistributedXGBoost(plan)


def export_result(
    *,
    result: object,
    checkpoint: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    import json

    import xgboost
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
        raise AlgorithmConfigurationError("XGBoost output.bundle_uri is required")
    with cast(Any, checkpoint).as_directory() as directory:
        root = Path(directory)
        booster = xgboost.Booster()
        booster.load_model(root / "model.ubj")
        feature_names = tuple(
            json.loads((root / "feature_names.json").read_text(encoding="utf-8"))
        )
        learner = json.loads(booster.save_config())["learner"]
        objective = str(learner["objective"]["name"])
        classification = objective.startswith(("binary:", "multi:"))
        class_count = max(2, int(learner["learner_model_param"]["num_class"]))
        output_schema = (
            (
                CheckpointField(name="label", dtype="int64", shape=("batch",)),
                CheckpointField(
                    name="probabilities",
                    dtype="float32",
                    shape=("batch", class_count),
                ),
            )
            if classification
            else (
                CheckpointField(
                    name="prediction",
                    dtype="float32",
                    shape=("batch", 1),
                ),
            )
        )
        checkpoint_contract = ExportCheckpointV1(
            trainer_type="xgboost",
            architecture_id="xgboost",
            input_schema=(
                CheckpointField(
                    name="float_input",
                    dtype="float32",
                    shape=("batch", len(feature_names)),
                ),
            ),
            output_schema=output_schema,
            preprocessing={"type": "none"},
            task_type="classification" if classification else "regression",
            framework="xgboost",
            framework_version=xgboost.__version__,
            checkpoint_format_version=1,
        )
        source_fingerprint = hashlib.sha256(
            bytes(booster.save_raw(raw_format="ubj"))
        ).hexdigest()
        source = ExportSource(
            source_kind="xgboost_result",
            model_object=booster,
            architecture_id="xgboost",
            feature_schema={"feature_names": list(feature_names)},
            metadata={
                "framework": "xgboost",
                "framework_versions": {"xgboost": xgboost.__version__},
                "objective": objective,
                "producer_distribution": "tributo-algorithms",
            },
            source_fingerprint=source_fingerprint,
            checkpoint_contract=checkpoint_contract,
        )
        bundle = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=str(output["bundle_uri"]),
                request_id=run_id,
                run_id=run_id,
                targets=[
                    ExportTarget(
                        name="onnx-model",
                        format="onnx",
                        exporter_id="official-xgboost-onnx-v1",
                    ),
                    ExportTarget(
                        name="native-model",
                        format="ubj",
                        exporter_id="official-xgboost-ubj-v1",
                    ),
                ],
                roles={"inference": "onnx-model", "native": "native-model"},
            ),
        )
    return AlgorithmExecutionResult(
        status="succeeded",
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
        },
    )


__all__ = ["DistributedXGBoost", "create_algorithm", "export_result"]
