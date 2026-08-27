"""Conditional CatBoost ensemble implementation over Ray shards."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from tributo.algorithms import EnsembleUnitSpec, ParallelEnsembleAlgorithm
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    MaterializedTabularInputView,
)


class CatBoostModel:
    """Final CatBoost model plus ordered feature metadata."""

    def __init__(
        self,
        model: object,
        feature_names: tuple[str, ...],
        task: str,
        cat_features: tuple[int, ...] = (),
        classes: tuple[object, ...] = (),
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.task = task
        self.cat_features = cat_features
        self.classes = classes


class _CheckpointCodec:
    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


def _arrays(
    inputs: Mapping[str, object],
    *,
    cat_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    view = cast(MaterializedTabularInputView, inputs["train"])
    if view.label_name is None:
        raise AlgorithmConfigurationError("CatBoost requires a label")
    columns = view.columns()
    raw_features = [np.asarray(columns[name]) for name in view.feature_names]
    if not raw_features or any(
        values.ndim != 1
        or not values.shape[0]
        or values.shape[0] != raw_features[0].shape[0]
        for values in raw_features
    ):
        raise AlgorithmExecutionError(
            "CatBoost input features must be non-empty columns"
        )
    if cat_indices:
        features = np.empty((raw_features[0].shape[0], len(raw_features)), dtype=object)
        for index, values in enumerate(raw_features):
            if index in cat_indices:
                features[:, index] = np.asarray(values).astype(str)
            else:
                features[:, index] = np.asarray(values, dtype=np.float64)
        numeric_features = [
            np.asarray(values, dtype=np.float64)
            for index, values in enumerate(raw_features)
            if index not in cat_indices
        ]
        if (
            numeric_features
            and not np.isfinite(np.column_stack(numeric_features)).all()
        ):
            raise AlgorithmExecutionError(
                "CatBoost numeric input contains non-finite values"
            )
    else:
        try:
            features = np.column_stack(
                [np.asarray(values, dtype=np.float64) for values in raw_features]
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "CatBoost categorical columns require explicit cat_features"
            ) from exc
        if not np.isfinite(features).all():
            raise AlgorithmExecutionError(
                "CatBoost input contains non-finite numeric values"
            )
    labels = np.asarray(columns[view.label_name])
    if not len(labels) or labels.shape != (features.shape[0],):
        raise AlgorithmExecutionError(
            "CatBoost input contains no rows or mismatched labels"
        )
    return features, labels, tuple(view.feature_names)


class CatBoostEnsemble(ParallelEnsembleAlgorithm[CatBoostModel, CatBoostModel]):
    """Train one CatBoost model per shard and combine additive models conditionally."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self._plan = plan
        self._feature_names = plan.primary_input_binding.feature_names

    def plan_units(
        self,
        config: Mapping[str, Any],
        input_descriptor: object,
        seed: int,
    ) -> tuple[EnsembleUnitSpec, ...]:
        del input_descriptor
        count = int(config.get("unit_count", config.get("n_estimators", 4)))
        if count < 1:
            raise AlgorithmConfigurationError("CatBoost unit_count must be positive")
        return tuple(
            EnsembleUnitSpec(unit_id=f"catboost-{index}", seed=seed + index, payload={})
            for index in range(count)
        )

    def fit_unit(
        self,
        unit: EnsembleUnitSpec,
        inputs: Mapping[str, object],
        context: AlgorithmExecutionContext,
    ) -> CatBoostModel:
        del context
        from catboost import CatBoostClassifier, CatBoostRegressor

        config = self._plan.algorithm_config
        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise AlgorithmConfigurationError("CatBoost model config must be a mapping")
        task = str(config.get("task", "classification"))
        cat_features = model_config.get("cat_features", ())
        if not isinstance(cat_features, (list, tuple)):
            raise AlgorithmConfigurationError(
                "CatBoost cat_features must be a sequence"
            )
        feature_names = self._feature_names
        cat_indices: list[int] = []
        for item in cat_features:
            if isinstance(item, str):
                if item not in feature_names:
                    raise AlgorithmConfigurationError(
                        f"CatBoost cat feature is not bound: {item!r}"
                    )
                cat_indices.append(feature_names.index(item))
            elif isinstance(item, int) and not isinstance(item, bool):
                if not 0 <= item < len(feature_names):
                    raise AlgorithmConfigurationError(
                        f"CatBoost cat feature index is out of range: {item}"
                    )
                cat_indices.append(item)
            else:
                raise AlgorithmConfigurationError(
                    "CatBoost cat_features entries must be names or indices"
                )
        if len(set(cat_indices)) != len(cat_indices):
            raise AlgorithmConfigurationError("CatBoost cat_features must be unique")
        features, labels, feature_names = _arrays(
            inputs, cat_indices=tuple(cat_indices)
        )
        if task == "regression":
            try:
                labels = np.asarray(labels, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise AlgorithmInputError(
                    "CatBoost regression labels must be numeric"
                ) from exc
            if not np.isfinite(labels).all():
                raise AlgorithmInputError("CatBoost regression labels must be finite")
        elif task == "classification":
            try:
                numeric_labels = np.asarray(labels, dtype=np.float64)
                if (
                    not np.isfinite(numeric_labels).all()
                    or not np.equal(numeric_labels, np.floor(numeric_labels)).all()
                ):
                    raise ValueError
                labels = numeric_labels.astype(np.int64)
            except (TypeError, ValueError):
                if any(item is None for item in labels):
                    raise AlgorithmInputError(
                        "CatBoost classification labels must not contain nulls"
                    ) from None
                labels = np.asarray(labels, dtype=str)
        else:
            raise AlgorithmConfigurationError(
                "CatBoost task must be classification or regression"
            )
        params: dict[str, Any] = {
            "iterations": int(model_config.get("iterations", 100)),
            "depth": int(model_config.get("depth", 6)),
            "learning_rate": float(model_config.get("learning_rate", 0.05)),
            "random_seed": int(unit.seed),
            "verbose": False,
            "thread_count": 1,
            "allow_writing_files": False,
        }
        if cat_features:
            params["cat_features"] = cat_indices
        estimator: Any
        if task == "regression":
            estimator = CatBoostRegressor(**params)
        elif task == "classification":
            estimator = CatBoostClassifier(**params)
        estimator.fit(features, labels)
        raw_classes = getattr(estimator, "classes_", ())
        classes = tuple(
            raw_classes.tolist() if hasattr(raw_classes, "tolist") else raw_classes
        )
        return CatBoostModel(
            estimator, feature_names, task, tuple(cat_indices), classes
        )

    def merge_units(self, ordered_units: tuple[CatBoostModel, ...]) -> object:
        return ordered_units

    def finalize_ensemble(self, merged: object) -> CatBoostModel:
        from catboost import sum_models

        units = cast(tuple[CatBoostModel, ...], merged)
        if not units:
            raise AlgorithmExecutionError("CatBoost requires at least one fitted unit")
        first = units[0]
        if any(
            unit.task != first.task
            or unit.feature_names != first.feature_names
            or unit.cat_features != first.cat_features
            or unit.classes != first.classes
            for unit in units
        ):
            raise AlgorithmExecutionError("CatBoost unit metadata is inconsistent")
        models = [unit.model for unit in units]
        combined = sum_models(models, weights=[1.0 / len(models)] * len(models))
        return CatBoostModel(
            combined,
            first.feature_names,
            first.task,
            first.cat_features,
            first.classes,
        )

    def unit_schema(self) -> Mapping[str, object]:
        return {
            "model": "catboost",
            "feature_names": "tuple[str]",
            "cat_features": "tuple[int]",
            "classes": "tuple[scalar]",
            "task": "classification|regression",
        }

    @property
    def retry_safe(self) -> bool:
        return True


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> CatBoostEnsemble:
    """Construct the descriptor-selected CatBoost ensemble."""
    del artifacts
    if implementation is not CatBoostEnsemble:
        raise AlgorithmConfigurationError("CatBoost implementation reference drifted")
    return CatBoostEnsemble(plan)


def export_result(
    *,
    model: CatBoostModel,
    plan: ResolvedAlgorithmPlan,
    run_id: str | None = None,
) -> AlgorithmExecutionResult:
    """Export the conditional CatBoost ensemble through native Bundle format."""
    if not isinstance(model, CatBoostModel):
        raise AlgorithmExecutionError("CatBoost exporter received an invalid model")
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError("CatBoost output.bundle_uri is required")
    import catboost
    from tributo.exporting.models import (
        BundleOutputConfig,
        CheckpointField,
        ExportCheckpointV1,
        ExportSource,
        ExportTarget,
    )
    from tributo.exporting.service import BundleExportService

    classification = model.task == "classification"
    raw_classes = getattr(model.model, "classes_", ())
    classes = model.classes or tuple(
        raw_classes.tolist() if hasattr(raw_classes, "tolist") else raw_classes
    )
    class_count = len(classes) or 2
    output_schema = (
        (
            CheckpointField(name="label", dtype="int64", shape=("batch",)),
            CheckpointField(
                name="probabilities", dtype="float32", shape=("batch", class_count)
            ),
        )
        if classification
        else (CheckpointField(name="variable", dtype="float32", shape=("batch", 1)),)
    )
    source = ExportSource(
        source_kind="catboost_result",
        model_object=model,
        feature_schema={
            "feature_names": list(model.feature_names),
            "cat_features": list(model.cat_features),
            "classes": list(classes),
        },
        metadata={
            "framework": "catboost",
            "framework_versions": {"catboost": catboost.__version__},
            "task_type": model.task,
            "producer_distribution": "tributo-algorithms-catboost",
        },
        checkpoint_contract=ExportCheckpointV1(
            trainer_type="catboost",
            architecture_id=plan.resolution.algorithm,
            input_schema=(
                CheckpointField(
                    name="float_input",
                    dtype="object" if model.cat_features else "float32",
                    shape=("batch", len(model.feature_names)),
                ),
            ),
            output_schema=output_schema,
            task_type=model.task,
            framework="catboost",
            framework_version="1",
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
                    name="native-model",
                    format="catboost",
                    exporter_id="catboost-native-v1",
                )
            ],
            roles={"inference": "native-model"},
        ),
    )
    return AlgorithmExecutionResult(
        status="succeeded",
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
            "feature_names": list(model.feature_names),
            "task": model.task,
        },
    )


__all__ = ["CatBoostEnsemble", "CatBoostModel", "create_algorithm", "export_result"]
