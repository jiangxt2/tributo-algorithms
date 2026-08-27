"""Conditional distributed Isolation Forest parallel-ensemble algorithm."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from tributo.algorithms import EnsembleUnitSpec, ParallelEnsembleAlgorithm
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmInputError,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    MaterializedTabularInputView,
)

from tributo_algorithms_classical.unsupervised_models import IsolationForestModel


class _CheckpointCodec:
    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


def _arrays(inputs: Mapping[str, object]) -> tuple[np.ndarray, tuple[str, ...]]:
    view = cast(MaterializedTabularInputView, inputs["train"])
    columns = view.columns()
    values = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in view.feature_names]
    )
    if not values.shape[0] or not np.isfinite(values).all():
        raise AlgorithmInputError("Isolation Forest input must contain finite rows")
    return values, tuple(view.feature_names)


class IsolationForestEnsemble(
    ParallelEnsembleAlgorithm[IsolationForestModel, IsolationForestModel]
):
    """Train independent one-tree forests and combine them conditionally."""

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
        count = int(config.get("unit_count", config.get("n_estimators", 100)))
        if count < 1:
            raise AlgorithmConfigurationError(
                "Isolation Forest n_estimators must be positive"
            )
        return tuple(
            EnsembleUnitSpec(
                unit_id=f"isolation-tree-{index}",
                seed=seed + index,
                payload={},
            )
            for index in range(count)
        )

    def fit_unit(
        self,
        unit: EnsembleUnitSpec,
        inputs: Mapping[str, object],
        context: AlgorithmExecutionContext,
    ) -> IsolationForestModel:
        del context
        from sklearn.ensemble import IsolationForest

        values, feature_names = _arrays(inputs)
        config = self._plan.algorithm_config
        estimator = IsolationForest(
            n_estimators=1,
            max_samples=config.get("max_samples", "auto"),
            contamination=config.get("contamination", "auto"),
            random_state=unit.seed,
            n_jobs=1,
        )
        estimator.fit(values)
        training_values = values if unit.unit_id == "isolation-tree-0" else None
        return IsolationForestModel(
            estimator,
            feature_names,
            training_values,
        )

    def merge_units(self, ordered_units: tuple[IsolationForestModel, ...]) -> object:
        return ordered_units

    def finalize_ensemble(self, merged: object) -> IsolationForestModel:
        units = cast(tuple[IsolationForestModel, ...], merged)
        if not units:
            raise AlgorithmExecutionError("Isolation Forest requires fitted units")
        first = units[0]
        if any(unit.feature_names != first.feature_names for unit in units):
            raise AlgorithmExecutionError(
                "Isolation Forest unit metadata is inconsistent"
            )
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(
            n_estimators=len(units),
            contamination=cast(Any, first.estimator).contamination,
            n_jobs=1,
        )
        estimators = [cast(Any, unit.estimator).estimators_[0] for unit in units]
        model.estimators_ = estimators
        model.estimators_features_ = [
            np.arange(len(first.feature_names)) for _ in estimators
        ]
        model.max_samples_ = int(cast(Any, first.estimator).max_samples_)
        model._max_samples = model.max_samples_
        model.n_features_in_ = len(first.feature_names)
        model._max_features = len(first.feature_names)
        model._decision_path_lengths = tuple(
            cast(Any, unit.estimator)._decision_path_lengths[0] for unit in units
        )
        model._average_path_length_per_tree = tuple(
            cast(Any, unit.estimator)._average_path_length_per_tree[0] for unit in units
        )
        model.offset_ = float(cast(Any, first.estimator).offset_)
        contamination = cast(Any, first.estimator).contamination
        if contamination != "auto":
            training_values = first.training_values
            if training_values is None:
                raise AlgorithmExecutionError(
                    "Isolation Forest threshold calibration data is missing"
                )
            scores = model.score_samples(training_values)
            model.offset_ = float(np.quantile(scores, float(contamination)))
        model.n_estimators = len(estimators)
        model.max_samples = cast(Any, first.estimator).max_samples
        return IsolationForestModel(model, first.feature_names)

    def unit_schema(self) -> Mapping[str, object]:
        return {
            "estimator": "sklearn.ensemble.IsolationForest",
            "feature_names": "tuple[str]",
        }

    @property
    def retry_safe(self) -> bool:
        return True


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> IsolationForestEnsemble:
    """Construct the descriptor-selected Isolation Forest implementation."""
    del artifacts
    if implementation is not IsolationForestEnsemble:
        raise AlgorithmConfigurationError("Isolation Forest implementation drifted")
    return IsolationForestEnsemble(plan)


__all__ = ["IsolationForestEnsemble", "create_algorithm"]
