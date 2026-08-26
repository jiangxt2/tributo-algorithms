"""Distributed full-gradient linear regression mathematical Hook."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from tributo.algorithms import IterativeOptimizationAlgorithm
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    MaterializedTabularInputView,
    TabularBatchInputView,
)

from tributo_algorithms_classical.models import SklearnModel


class LinearRegressionCheckpointCodec:
    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


class DistributedLinearRegression(
    IterativeOptimizationAlgorithm[
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
        SklearnModel,
    ]
):
    """Synchronous squared-error linear regression with L2 stabilization."""

    def initialize_state(
        self,
        config: Mapping[str, Any],
        input_descriptor: object,
    ) -> Mapping[str, object]:
        del input_descriptor
        feature_count = int(config.get("feature_count", 2))
        feature_names = tuple(
            str(name)
            for name in config.get(
                "_tributo_feature_names",
                tuple(f"feature_{index}" for index in range(feature_count)),
            )
        )
        if len(feature_names) != feature_count:
            raise ValueError("feature_count does not match input feature names")
        return {
            "coef": np.zeros(feature_count, dtype=np.float64),
            "intercept": np.zeros(1, dtype=np.float64),
            "round": np.zeros(1, dtype=np.int64),
            "feature_count": np.asarray([feature_count], dtype=np.int64),
            "learning_rate": np.asarray(
                [float(config.get("learning_rate", 0.05))], dtype=np.float64
            ),
            "tolerance": np.asarray(
                [float(config.get("tolerance", 1e-6))], dtype=np.float64
            ),
            "feature_names": feature_names,
        }

    def compute_partition_update(
        self,
        batches: Iterable[Mapping[str, object]],
        state: Mapping[str, object],
        round_index: int,
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        del round_index
        view = context.inputs["train"]
        if not isinstance(view, (TabularBatchInputView, MaterializedTabularInputView)):
            raise ValueError("Linear Regression requires tabular batch input")
        if view.label_name is None:
            raise ValueError("Linear Regression requires a label")
        coef = np.asarray(state["coef"], dtype=np.float64)
        intercept = float(np.asarray(state["intercept"])[0])
        gradient = np.zeros_like(coef)
        intercept_gradient = 0.0
        loss_sum = 0.0
        rows = 0
        for batch in batches:
            features = np.column_stack(
                [
                    np.asarray(batch[name], dtype=np.float64)
                    for name in view.feature_names
                ]
            )
            labels = np.asarray(batch[view.label_name], dtype=np.float64)
            if not np.isfinite(features).all() or not np.isfinite(labels).all():
                raise ValueError("Linear Regression requires finite input")
            errors = features @ coef + intercept - labels
            gradient += features.T @ errors
            intercept_gradient += float(errors.sum())
            loss_sum += float(np.square(errors).sum())
            rows += len(labels)
        return {
            "gradient_sum": gradient,
            "intercept_gradient_sum": np.asarray([intercept_gradient]),
            "loss_sum": np.asarray([loss_sum]),
            "row_count": np.asarray([rows], dtype=np.int64),
        }

    def merge_updates(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        return {name: np.asarray(left[name]) + np.asarray(right[name]) for name in left}

    def apply_update(
        self,
        state: Mapping[str, object],
        update: Mapping[str, object],
        round_index: int,
    ) -> Mapping[str, object]:
        rows = int(np.asarray(update["row_count"])[0])
        if rows < 1:
            raise ValueError("Linear Regression cannot update from zero rows")
        rate = float(np.asarray(state["learning_rate"])[0])
        coef = np.asarray(state["coef"], dtype=np.float64)
        return {
            **state,
            "coef": coef
            - rate
            * (
                np.asarray(update["gradient_sum"], dtype=np.float64) / rows
                + coef / rows
            ),
            "intercept": np.asarray(state["intercept"], dtype=np.float64)
            - rate
            * np.asarray(update["intercept_gradient_sum"], dtype=np.float64)
            / rows,
            "round": np.asarray([round_index + 1], dtype=np.int64),
        }

    def evaluate_round(
        self,
        state: Mapping[str, object],
        update: Mapping[str, object],
        round_index: int,
    ) -> Mapping[str, int | float]:
        del state, round_index
        rows = int(np.asarray(update["row_count"])[0])
        gradient = np.asarray(update["gradient_sum"], dtype=np.float64) / rows
        return {
            "loss": float(np.asarray(update["loss_sum"])[0] / rows),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "row_count": rows,
        }

    def should_stop(
        self,
        state: Mapping[str, object],
        metrics: Mapping[str, int | float],
        round_index: int,
    ) -> bool:
        del round_index
        return float(metrics["gradient_norm"]) < float(
            np.asarray(state["tolerance"])[0]
        )

    def finalize_model(self, state: Mapping[str, object]) -> SklearnModel:
        from sklearn.linear_model import LinearRegression

        estimator = LinearRegression()
        estimator.coef_ = np.asarray(state["coef"], dtype=np.float64)
        estimator.intercept_ = float(np.asarray(state["intercept"])[0])
        estimator.n_features_in_ = int(np.asarray(state["feature_count"])[0])
        feature_names = tuple(str(name) for name in state["feature_names"])
        return SklearnModel(estimator, feature_names, "regression")

    def state_schema(self) -> Mapping[str, object]:
        return {
            "coef": "float64[*]",
            "intercept": "float64[1]",
            "round": "int64[1]",
            "feature_count": "int64[1]",
            "learning_rate": "float64[1]",
            "tolerance": "float64[1]",
        }

    def update_schema(self) -> Mapping[str, object]:
        return {
            "gradient_sum": "float64[*]",
            "intercept_gradient_sum": "float64[1]",
            "loss_sum": "float64[1]",
            "row_count": "int64[1]",
        }

    def checkpoint_codec(self) -> object:
        return LinearRegressionCheckpointCodec()

    @property
    def retry_safe(self) -> bool:
        return True


__all__ = ["DistributedLinearRegression", "LinearRegressionCheckpointCodec"]
