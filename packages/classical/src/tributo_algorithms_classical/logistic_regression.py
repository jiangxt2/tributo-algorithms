"""Binary L2 logistic regression mathematical Hook implementation."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
from tributo.algorithms import IterativeOptimizationAlgorithm
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    MaterializedTabularInputView,
    TabularBatchInputView,
)

from tributo_algorithms_classical.models import SklearnModel


class PickleCheckpointCodec:
    """Serialize complete iterative optimizer state."""

    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


class BinaryL2LogisticRegression(
    IterativeOptimizationAlgorithm[
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
        SklearnModel,
    ]
):
    """Synchronous full-gradient binary logistic regression."""

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
        c_value = float(config.get("C", 1.0))
        if not np.isfinite(c_value) or c_value <= 0:
            raise ValueError("C must be a positive finite value")
        return {
            "coef": np.zeros(feature_count, dtype=np.float64),
            "intercept": np.zeros(1, dtype=np.float64),
            "round": np.zeros(1, dtype=np.int64),
            "feature_count": np.asarray([feature_count], dtype=np.int64),
            "learning_rate": np.asarray(
                [float(config.get("learning_rate", 0.25))], dtype=np.float64
            ),
            "tolerance": np.asarray(
                [float(config.get("tolerance", 1e-6))], dtype=np.float64
            ),
            "regularization": np.asarray([1.0 / c_value], dtype=np.float64),
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
            raise ValueError("Logistic Regression requires tabular batch input")
        feature_names = tuple(str(name) for name in view.feature_names)
        label_name = view.label_name
        if not isinstance(label_name, str):
            raise ValueError("Logistic Regression requires a label")
        coef = np.asarray(state["coef"], dtype=np.float64)
        intercept = float(np.asarray(state["intercept"])[0])
        gradient = np.zeros_like(coef)
        hessian = np.zeros((len(coef) + 1, len(coef) + 1), dtype=np.float64)
        intercept_gradient = 0.0
        loss_sum = 0.0
        rows = 0
        for batch in batches:
            features = np.column_stack(
                [np.asarray(batch[name], dtype=np.float64) for name in feature_names]
            )
            labels = np.asarray(batch[label_name], dtype=np.float64)
            if not np.isin(labels, (0.0, 1.0)).all():
                raise ValueError("binary Logistic Regression labels must be 0 or 1")
            logits = features @ coef + intercept
            probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            error = probability - labels
            gradient += features.T @ error
            intercept_gradient += float(np.sum(error))
            design = np.column_stack((features, np.ones(len(labels))))
            weights = probability * (1.0 - probability)
            hessian += design.T @ (weights[:, None] * design)
            loss_sum += float(np.sum(np.logaddexp(0.0, logits) - labels * logits))
            rows += len(labels)
        return {
            "gradient_sum": gradient,
            "intercept_gradient_sum": np.asarray([intercept_gradient]),
            "hessian_sum": hessian,
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
            raise ValueError("Logistic Regression cannot update from zero rows")
        learning_rate = float(np.asarray(state["learning_rate"])[0])
        regularization = float(np.asarray(state["regularization"])[0]) / rows
        coef = np.asarray(state["coef"], dtype=np.float64)
        intercept = float(np.asarray(state["intercept"])[0])
        gradient = (
            np.concatenate(
                (
                    np.asarray(update["gradient_sum"], dtype=np.float64),
                    np.asarray(update["intercept_gradient_sum"], dtype=np.float64),
                )
            )
            / rows
        )
        gradient[: len(coef)] += regularization * coef
        hessian = np.asarray(update["hessian_sum"], dtype=np.float64) / rows
        hessian[: len(coef), : len(coef)] += np.eye(len(coef)) * regularization
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        return {
            **state,
            "coef": coef - learning_rate * step[:-1],
            "intercept": np.asarray([intercept - learning_rate * step[-1]]),
            "round": np.asarray([round_index + 1], dtype=np.int64),
        }

    def evaluate_round(
        self,
        state: Mapping[str, object],
        update: Mapping[str, object],
        round_index: int,
    ) -> Mapping[str, int | float]:
        del round_index
        rows = int(np.asarray(update["row_count"])[0])
        gradient = np.asarray(update["gradient_sum"], dtype=np.float64) / rows
        gradient += (
            float(np.asarray(state["regularization"])[0])
            / rows
            * np.asarray(state["coef"], dtype=np.float64)
        )
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
        tolerance = float(np.asarray(state["tolerance"])[0])
        return float(metrics["gradient_norm"]) < tolerance

    def finalize_model(self, state: Mapping[str, object]) -> SklearnModel:
        from sklearn.linear_model import LogisticRegression

        estimator = LogisticRegression()
        estimator.classes_ = np.asarray([0, 1], dtype=np.int64)
        estimator.coef_ = np.asarray(state["coef"], dtype=np.float64)[None, :]
        estimator.intercept_ = np.asarray(state["intercept"], dtype=np.float64)
        estimator.n_iter_ = np.asarray(state["round"], dtype=np.int32)
        estimator.n_features_in_ = int(np.asarray(state["feature_count"])[0])
        feature_names = tuple(
            str(name) for name in cast(Iterable[object], state["feature_names"])
        )
        estimator.C = 1.0 / float(np.asarray(state["regularization"])[0])
        return SklearnModel(estimator, feature_names, "classification")

    def state_schema(self) -> Mapping[str, object]:
        return {
            "coef": "float64[*]",
            "intercept": "float64[1]",
            "round": "int64[1]",
            "feature_count": "int64[1]",
            "learning_rate": "float64[1]",
            "tolerance": "float64[1]",
            "regularization": "float64[1]",
            "feature_names": "tuple[str]",
        }

    def update_schema(self) -> Mapping[str, object]:
        return {
            "gradient_sum": "float64[*]",
            "intercept_gradient_sum": "float64[1]",
            "hessian_sum": "float64[*,*]",
            "loss_sum": "float64[1]",
            "row_count": "int64[1]",
        }

    def checkpoint_codec(self) -> object:
        return PickleCheckpointCodec()

    @property
    def retry_safe(self) -> bool:
        return True


__all__ = ["BinaryL2LogisticRegression", "PickleCheckpointCodec"]
