"""Synchronous full-batch linear gradient algorithms."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmInputError,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    IterativeOptimizationAlgorithm,
)

from tributo_algorithms_classical.models import SklearnModel


class _CheckpointCodec:
    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


def _batch_arrays(
    batch: Mapping[str, object],
    feature_names: tuple[str, ...],
    label_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        features = np.column_stack(
            [np.asarray(batch[name], dtype=np.float64) for name in feature_names]
        )
        labels = np.asarray(batch[label_name], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise AlgorithmInputError(
            "SGD input batch is not a dense labeled matrix"
        ) from exc
    if not features.shape[0] or labels.shape != (features.shape[0],):
        raise AlgorithmInputError("SGD input batch dimensions are invalid")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise AlgorithmInputError("SGD input must contain finite values")
    return features, labels


class DistributedSGD(
    IterativeOptimizationAlgorithm[
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
        SklearnModel,
    ]
):
    """Synchronously update one global linear model per full-data round."""

    def __init__(self, plan: ResolvedAlgorithmPlan, *, task: str) -> None:
        if task not in {"classification", "regression"}:
            raise AlgorithmConfigurationError("SGD task is invalid")
        self._plan = plan
        self._task = task
        self._feature_names = plan.primary_input_binding.feature_names
        label_name = plan.primary_input_binding.label_name
        if label_name is None:
            raise AlgorithmConfigurationError("SGD requires a label")
        self._label_name = label_name

    def initialize_state(
        self,
        config: Mapping[str, Any],
        input_descriptor: object,
    ) -> Mapping[str, object]:
        del input_descriptor
        feature_count = len(self._feature_names)
        alpha = float(config.get("alpha", 0.0001))
        learning_rate = float(config.get("learning_rate", 0.01))
        decay = float(config.get("learning_rate_decay", 0.0))
        if alpha <= 0 or learning_rate <= 0 or decay < 0:
            raise AlgorithmConfigurationError("SGD optimization parameters are invalid")
        max_iter = int(config.get("max_iter", 20))
        if max_iter < 1:
            raise AlgorithmConfigurationError("SGD max_iter must be positive")
        tolerance = float(config.get("tolerance", 1e-5))
        if tolerance <= 0:
            raise AlgorithmConfigurationError("SGD tolerance must be positive")
        loss = str(
            config.get(
                "loss",
                "log_loss" if self._task == "classification" else "squared_error",
            )
        )
        expected_loss = (
            "log_loss" if self._task == "classification" else "squared_error"
        )
        if loss != expected_loss:
            raise AlgorithmConfigurationError(
                f"SGD {self._task} requires loss={expected_loss!r}"
            )
        return {
            "coef": np.zeros(feature_count, dtype=np.float64),
            "intercept": np.asarray(0.0, dtype=np.float64),
            "round": np.asarray(0, dtype=np.int64),
            "step": np.asarray(0, dtype=np.int64),
            "last_gradient_norm": np.asarray(np.inf, dtype=np.float64),
            "alpha": np.asarray(alpha, dtype=np.float64),
            "learning_rate": np.asarray(learning_rate, dtype=np.float64),
            "learning_rate_decay": np.asarray(decay, dtype=np.float64),
            "max_iter": np.asarray(max_iter, dtype=np.int64),
            "tolerance": np.asarray(tolerance, dtype=np.float64),
            "feature_names": self._feature_names,
        }

    def compute_partition_update(
        self,
        batches: Iterable[Mapping[str, object]],
        state: Mapping[str, object],
        round_index: int,
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        del round_index, context
        coef = np.asarray(state["coef"], dtype=np.float64)
        intercept = float(np.asarray(state["intercept"], dtype=np.float64))
        gradient = np.zeros_like(coef)
        intercept_gradient = 0.0
        loss_sum = 0.0
        row_count = 0
        for batch in batches:
            features, labels = _batch_arrays(
                batch, self._feature_names, self._label_name
            )
            logits = features @ coef + intercept
            if self._task == "classification":
                if not np.isin(labels, (0.0, 1.0)).all():
                    raise AlgorithmInputError("SGD classifier labels must be 0 or 1")
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
                errors = probabilities - labels
                loss = np.logaddexp(0.0, logits) - labels * logits
            else:
                errors = logits - labels
                loss = 0.5 * np.square(errors)
            gradient += features.T @ errors
            intercept_gradient += float(errors.sum())
            loss_sum += float(loss.sum())
            row_count += len(labels)
        return {
            "gradient": gradient,
            "intercept_gradient": np.asarray(intercept_gradient, dtype=np.float64),
            "loss": np.asarray(loss_sum, dtype=np.float64),
            "row_count": np.asarray(row_count, dtype=np.int64),
        }

    def merge_updates(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        dimension = len(self._feature_names)
        left_gradient = np.asarray(left["gradient"], dtype=np.float64)
        right_gradient = np.asarray(right["gradient"], dtype=np.float64)
        if left_gradient.shape != (dimension,) or right_gradient.shape != (dimension,):
            raise AlgorithmExecutionError("SGD gradient dimensions are inconsistent")
        return {
            "gradient": left_gradient + right_gradient,
            "intercept_gradient": np.asarray(left["intercept_gradient"])
            + np.asarray(right["intercept_gradient"]),
            "loss": np.asarray(left["loss"]) + np.asarray(right["loss"]),
            "row_count": np.asarray(left["row_count"]) + np.asarray(right["row_count"]),
        }

    def apply_update(
        self,
        state: Mapping[str, object],
        update: Mapping[str, object],
        round_index: int,
    ) -> Mapping[str, object]:
        if int(np.asarray(update["row_count"], dtype=np.int64)) < 1:
            raise AlgorithmInputError("SGD cannot update from zero rows")
        coef = np.asarray(state["coef"], dtype=np.float64).copy()
        intercept = float(np.asarray(state["intercept"], dtype=np.float64))
        step = int(np.asarray(state["step"], dtype=np.int64))
        alpha = float(np.asarray(state["alpha"], dtype=np.float64))
        base_rate = float(np.asarray(state["learning_rate"], dtype=np.float64))
        decay = float(np.asarray(state["learning_rate_decay"], dtype=np.float64))
        rows = int(np.asarray(update["row_count"], dtype=np.int64))
        gradient = np.asarray(update["gradient"], dtype=np.float64) / rows
        intercept_gradient = (
            float(np.asarray(update["intercept_gradient"], dtype=np.float64)) / rows
        )
        gradient += alpha * coef
        rate = base_rate / (1.0 + decay * step)
        coef -= rate * gradient
        intercept -= rate * intercept_gradient
        step += 1
        last_norm = float(np.linalg.norm(gradient))
        return {
            **state,
            "coef": coef,
            "intercept": np.asarray(intercept, dtype=np.float64),
            "round": np.asarray(round_index + 1, dtype=np.int64),
            "step": np.asarray(step, dtype=np.int64),
            "last_gradient_norm": np.asarray(last_norm, dtype=np.float64),
        }

    def evaluate_round(
        self,
        state: Mapping[str, object],
        update: Mapping[str, object],
        round_index: int,
    ) -> Mapping[str, int | float]:
        del state, round_index
        rows = int(np.asarray(update["row_count"], dtype=np.int64))
        loss = float(np.asarray(update["loss"], dtype=np.float64))
        return {"loss": loss / rows, "row_count": rows}

    def should_stop(
        self,
        state: Mapping[str, object],
        metrics: Mapping[str, int | float],
        round_index: int,
    ) -> bool:
        del metrics
        completed = int(np.asarray(state["round"], dtype=np.int64))
        max_iter = int(np.asarray(state["max_iter"], dtype=np.int64))
        tolerance = float(np.asarray(state["tolerance"], dtype=np.float64))
        return (
            completed >= max_iter
            or float(np.asarray(state["last_gradient_norm"], dtype=np.float64))
            < tolerance
        )

    def finalize_model(self, state: Mapping[str, object]) -> SklearnModel:
        coef = np.asarray(state["coef"], dtype=np.float64)
        intercept = float(np.asarray(state["intercept"], dtype=np.float64))
        rounds = int(np.asarray(state["round"], dtype=np.int64))
        steps = int(np.asarray(state["step"], dtype=np.int64))
        if rounds < 1:
            raise AlgorithmExecutionError("SGD did not complete a training round")
        if self._task == "classification":
            from sklearn.linear_model import SGDClassifier

            estimator: Any = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1)
            estimator.classes_ = np.asarray([0, 1], dtype=np.int64)
            estimator.coef_ = coef[None, :]
            estimator.intercept_ = np.asarray([intercept], dtype=np.float64)
            estimator.n_features_in_ = len(self._feature_names)
            estimator.n_iter_ = np.asarray([rounds], dtype=np.int32)
            estimator.t_ = float(steps + 1)
        else:
            from sklearn.linear_model import SGDRegressor

            estimator = SGDRegressor(loss="squared_error", penalty="l2", max_iter=1)
            estimator.coef_ = coef
            estimator.intercept_ = np.asarray([intercept], dtype=np.float64)
            estimator.n_features_in_ = len(self._feature_names)
            estimator.n_iter_ = rounds
            estimator.t_ = float(steps + 1)
        return SklearnModel(estimator, self._feature_names, self._task)

    def state_schema(self) -> Mapping[str, object]:
        return {
            "coef": "float64[features]",
            "intercept": "float64[1]",
            "round": "int64[1]",
            "step": "int64[1]",
            "last_gradient_norm": "float64[1]",
            "alpha": "float64[1]",
            "learning_rate": "float64[1]",
            "learning_rate_decay": "float64[1]",
            "max_iter": "int64[1]",
            "tolerance": "float64[1]",
            "feature_names": "tuple[str]",
        }

    def update_schema(self) -> Mapping[str, object]:
        return {
            "gradient": "float64[features]",
            "intercept_gradient": "float64[1]",
            "loss": "float64[1]",
            "row_count": "int64[1]",
        }

    def checkpoint_codec(self) -> object:
        return _CheckpointCodec()

    @property
    def retry_safe(self) -> bool:
        return True


class DistributedSGDClassifier(DistributedSGD):
    """Binary synchronous SGD classifier."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        super().__init__(plan, task="classification")


class DistributedSGDRegressor(DistributedSGD):
    """Squared-error synchronous SGD regressor."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        super().__init__(plan, task="regression")


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedSGD:
    """Construct the descriptor-selected SGD implementation."""
    del artifacts
    if not isinstance(implementation, type) or not issubclass(
        implementation, DistributedSGD
    ):
        raise AlgorithmConfigurationError("SGD implementation reference drifted")
    constructor = cast(Any, implementation)
    return cast(DistributedSGD, constructor(plan))


__all__ = [
    "DistributedSGD",
    "DistributedSGDClassifier",
    "DistributedSGDRegressor",
    "create_algorithm",
]
