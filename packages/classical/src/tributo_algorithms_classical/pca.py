"""Distributed PCA using Chan/Welford mergeable sufficient statistics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmInputError,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import AlgorithmExecutionContext, MapReduceAlgorithm

from tributo_algorithms_classical.unsupervised_models import PCAModel

_MAX_PARTIAL_STATE_BYTES = 64 * 1024 * 1024


def _matrix(batch: Mapping[str, object], feature_names: tuple[str, ...]) -> np.ndarray:
    try:
        values = np.column_stack(
            [np.asarray(batch[name], dtype=np.float64) for name in feature_names]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AlgorithmInputError(
            "PCA input batch is not a dense feature matrix"
        ) from exc
    if values.ndim != 2 or not values.shape[0]:
        raise AlgorithmInputError("PCA input batches must contain rows")
    if not np.isfinite(values).all():
        raise AlgorithmInputError("PCA input must contain finite values")
    return values


def _batch_state(values: np.ndarray) -> dict[str, object]:
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    return {
        "count": np.asarray(values.shape[0], dtype=np.int64),
        "mean": mean,
        "M2": centered.T @ centered,
    }


class DistributedPCA(
    MapReduceAlgorithm[Mapping[str, object], Mapping[str, object], PCAModel]
):
    """Fit PCA from associative per-shard count, mean, and M2 states."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self._plan = plan
        self._feature_names = plan.primary_input_binding.feature_names
        feature_count = len(self._feature_names)
        configured_count = plan.algorithm_config.get("feature_count")
        if configured_count is not None and int(configured_count) != feature_count:
            raise AlgorithmConfigurationError("PCA feature_count disagrees with input")
        estimated_state_bytes = 8 * (feature_count * feature_count + feature_count + 1)
        if estimated_state_bytes > _MAX_PARTIAL_STATE_BYTES:
            raise AlgorithmConfigurationError(
                "PCA feature dimension exceeds the bounded partial-state limit"
            )

    def map_partition(
        self,
        batches: Iterable[Mapping[str, object]],
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        del context
        state = self.empty_partition()
        for batch in batches:
            state = self.merge_states(
                state, _batch_state(_matrix(batch, self._feature_names))
            )
        return state

    def merge_states(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        left_count = int(np.asarray(left["count"], dtype=np.int64))
        right_count = int(np.asarray(right["count"], dtype=np.int64))
        left_mean = np.asarray(left["mean"], dtype=np.float64)
        right_mean = np.asarray(right["mean"], dtype=np.float64)
        left_m2 = np.asarray(left["M2"], dtype=np.float64)
        right_m2 = np.asarray(right["M2"], dtype=np.float64)
        dimension = len(self._feature_names)
        if (
            left_mean.shape != (dimension,)
            or right_mean.shape != (dimension,)
            or left_m2.shape != (dimension, dimension)
            or right_m2.shape != (dimension, dimension)
        ):
            raise AlgorithmExecutionError(
                "PCA partial state dimensions are inconsistent"
            )
        if left_count == 0:
            return {
                "count": np.asarray(right_count),
                "mean": right_mean,
                "M2": right_m2,
            }
        if right_count == 0:
            return {"count": np.asarray(left_count), "mean": left_mean, "M2": left_m2}
        total = left_count + right_count
        delta = right_mean - left_mean
        mean = left_mean + delta * (right_count / total)
        m2 = (
            left_m2
            + right_m2
            + np.outer(delta, delta) * (left_count * right_count / total)
        )
        return {"count": np.asarray(total, dtype=np.int64), "mean": mean, "M2": m2}

    def finalize_model(self, state: Mapping[str, object]) -> PCAModel:
        count = int(np.asarray(state["count"], dtype=np.int64))
        if count < 2:
            raise AlgorithmInputError("PCA requires at least two rows")
        mean = np.asarray(state["mean"], dtype=np.float64)
        covariance = np.asarray(state["M2"], dtype=np.float64) / (count - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        components_count = int(
            self._plan.algorithm_config.get(
                "n_components", min(len(self._feature_names), count)
            )
        )
        if not 1 <= components_count <= min(len(self._feature_names), count):
            raise AlgorithmConfigurationError(
                "PCA n_components must be between 1 and min(feature_count, row_count)"
            )
        selected = order[:components_count]
        explained = np.maximum(eigenvalues[selected], 0.0)
        total_variance = float(np.maximum(eigenvalues, 0.0).sum())
        ratio = (
            explained / total_variance
            if total_variance > 0
            else np.zeros_like(explained)
        )
        return PCAModel(
            components=eigenvectors[:, selected].T,
            mean=mean,
            explained_variance=explained,
            explained_variance_ratio=ratio,
            feature_names=self._feature_names,
            n_samples=count,
        )

    def state_schema(self) -> tuple[Any, ...]:
        from tributo.algorithms.api import StateField

        return (
            StateField("count", "int64", ()),
            StateField("mean", "float64", (None,)),
            StateField("M2", "float64", (None, None)),
        )

    def empty_partition(self) -> Mapping[str, object]:
        dimension = len(self._feature_names)
        return {
            "count": np.asarray(0, dtype=np.int64),
            "mean": np.zeros(dimension, dtype=np.float64),
            "M2": np.zeros((dimension, dimension), dtype=np.float64),
        }

    @property
    def retry_safe(self) -> bool:
        return True


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedPCA:
    """Construct the descriptor-selected PCA implementation."""
    del artifacts
    if implementation is not DistributedPCA:
        raise AlgorithmConfigurationError("PCA implementation reference drifted")
    return DistributedPCA(plan)


__all__ = ["DistributedPCA", "create_algorithm"]
