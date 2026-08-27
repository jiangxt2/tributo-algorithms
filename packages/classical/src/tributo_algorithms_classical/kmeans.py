"""Distributed KMeans and MiniBatchKMeans with synchronous centroid updates."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from typing import Any

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

from tributo_algorithms_classical.unsupervised_models import KMeansModel


class _CheckpointCodec:
    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


def _matrix(batch: Mapping[str, object], feature_names: tuple[str, ...]) -> np.ndarray:
    try:
        values = np.column_stack(
            [np.asarray(batch[name], dtype=np.float64) for name in feature_names]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AlgorithmInputError(
            "KMeans input batch is not a dense feature matrix"
        ) from exc
    if values.ndim != 2 or not values.shape[0] or not np.isfinite(values).all():
        raise AlgorithmInputError("KMeans input batches must contain finite rows")
    return values


class DistributedKMeans(
    IterativeOptimizationAlgorithm[
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
        KMeansModel,
    ]
):
    """Fit centroids through synchronous full-data or mini-batch rounds."""

    def __init__(self, plan: ResolvedAlgorithmPlan, *, variant: str) -> None:
        if variant not in {"kmeans", "minibatch"}:
            raise AlgorithmConfigurationError("unsupported KMeans variant")
        self._plan = plan
        self._feature_names = plan.primary_input_binding.feature_names
        self._variant = variant

    def initialize_state(
        self,
        config: Mapping[str, Any],
        input_descriptor: object,
    ) -> Mapping[str, object]:
        del input_descriptor
        n_clusters = int(config.get("n_clusters", 8))
        feature_count = len(self._feature_names)
        declared = config.get("feature_count")
        if declared is not None and int(declared) != feature_count:
            raise AlgorithmConfigurationError(
                "KMeans feature_count disagrees with input"
            )
        if n_clusters < 1:
            raise AlgorithmConfigurationError("n_clusters must be positive")
        max_iter = int(config.get("max_iter", 100))
        if max_iter < 1:
            raise AlgorithmConfigurationError("KMeans max_iter must be positive")
        seed = int(config.get("seed", 0))
        centers = np.random.default_rng(seed).normal(
            loc=0.0,
            scale=1.0,
            size=(n_clusters, feature_count),
        )
        return {
            "centers": centers,
            "round": np.asarray(0, dtype=np.int64),
            "max_iter": np.asarray(max_iter, dtype=np.int64),
            "last_shift": np.asarray(np.inf, dtype=np.float64),
            "rows_seen": np.asarray(0, dtype=np.int64),
        }

    def compute_partition_update(
        self,
        batches: Iterable[Mapping[str, object]],
        state: Mapping[str, object],
        round_index: int,
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        del context
        centers = np.asarray(state["centers"], dtype=np.float64)
        sums = np.zeros_like(centers)
        counts = np.zeros(centers.shape[0], dtype=np.int64)
        loss = 0.0
        sampled_rows = 0
        dataset_rows = 0
        reservoir: list[np.ndarray] = []
        seen = 0
        batch_size = int(self._plan.algorithm_config.get("batch_size", 128))
        if self._variant == "minibatch" and batch_size < 1:
            raise AlgorithmConfigurationError(
                "MiniBatchKMeans batch_size must be positive"
            )
        rng = np.random.default_rng(
            int(self._plan.algorithm_config.get("seed", 0)) + round_index
        )

        def accumulate(values: np.ndarray) -> None:
            nonlocal loss, sampled_rows
            distances = np.sum(
                (values[:, None, :] - centers[None, :, :]) ** 2,
                axis=2,
            )
            assignments = np.argmin(distances, axis=1)
            nearest = distances[np.arange(values.shape[0]), assignments]
            loss += float(nearest.sum())
            sampled_rows += values.shape[0]
            for index in range(centers.shape[0]):
                selected = values[assignments == index]
                if selected.size:
                    counts[index] += selected.shape[0]
                    sums[index] += selected.sum(axis=0, dtype=np.float64)

        for batch in batches:
            batch_values = _matrix(batch, self._feature_names)
            dataset_rows += batch_values.shape[0]
            if self._variant != "minibatch":
                accumulate(batch_values)
                continue
            for row in batch_values:
                seen += 1
                if len(reservoir) < batch_size:
                    reservoir.append(row.copy())
                    continue
                replacement = int(rng.integers(seen))
                if replacement < batch_size:
                    reservoir[replacement] = row.copy()
        if self._variant == "minibatch" and reservoir:
            accumulate(np.asarray(reservoir, dtype=np.float64))
        if dataset_rows == 0:
            return {
                "sums": sums,
                "counts": counts,
                "loss": np.asarray(0.0, dtype=np.float64),
                "row_count": np.asarray(0, dtype=np.int64),
                "dataset_row_count": np.asarray(0, dtype=np.int64),
            }
        return {
            "sums": sums,
            "counts": counts,
            "loss": np.asarray(loss, dtype=np.float64),
            "row_count": np.asarray(sampled_rows, dtype=np.int64),
            "dataset_row_count": np.asarray(dataset_rows, dtype=np.int64),
        }

    def merge_updates(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        return {
            "sums": np.asarray(left["sums"]) + np.asarray(right["sums"]),
            "counts": np.asarray(left["counts"]) + np.asarray(right["counts"]),
            "loss": np.asarray(left["loss"]) + np.asarray(right["loss"]),
            "row_count": np.asarray(left["row_count"]) + np.asarray(right["row_count"]),
            "dataset_row_count": np.asarray(left["dataset_row_count"])
            + np.asarray(right["dataset_row_count"]),
        }

    def apply_update(
        self,
        state: Mapping[str, object],
        update: Mapping[str, object],
        round_index: int,
    ) -> Mapping[str, object]:
        old = np.asarray(state["centers"], dtype=np.float64)
        sums = np.asarray(update["sums"], dtype=np.float64)
        counts = np.asarray(update["counts"], dtype=np.int64)
        dataset_rows = int(np.asarray(update["dataset_row_count"], dtype=np.int64))
        if int(np.asarray(update["row_count"], dtype=np.int64)) < 1:
            raise AlgorithmInputError("KMeans cannot update from zero rows")
        # The runtime tree-reduces all shard updates before this method runs;
        # dataset_row_count is therefore the merged count for this round.
        if round_index == 0 and counts.shape[0] > dataset_rows:
            raise AlgorithmInputError("KMeans n_clusters cannot exceed row_count")
        means = old.copy()
        populated = counts > 0
        means[populated] = sums[populated] / counts[populated, None]
        if self._variant == "minibatch":
            rate = float(
                self._plan.algorithm_config.get(
                    "learning_rate", 1.0 / (round_index + 1)
                )
            )
            if not 0 < rate <= 1:
                raise AlgorithmConfigurationError(
                    "MiniBatchKMeans learning_rate must be in (0, 1]"
                )
            new_centers = old + rate * (means - old)
        else:
            new_centers = means
        return {
            "centers": new_centers,
            "round": np.asarray(round_index + 1, dtype=np.int64),
            "max_iter": state["max_iter"],
            "last_shift": np.asarray(
                np.linalg.norm(new_centers - old), dtype=np.float64
            ),
            "rows_seen": np.asarray(
                int(np.asarray(state.get("rows_seen", 0), dtype=np.int64))
                + dataset_rows,
                dtype=np.int64,
            ),
        }

    def evaluate_round(
        self,
        state: Mapping[str, object],
        update: Mapping[str, object],
        round_index: int,
    ) -> Mapping[str, int | float]:
        del round_index
        rows = int(np.asarray(update["row_count"], dtype=np.int64))
        return {
            "inertia": float(np.asarray(update["loss"], dtype=np.float64) / rows),
            "center_shift": float(np.asarray(state["last_shift"], dtype=np.float64)),
            "row_count": rows,
        }

    def should_stop(
        self,
        state: Mapping[str, object],
        metrics: Mapping[str, int | float],
        round_index: int,
    ) -> bool:
        del round_index
        completed = int(np.asarray(state["round"], dtype=np.int64))
        max_iter = int(np.asarray(state["max_iter"], dtype=np.int64))
        return (
            float(metrics["center_shift"])
            <= float(self._plan.algorithm_config.get("tolerance", 1e-4))
            or completed >= max_iter
        )

    def finalize_model(self, state: Mapping[str, object]) -> KMeansModel:
        centers = np.asarray(state["centers"], dtype=np.float64)
        n_iter = int(np.asarray(state["round"], dtype=np.int64))
        if n_iter < 1:
            raise AlgorithmExecutionError("KMeans did not complete a training round")
        rows_seen = int(np.asarray(state.get("rows_seen", 0), dtype=np.int64))
        if centers.shape[0] > rows_seen:
            raise AlgorithmInputError("KMeans n_clusters cannot exceed row_count")
        return KMeansModel(centers, self._feature_names, n_iter, self._variant)

    def state_schema(self) -> Mapping[str, object]:
        return {
            "centers": "float64[clusters,features]",
            "round": "int64[1]",
            "max_iter": "int64[1]",
            "last_shift": "float64[1]",
            "rows_seen": "int64[1]",
        }

    def update_schema(self) -> Mapping[str, object]:
        return {
            "sums": "float64[clusters,features]",
            "counts": "int64[clusters]",
            "loss": "float64[1]",
            "row_count": "int64[1]",
            "dataset_row_count": "int64[1]",
        }

    def checkpoint_codec(self) -> object:
        return _CheckpointCodec()

    @property
    def retry_safe(self) -> bool:
        return True


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedKMeans:
    """Construct the descriptor-selected KMeans implementation."""
    del artifacts
    if implementation is DistributedKMeans:
        return DistributedKMeans(plan, variant="kmeans")
    if implementation is MiniBatchKMeans:
        return DistributedKMeans(plan, variant="minibatch")
    raise AlgorithmConfigurationError("KMeans implementation reference drifted")


class MiniBatchKMeans(DistributedKMeans):
    """Marker class selecting the MiniBatchKMeans update rule."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        super().__init__(plan, variant="minibatch")


__all__ = ["DistributedKMeans", "MiniBatchKMeans", "create_algorithm"]
