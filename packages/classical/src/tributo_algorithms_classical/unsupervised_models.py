"""Typed model metadata for official unsupervised algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAModel:
    """Principal-component transform state produced by distributed fitting."""

    components: np.ndarray
    mean: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    feature_names: tuple[str, ...]
    n_samples: int

    def __post_init__(self) -> None:
        components = np.asarray(self.components, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        explained_variance = np.asarray(self.explained_variance, dtype=np.float64)
        explained_variance_ratio = np.asarray(
            self.explained_variance_ratio, dtype=np.float64
        )
        if components.ndim != 2 or mean.ndim != 1:
            raise ValueError("PCA state must contain a matrix and a vector")
        if components.shape[1] != mean.shape[0]:
            raise ValueError("PCA components and mean dimensions disagree")
        if explained_variance.shape != (components.shape[0],):
            raise ValueError("PCA explained variance dimensions disagree")
        if explained_variance_ratio.shape != explained_variance.shape:
            raise ValueError("PCA explained variance ratio dimensions disagree")
        if len(self.feature_names) != mean.shape[0] or self.n_samples < 2:
            raise ValueError("PCA metadata is inconsistent")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "explained_variance", explained_variance)
        object.__setattr__(self, "explained_variance_ratio", explained_variance_ratio)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))


@dataclass(frozen=True)
class KMeansModel:
    """Cluster-center state produced by distributed KMeans fitting."""

    centers: np.ndarray
    feature_names: tuple[str, ...]
    n_iter: int
    variant: str

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        if centers.ndim != 2 or not centers.shape[0] or not centers.shape[1]:
            raise ValueError("KMeans centers must be a non-empty matrix")
        if len(self.feature_names) != centers.shape[1]:
            raise ValueError("KMeans centers and feature names disagree")
        if self.n_iter < 1 or self.variant not in {"kmeans", "minibatch"}:
            raise ValueError("KMeans metadata is invalid")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))


@dataclass(frozen=True)
class IsolationForestModel:
    """Isolation Forest model and ordered feature metadata."""

    estimator: object
    feature_names: tuple[str, ...]
    training_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("Isolation Forest requires feature names")
        if self.training_values is not None:
            values = np.asarray(self.training_values, dtype=np.float64)
            if (
                values.ndim != 2
                or not values.shape[0]
                or values.shape[1] != len(self.feature_names)
                or not np.isfinite(values).all()
            ):
                raise ValueError("Isolation Forest training values are inconsistent")
            object.__setattr__(self, "training_values", values)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))


__all__ = ["IsolationForestModel", "KMeansModel", "PCAModel"]
