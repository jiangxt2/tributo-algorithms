"""Executable contracts for the official unsupervised algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class UnsupervisedConfigValidator:
    """Validate the shared bounded configuration envelope."""

    api_version = 1
    schema_digest = "6" * 64
    _ALLOWED = {
        "batch_size",
        "checkpoint",
        "contamination",
        "distance",
        "feature_count",
        "init",
        "learning_rate",
        "max_iter",
        "max_samples",
        "n_clusters",
        "n_components",
        "n_estimators",
        "output",
        "ray",
        "runtime",
        "seed",
        "tolerance",
        "unit_count",
    }

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = sorted(set(value) - self._ALLOWED)
        if unknown:
            raise ValueError(f"unknown unsupervised config keys: {unknown}")
        normalized = dict(value)
        for name in (
            "batch_size",
            "feature_count",
            "max_iter",
            "n_clusters",
            "n_components",
            "n_estimators",
            "unit_count",
        ):
            item = normalized.get(name)
            if item is not None and (
                not isinstance(item, int) or isinstance(item, bool) or item < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        seed = normalized.get("seed")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        max_samples = normalized.get("max_samples")
        if max_samples is not None and max_samples != "auto":
            if isinstance(max_samples, int) and not isinstance(max_samples, bool):
                if max_samples < 1:
                    raise ValueError("max_samples integer must be positive")
            elif (
                not isinstance(max_samples, (int, float))
                or isinstance(max_samples, bool)
                or not 0 < float(max_samples) <= 1
            ):
                raise ValueError(
                    "max_samples must be 'auto', a positive integer, or a value in (0, 1]"
                )
        for name in ("learning_rate", "tolerance"):
            item = normalized.get(name)
            if item is not None and (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or float(item) <= 0
            ):
                raise ValueError(f"{name} must be a positive number")
        contamination = normalized.get("contamination")
        if contamination is not None and contamination != "auto":
            if (
                not isinstance(contamination, (int, float))
                or isinstance(contamination, bool)
                or not 0 < float(contamination) <= 0.5
            ):
                raise ValueError("contamination must be 'auto' or a number in (0, 0.5]")
        output = normalized.get("output")
        if output is not None and (
            not isinstance(output, Mapping)
            or not isinstance(output.get("bundle_uri"), str)
            or not output["bundle_uri"]
        ):
            raise ValueError("output.bundle_uri must be non-empty")
        return normalized


class UnlabeledDenseInputValidator:
    """Require one feature-only tabular input role."""

    api_version = 1
    schema_digest = "7" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        descriptors = value.get("descriptors")
        if not isinstance(bindings, list) or not isinstance(descriptors, Mapping):
            raise ValueError("unlabeled input requires bindings and descriptors")
        train = next(
            (
                item
                for item in bindings
                if isinstance(item, Mapping) and item.get("name") == "train"
            ),
            None,
        )
        descriptor = descriptors.get("train")
        if not isinstance(train, Mapping) or not isinstance(descriptor, Mapping):
            raise ValueError("unlabeled input requires the train role")
        if train.get("label_name") is not None:
            raise ValueError("unlabeled input must not declare label_name")
        features = train.get("feature_names")
        if not isinstance(features, list) or not features:
            raise ValueError("unlabeled input requires feature_names")
        return value


class PCAOutputValidator:
    """Require a successful PCA Bundle result."""

    api_version = 1
    schema_digest = "8" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("PCA output requires successful execution")
        if not outputs.get("bundle_uri"):
            raise ValueError("PCA output requires bundle_uri")
        return value


class KMeansOutputValidator:
    """Require a successful KMeans Bundle result."""

    api_version = 1
    schema_digest = "9" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("KMeans output requires successful execution")
        if not outputs.get("bundle_uri"):
            raise ValueError("KMeans output requires bundle_uri")
        if not isinstance(outputs.get("n_clusters"), int):
            raise ValueError("KMeans output requires n_clusters")
        return value


class IsolationForestOutputValidator:
    """Require a successful anomaly-score Bundle result."""

    api_version = 1
    schema_digest = "b" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("Isolation Forest output requires successful execution")
        if not outputs.get("bundle_uri"):
            raise ValueError("Isolation Forest output requires bundle_uri")
        if "threshold" not in outputs:
            raise ValueError("Isolation Forest output requires threshold")
        return value


class UnsupervisedCoverageValidator:
    """Require complete distributed input coverage."""

    api_version = 1
    schema_digest = "a" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("input_complete") is not True:
            raise ValueError("unsupervised input coverage is incomplete")
        if value.get("distributed") is not True:
            raise ValueError("unsupervised execution did not prove distribution")
        return value


__all__ = [
    "KMeansOutputValidator",
    "IsolationForestOutputValidator",
    "PCAOutputValidator",
    "UnlabeledDenseInputValidator",
    "UnsupervisedConfigValidator",
    "UnsupervisedCoverageValidator",
]
