"""Executable config, input, output, and coverage contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ClassicalConfigValidator:
    """Validate the shared strict classical-algorithm configuration."""

    api_version = 1
    schema_digest = "5" * 64
    _ALLOWED = {
        "checkpoint",
        "C",
        "alpha",
        "class_prior",
        "class_weight",
        "feature_count",
        "fit_prior",
        "force_alpha",
        "learning_rate",
        "max_depth",
        "max_features",
        "n_estimators",
        "output",
        "runtime",
        "seed",
        "task",
        "task_count",
        "tolerance",
        "unit_count",
    }

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = sorted(set(value) - self._ALLOWED)
        if unknown:
            raise ValueError(f"unknown classical config keys: {unknown}")
        normalized = dict(value)
        task = normalized.get("task")
        if task is not None and task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        for name, minimum in (
            ("n_estimators", 1),
            ("unit_count", 1),
            ("seed", 0),
            ("feature_count", 1),
            ("task_count", 1),
        ):
            item = normalized.get(name)
            if item is None:
                continue
            if not isinstance(item, int) or isinstance(item, bool) or item < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for numeric_name, numeric_minimum in (
            ("learning_rate", 0.0),
            ("tolerance", 0.0),
            ("C", 0.0),
        ):
            numeric_item = normalized.get(numeric_name)
            if numeric_item is None:
                continue
            if (
                not isinstance(numeric_item, (int, float))
                or isinstance(numeric_item, bool)
                or float(numeric_item) <= numeric_minimum
            ):
                raise ValueError(
                    f"{numeric_name} must be greater than {numeric_minimum}"
                )
            normalized[numeric_name] = float(numeric_item)
        output = normalized.get("output")
        if output is not None and (
            not isinstance(output, Mapping)
            or not isinstance(output.get("bundle_uri"), str)
            or not output["bundle_uri"]
        ):
            raise ValueError("output.bundle_uri must be non-empty")
        runtime = normalized.get("runtime")
        if runtime is not None and not isinstance(runtime, Mapping):
            raise ValueError("runtime must be a mapping")
        return normalized


class TabularInputValidator:
    """Validate one labeled train input without opening the data source."""

    api_version = 1
    schema_digest = "2" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        descriptors = value.get("descriptors")
        if not isinstance(bindings, list) or not isinstance(descriptors, Mapping):
            raise ValueError("tabular input requires bindings and descriptors")
        roles = {
            binding.get("name") for binding in bindings if isinstance(binding, Mapping)
        }
        if "train" not in roles or "train" not in descriptors:
            raise ValueError("tabular input requires the train role")
        return value


class SklearnOutputValidator:
    """Require successful Bundle-backed fit output."""

    api_version = 1
    schema_digest = "3" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("status") != "succeeded":
            raise ValueError("classical output requires successful execution")
        outputs = value.get("outputs")
        if not isinstance(outputs, Mapping) or not outputs.get("bundle_uri"):
            raise ValueError("classical output requires bundle_uri")
        return value


class DistributedCoverageValidator:
    """Require complete model-internal distributed execution evidence."""

    api_version = 1
    schema_digest = "4" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("input_complete") is not True:
            raise ValueError("classical distributed input coverage is incomplete")
        if value.get("distributed") is not True:
            raise ValueError("classical execution did not prove distributed training")
        return value


__all__ = [
    "ClassicalConfigValidator",
    "DistributedCoverageValidator",
    "SklearnOutputValidator",
    "TabularInputValidator",
]
