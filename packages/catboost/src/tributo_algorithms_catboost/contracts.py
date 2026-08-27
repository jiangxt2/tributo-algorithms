"""Executable contracts for the CatBoost algorithm package."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CatBoostConfigValidator:
    """Validate the intentionally narrow CatBoost configuration surface."""

    api_version = 1
    schema_digest = "5" * 64
    _MODEL_ALLOWED = {"cat_features", "depth", "iterations", "learning_rate"}

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {
            "model",
            "n_estimators",
            "output",
            "runtime",
            "seed",
            "task",
            "training",
            "unit_count",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown CatBoost config keys: {unknown}")
        for name in {"model", "output", "runtime", "training"}:
            item = value.get(name)
            if item is not None and not isinstance(item, Mapping):
                raise ValueError(f"{name} must be a mapping")
        model = value.get("model", {})
        if not isinstance(model, Mapping):
            raise ValueError("CatBoost model must be a mapping")
        unknown_model = sorted(set(model) - self._MODEL_ALLOWED)
        if unknown_model:
            raise ValueError(f"unknown CatBoost model keys: {unknown_model}")
        for name in ("depth", "iterations"):
            item = model.get(name)
            if item is not None and (
                not isinstance(item, int) or isinstance(item, bool) or item < 1
            ):
                raise ValueError(f"model.{name} must be a positive integer")
        learning_rate = model.get("learning_rate")
        if learning_rate is not None and (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or not 0 < float(learning_rate)
        ):
            raise ValueError("model.learning_rate must be positive")
        cat_features = model.get("cat_features", ())
        if not isinstance(cat_features, (list, tuple)) or any(
            not isinstance(item, (str, int)) or isinstance(item, bool)
            for item in cat_features
        ):
            raise ValueError("model.cat_features must contain names or indices")
        if len(set(cat_features)) != len(cat_features):
            raise ValueError("model.cat_features must be unique")
        task = value.get("task")
        if task is not None and task not in {"classification", "regression"}:
            raise ValueError("CatBoost task must be classification or regression")
        for name in ("n_estimators", "unit_count"):
            item = value.get(name)
            if item is not None and (
                not isinstance(item, int) or isinstance(item, bool) or item < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        seed = value.get("seed")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ValueError("CatBoost seed must be a non-negative integer")
        output = value.get("output")
        if (
            not isinstance(output, Mapping)
            or not isinstance(output.get("bundle_uri"), str)
            or not output["bundle_uri"]
        ):
            raise ValueError("CatBoost output.bundle_uri is required")
        return dict(value)


class CatBoostInputValidator:
    """Require one labeled feature-only-compatible tabular role."""

    api_version = 1
    schema_digest = "6" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        descriptors = value.get("descriptors")
        if not isinstance(bindings, list) or not isinstance(descriptors, Mapping):
            raise ValueError("CatBoost input requires bindings and descriptors")
        binding = next(
            (
                item
                for item in bindings
                if isinstance(item, Mapping) and item.get("name") == "train"
            ),
            None,
        )
        if not isinstance(binding, Mapping) or not binding.get("feature_names"):
            raise ValueError("CatBoost input requires train feature columns")
        if not isinstance(binding.get("label_name"), str) or not binding["label_name"]:
            raise ValueError("CatBoost input requires a label")
        return value


class CatBoostOutputValidator:
    """Require successful CatBoost Bundle output."""

    api_version = 1
    schema_digest = "7" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("CatBoost output requires successful execution")
        if not outputs.get("bundle_uri"):
            raise ValueError("CatBoost output requires bundle_uri")
        return value


class CatBoostCoverageValidator:
    """Require complete distributed CatBoost ensemble coverage."""

    api_version = 1
    schema_digest = "8" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("input_complete") is not True:
            raise ValueError("CatBoost input coverage is incomplete")
        if value.get("distributed") is not True:
            raise ValueError("CatBoost execution did not prove distribution")
        return value


__all__ = [
    "CatBoostConfigValidator",
    "CatBoostCoverageValidator",
    "CatBoostInputValidator",
    "CatBoostOutputValidator",
]
