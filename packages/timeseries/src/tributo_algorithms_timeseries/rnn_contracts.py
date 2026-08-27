"""Contracts for fixed-window recurrent time-series algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RNNConfigValidator:
    """Validate the bounded RNN configuration namespaces."""

    api_version = 1
    schema_digest = "9" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"metrics", "model", "optimizer", "output", "ray", "training"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown RNN config keys: {unknown}")
        for name in allowed:
            item = value.get(name)
            if item is not None and not isinstance(item, Mapping):
                raise ValueError(f"{name} must be a mapping")
        optimizer = value.get("optimizer", {})
        if not isinstance(optimizer, Mapping):
            raise ValueError("optimizer must be a mapping")
        learning_rate = optimizer.get("learning_rate", 0.001)
        weight_decay = optimizer.get("weight_decay", 0.0)
        accumulation_steps = optimizer.get("accumulation_steps", 1)
        max_gradient_norm = optimizer.get("max_gradient_norm", 1.0)
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or learning_rate <= 0
            or not isinstance(weight_decay, (int, float))
            or isinstance(weight_decay, bool)
            or weight_decay < 0
            or not isinstance(accumulation_steps, int)
            or isinstance(accumulation_steps, bool)
            or accumulation_steps < 1
            or not isinstance(max_gradient_norm, (int, float))
            or isinstance(max_gradient_norm, bool)
            or max_gradient_norm <= 0
        ):
            raise ValueError("optimizer parameters are invalid")
        model = value.get("model", {})
        if isinstance(model, Mapping):
            for name in ("input_features", "hidden_size", "num_layers"):
                item = model.get(name)
                if item is not None and (
                    not isinstance(item, int) or isinstance(item, bool) or item < 1
                ):
                    raise ValueError(f"model.{name} must be a positive integer")
        output = value.get("output")
        if not isinstance(output, Mapping) or not isinstance(
            output.get("bundle_uri"), str
        ):
            raise ValueError("output.bundle_uri is required")
        return dict(value)


class FixedWindowInputValidator:
    """Require a labeled, ordered, fixed-width time-series window."""

    api_version = 1
    schema_digest = "a" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("RNN training requires one train binding")
        binding = bindings[0]
        if not isinstance(binding, Mapping):
            raise ValueError("RNN input binding is invalid")
        features = binding.get("feature_names")
        if not isinstance(features, list) or len(features) < 2:
            raise ValueError("RNN input requires at least two ordered lags")
        if not isinstance(binding.get("label_name"), str) or not binding["label_name"]:
            raise ValueError("RNN input requires a label")
        return value


class RNNOutputValidator:
    """Require successful Bundle-backed recurrent model output."""

    api_version = 1
    schema_digest = "b" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("RNN execution failed")
        if not outputs.get("bundle_uri"):
            raise ValueError("RNN output requires a Bundle")
        return value


class RNNCoverageValidator:
    """Require complete distributed fixed-window coverage."""

    api_version = 1
    schema_digest = "c" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("RNN window coverage is incomplete")
        return value


__all__ = [
    "FixedWindowInputValidator",
    "RNNCoverageValidator",
    "RNNConfigValidator",
    "RNNOutputValidator",
]
