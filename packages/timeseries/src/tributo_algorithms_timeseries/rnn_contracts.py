"""Versioned contracts for fixed-window recurrent TorchRecipe algorithms."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any


def _digest(name: str) -> str:
    return hashlib.sha256(f"tributo.timeseries.rnn.{name}.v2".encode()).hexdigest()


class RNNConfigValidator:
    api_version = 1
    schema_digest = "9" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"metrics", "model", "optimizer", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown RNN config keys: {unknown}")
        for name in allowed:
            item = value.get(name)
            if item is not None and not isinstance(item, Mapping):
                raise ValueError(f"{name} must be a mapping")
        optimizer = value.get("optimizer", {})
        if isinstance(optimizer, Mapping):
            learning_rate = optimizer.get("learning_rate", 0.001)
            weight_decay = optimizer.get("weight_decay", 0.0)
            accumulation = optimizer.get("accumulation_steps", 1)
            max_gradient_norm = optimizer.get("max_gradient_norm", 1.0)
            if (
                not isinstance(learning_rate, (int, float))
                or isinstance(learning_rate, bool)
                or not math.isfinite(float(learning_rate))
                or float(learning_rate) <= 0
            ):
                raise ValueError("optimizer.learning_rate must be positive and finite")
            if (
                not isinstance(accumulation, int)
                or isinstance(accumulation, bool)
                or accumulation < 1
            ):
                raise ValueError("optimizer.accumulation_steps must be positive")
            if (
                not isinstance(weight_decay, (int, float))
                or isinstance(weight_decay, bool)
                or weight_decay < 0
            ):
                raise ValueError("optimizer.weight_decay must be non-negative")
            if (
                not isinstance(max_gradient_norm, (int, float))
                or isinstance(max_gradient_norm, bool)
                or max_gradient_norm <= 0
            ):
                raise ValueError("optimizer.max_gradient_norm must be positive")
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
    api_version = 1
    schema_digest = "a" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or not 1 <= len(bindings) <= 3:
            raise ValueError("RNN input requires train and optional val/test bindings")
        if any(not isinstance(binding, Mapping) for binding in bindings):
            raise ValueError("RNN bindings must be mappings")
        names = [binding.get("name") for binding in bindings]
        if any(not isinstance(name, str) for name in names):
            raise ValueError("RNN input roles must be named")
        by_role = {str(binding["name"]): binding for binding in bindings}
        if (
            len(by_role) != len(bindings)
            or "train" not in by_role
            or not set(by_role).issubset({"train", "val", "test"})
            or value.get("primary_role") != "train"
        ):
            raise ValueError("RNN input roles are invalid")
        train = by_role["train"]
        expected_features = train.get("feature_names")
        expected_label = train.get("label_name")
        for binding in by_role.values():
            features = binding.get("feature_names")
            if (
                not isinstance(features, list)
                or len(features) < 2
                or features != expected_features
            ):
                raise ValueError("RNN input requires consistent ordered lags")
            if binding.get("label_name") != expected_label or not isinstance(
                expected_label, str
            ):
                raise ValueError("RNN input requires a consistent label")
            if binding.get("sample_weight_name") is not None:
                raise ValueError("RNN algorithms do not support sample-weight binding")
        return value


class RNNOutputValidator:
    api_version = 1
    schema_digest = "b" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("RNN execution failed")
        if not isinstance(outputs.get("bundle_uri"), str) or not outputs["bundle_uri"]:
            raise ValueError("RNN output requires a Bundle")
        return value


class RNNCoverageValidator:
    api_version = 1
    schema_digest = "c" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("RNN window coverage is incomplete")
        return value


class LSTMTensorInputValidator(FixedWindowInputValidator):
    api_version = 1
    schema_digest = _digest("lstm-window-input-role-routed")


class GRUTensorInputValidator(FixedWindowInputValidator):
    api_version = 1
    schema_digest = _digest("gru-window-input-role-routed")


class LSTMTorchCoverageValidator(RNNCoverageValidator):
    api_version = 1
    schema_digest = _digest("lstm-coverage")


class GRUTorchCoverageValidator(RNNCoverageValidator):
    api_version = 1
    schema_digest = _digest("gru-coverage")


__all__ = [
    "FixedWindowInputValidator",
    "GRUTensorInputValidator",
    "GRUTorchCoverageValidator",
    "LSTMTensorInputValidator",
    "LSTMTorchCoverageValidator",
    "RNNCoverageValidator",
    "RNNConfigValidator",
    "RNNOutputValidator",
]
