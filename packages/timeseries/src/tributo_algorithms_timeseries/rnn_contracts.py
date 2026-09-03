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
            accumulation = optimizer.get("accumulation_steps", 1)
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
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("RNN training requires one train binding")
        binding = bindings[0]
        if not isinstance(binding, Mapping) or binding.get("role", "train") != "train":
            raise ValueError("RNN input requires a train binding")
        features = binding.get("feature_names")
        if not isinstance(features, list) or len(features) < 2:
            raise ValueError("RNN input requires at least two ordered lags")
        if not isinstance(binding.get("label_name"), str) or not binding["label_name"]:
            raise ValueError("RNN input requires a label")
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
    schema_digest = _digest("lstm-window-input")


class GRUTensorInputValidator(FixedWindowInputValidator):
    api_version = 1
    schema_digest = _digest("gru-window-input")


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
