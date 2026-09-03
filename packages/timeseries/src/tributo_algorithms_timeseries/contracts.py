"""Versioned contracts for the temporal convolution TorchRecipe."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def _digest(name: str) -> str:
    return hashlib.sha256(f"tributo.timeseries.{name}.v2".encode()).hexdigest()


class TimeSeriesConfigValidator:
    api_version = 1
    schema_digest = "5" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"metrics", "model", "optimizer", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown time-series config keys: {unknown}")
        for name in allowed:
            item = value.get(name)
            if item is not None and not isinstance(item, Mapping):
                raise ValueError(f"{name} must be a mapping")
        output = value.get("output")
        if not isinstance(output, Mapping) or not isinstance(
            output.get("bundle_uri"), str
        ):
            raise ValueError("output.bundle_uri is required")
        return dict(value)


class WindowInputValidator:
    api_version = 1
    schema_digest = "6" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("time-series training requires one train binding")
        binding = bindings[0]
        if not isinstance(binding, Mapping) or binding.get("role", "train") != "train":
            raise ValueError("time-series input requires a train binding")
        features = binding.get("feature_names")
        if not isinstance(features, list) or len(features) < 2:
            raise ValueError("time-series input requires at least two ordered lags")
        if not isinstance(binding.get("label_name"), str) or not binding["label_name"]:
            raise ValueError("time-series input requires a label")
        if binding.get("sample_weight_name") is not None:
            raise ValueError(
                "time-series algorithms do not support sample-weight binding"
            )
        return value


class TimeSeriesOutputValidator:
    api_version = 1
    schema_digest = "7" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("time-series execution failed")
        if not isinstance(outputs.get("bundle_uri"), str) or not outputs["bundle_uri"]:
            raise ValueError("time-series output requires a Bundle")
        return value


class WindowCoverageValidator:
    api_version = 1
    schema_digest = "8" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("time-series window coverage is incomplete")
        return value


class TemporalConvTensorInputValidator(WindowInputValidator):
    api_version = 1
    schema_digest = _digest("tcn-window-input")


class TemporalConvTorchCoverageValidator(WindowCoverageValidator):
    api_version = 1
    schema_digest = _digest("tcn-coverage")


__all__ = [
    "TimeSeriesConfigValidator",
    "TemporalConvTensorInputValidator",
    "TemporalConvTorchCoverageValidator",
    "TimeSeriesOutputValidator",
    "WindowCoverageValidator",
    "WindowInputValidator",
]
