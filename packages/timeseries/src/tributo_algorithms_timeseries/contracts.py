"""Executable contracts for the official time-series Wheel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TimeSeriesConfigValidator:
    api_version = 1
    schema_digest = "5" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"metrics", "model", "optimizer", "output", "ray", "training"}
        unknown = sorted(set(value) - allowed)
        if unknown:
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
        if (
            not isinstance(binding, Mapping)
            or len(binding.get("feature_names", ())) < 2
        ):
            raise ValueError("time-series input requires at least two ordered lags")
        if not binding.get("label_name"):
            raise ValueError("time-series input requires a label")
        return value


class TimeSeriesOutputValidator:
    api_version = 1
    schema_digest = "7" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("status") != "succeeded":
            raise ValueError("time-series execution failed")
        outputs = value.get("outputs")
        if not isinstance(outputs, Mapping) or not outputs.get("bundle_uri"):
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


__all__ = [
    "TimeSeriesConfigValidator",
    "TimeSeriesOutputValidator",
    "WindowCoverageValidator",
    "WindowInputValidator",
]
