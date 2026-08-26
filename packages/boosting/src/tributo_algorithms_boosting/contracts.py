"""Executable contracts for official distributed XGBoost."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class XGBoostConfigValidator:
    api_version = 1
    schema_digest = "1" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"data", "model", "output", "ray", "resource", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown XGBoost config keys: {unknown}")
        for name in allowed:
            item = value.get(name)
            if item is not None and not isinstance(item, Mapping):
                raise ValueError(f"{name} must be a mapping")
        data = value.get("data")
        if not isinstance(data, Mapping) or not data.get("label_col"):
            raise ValueError("data.label_col is required")
        output = value.get("output")
        if not isinstance(output, Mapping) or not output.get("bundle_uri"):
            raise ValueError("output.bundle_uri is required")
        ray = value.get("ray")
        if not isinstance(ray, Mapping) or not ray.get("storage_path"):
            raise ValueError("ray.storage_path is required")
        return dict(value)


class XGBoostInputValidator:
    api_version = 1
    schema_digest = "2" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("XGBoost requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or not binding.get("feature_names")
            or not binding.get("label_name")
        ):
            raise ValueError("XGBoost requires features and label")
        return value


class XGBoostOutputValidator:
    api_version = 1
    schema_digest = "3" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("XGBoost execution failed")
        if not outputs.get("bundle_uri"):
            raise ValueError("XGBoost output requires Bundle")
        return value


class XGBoostCoverageValidator:
    api_version = 1
    schema_digest = "4" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("XGBoost distributed coverage is incomplete")
        return value


__all__ = [
    "XGBoostConfigValidator",
    "XGBoostCoverageValidator",
    "XGBoostInputValidator",
    "XGBoostOutputValidator",
]
