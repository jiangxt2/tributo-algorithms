"""Executable contracts for self-supervised representation algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RepresentationConfigValidator:
    api_version = 1
    schema_digest = "9" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"metrics", "model", "optimizer", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown representation config keys: {unknown}")
        if not isinstance(value.get("output"), Mapping):
            raise ValueError("representation output config is required")
        return dict(value)


class SelfSupervisedInputValidator:
    api_version = 1
    schema_digest = "a" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("Autoencoder requires one train binding")
        binding = bindings[0]
        if not isinstance(binding, Mapping) or not binding.get("feature_names"):
            raise ValueError("Autoencoder requires dense feature columns")
        return value


class RepresentationOutputValidator:
    api_version = 1
    schema_digest = "b" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("representation execution failed")
        if not outputs.get("bundle_uri"):
            raise ValueError("representation output requires Bundle")
        return value


class SelfSupervisedCoverageValidator:
    api_version = 1
    schema_digest = "c" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("self-supervised input coverage is incomplete")
        return value


__all__ = [
    "RepresentationConfigValidator",
    "RepresentationOutputValidator",
    "SelfSupervisedCoverageValidator",
    "SelfSupervisedInputValidator",
]
