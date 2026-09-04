"""Versioned contracts for the tabular autoencoder."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def _digest(name: str) -> str:
    return hashlib.sha256(f"tributo.representation.{name}.v2".encode()).hexdigest()


class RepresentationConfigValidator:
    api_version = 1
    schema_digest = "9" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"metrics", "model", "optimizer", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown representation config keys: {unknown}")
        for name in allowed:
            item = value.get(name)
            if item is not None and not isinstance(item, Mapping):
                raise ValueError(f"{name} must be a mapping")
        if not isinstance(value.get("output"), Mapping) or not isinstance(
            value["output"].get("bundle_uri"), str
        ):
            raise ValueError("representation output.bundle_uri is required")
        optimizer = value.get("optimizer", {})
        if isinstance(optimizer, Mapping):
            accumulation = optimizer.get("accumulation_steps", 1)
            if (
                not isinstance(accumulation, int)
                or isinstance(accumulation, bool)
                or accumulation < 1
            ):
                raise ValueError("Autoencoder accumulation_steps must be positive")
        return dict(value)


class AutoencoderTensorInputValidator:
    api_version = 1
    schema_digest = _digest("tensor-input-role-routed")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or not 1 <= len(bindings) <= 3:
            raise ValueError(
                "Autoencoder requires train and optional val/test bindings"
            )
        if any(not isinstance(binding, Mapping) for binding in bindings):
            raise ValueError("Autoencoder bindings must be mappings")
        names = [binding.get("name") for binding in bindings]
        if any(not isinstance(name, str) for name in names):
            raise ValueError("Autoencoder input roles must be named")
        by_role = {str(binding["name"]): binding for binding in bindings}
        if (
            len(by_role) != len(bindings)
            or "train" not in by_role
            or not set(by_role).issubset({"train", "val", "test"})
            or value.get("primary_role") != "train"
        ):
            raise ValueError("Autoencoder input roles are invalid")
        expected_features = by_role["train"].get("feature_names")
        for binding in by_role.values():
            features = binding.get("feature_names")
            if (
                not isinstance(features, list)
                or not features
                or features != expected_features
            ):
                raise ValueError(
                    "Autoencoder requires consistent named dense feature columns"
                )
            if (
                binding.get("label_name") is not None
                or binding.get("sample_weight_name") is not None
            ):
                raise ValueError(
                    "Autoencoder does not accept label or sample-weight bindings"
                )
        return value


class RepresentationOutputValidator:
    api_version = 1
    schema_digest = "b" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("representation execution failed")
        if not isinstance(outputs.get("bundle_uri"), str) or not outputs["bundle_uri"]:
            raise ValueError("representation output requires Bundle")
        return value


class AutoencoderTorchCoverageValidator:
    api_version = 1
    schema_digest = _digest("coverage")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("Autoencoder input coverage is incomplete")
        return value


__all__ = [
    "AutoencoderTensorInputValidator",
    "AutoencoderTorchCoverageValidator",
    "RepresentationConfigValidator",
    "RepresentationOutputValidator",
]
