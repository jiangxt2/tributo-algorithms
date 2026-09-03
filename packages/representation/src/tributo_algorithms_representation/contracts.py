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
        if (
            isinstance(optimizer, Mapping)
            and "accumulation_steps" in optimizer
            and optimizer["accumulation_steps"] != 1
        ):
            raise ValueError("Autoencoder accumulation_steps must be positive")
        return dict(value)


class AutoencoderTensorInputValidator:
    api_version = 1
    schema_digest = _digest("tensor-input")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("Autoencoder requires one train binding")
        binding = bindings[0]
        if not isinstance(binding, Mapping) or binding.get("role", "train") != "train":
            raise ValueError("Autoencoder requires a train binding")
        features = binding.get("feature_names")
        if not isinstance(features, list) or not features:
            raise ValueError("Autoencoder requires named dense feature columns")
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
