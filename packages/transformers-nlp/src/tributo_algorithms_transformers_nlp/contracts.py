"""Versioned contracts for the pre-tokenized Transformer classifier."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def _digest(name: str) -> str:
    return hashlib.sha256(f"tributo.transformer.{name}.v2".encode()).hexdigest()


class TransformerConfigValidator:
    api_version = 1
    schema_digest = "d" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"metrics", "model", "optimizer", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown Transformer config keys: {unknown}")
        for name in allowed:
            item = value.get(name)
            if item is not None and not isinstance(item, Mapping):
                raise ValueError(f"{name} must be a mapping")
        if not isinstance(value.get("output"), Mapping) or not isinstance(
            value["output"].get("bundle_uri"), str
        ):
            raise ValueError("Transformer output.bundle_uri is required")
        return dict(value)


class TokenTensorInputValidator:
    api_version = 1
    schema_digest = _digest("tokens-role-routed")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or not 1 <= len(bindings) <= 3:
            raise ValueError(
                "Transformer requires train and optional val/test bindings"
            )
        if any(not isinstance(binding, Mapping) for binding in bindings):
            raise ValueError("Transformer bindings must be mappings")
        names = [binding.get("name") for binding in bindings]
        if any(not isinstance(name, str) for name in names):
            raise ValueError("Transformer input roles must be named")
        by_role = {str(binding["name"]): binding for binding in bindings}
        if (
            len(by_role) != len(bindings)
            or "train" not in by_role
            or not set(by_role).issubset({"train", "val", "test"})
            or value.get("primary_role") != "train"
        ):
            raise ValueError("Transformer input roles are invalid")
        train = by_role["train"]
        expected_features = train.get("feature_names")
        expected_label = train.get("label_name")
        for binding in by_role.values():
            features = binding.get("feature_names")
            if (
                not isinstance(features, list)
                or not features
                or features != expected_features
            ):
                raise ValueError(
                    "Transformer requires consistent ordered token columns"
                )
            if binding.get("label_name") != expected_label or not isinstance(
                expected_label, str
            ):
                raise ValueError("Transformer classification requires a label")
            if binding.get("sample_weight_name") is not None:
                raise ValueError("Transformer does not support sample-weight binding")
        return value


class TransformerOutputValidator:
    api_version = 1
    schema_digest = "f" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("Transformer execution failed")
        if not isinstance(outputs.get("bundle_uri"), str) or not outputs["bundle_uri"]:
            raise ValueError("Transformer output requires Bundle")
        return value


class TransformerTorchCoverageValidator:
    api_version = 1
    schema_digest = _digest("coverage")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("Transformer token coverage is incomplete")
        return value


__all__ = [
    "TokenTensorInputValidator",
    "TransformerConfigValidator",
    "TransformerOutputValidator",
    "TransformerTorchCoverageValidator",
]
