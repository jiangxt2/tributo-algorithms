"""Contracts for pre-tokenized Transformer classification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TransformerConfigValidator:
    api_version = 1
    schema_digest = "d" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"metrics", "model", "optimizer", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown Transformer config keys: {unknown}")
        if not isinstance(value.get("output"), Mapping):
            raise ValueError("Transformer output config is required")
        return dict(value)


class TokenInputValidator:
    api_version = 1
    schema_digest = "e" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("Transformer requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or len(binding.get("feature_names", ())) < 2
        ):
            raise ValueError("Transformer requires ordered token columns")
        if not binding.get("label_name"):
            raise ValueError("Transformer classification requires a label")
        return value


class TransformerOutputValidator:
    api_version = 1
    schema_digest = "f" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("Transformer execution failed")
        if not outputs.get("bundle_uri"):
            raise ValueError("Transformer output requires Bundle")
        return value


class TokenCoverageValidator:
    api_version = 1
    schema_digest = "0" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("Transformer token coverage is incomplete")
        return value


__all__ = [
    "TokenCoverageValidator",
    "TokenInputValidator",
    "TransformerConfigValidator",
    "TransformerOutputValidator",
]
