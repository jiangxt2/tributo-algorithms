"""Contracts for the bounded synchronous SGD algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SGDConfigValidator:
    """Validate the intentionally small synchronous SGD configuration."""

    api_version = 1
    schema_digest = "c" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {
            "alpha",
            "learning_rate",
            "learning_rate_decay",
            "loss",
            "max_iter",
            "output",
            "runtime",
            "seed",
            "tolerance",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown SGD config keys: {unknown}")
        for name in ("max_iter",):
            item = value.get(name)
            if item is not None and (
                not isinstance(item, int) or isinstance(item, bool) or item < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        seed = value.get("seed")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        for name in ("alpha", "learning_rate", "learning_rate_decay", "tolerance"):
            item = value.get(name)
            if item is not None and (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or float(item) < 0
                or (name != "learning_rate_decay" and float(item) <= 0)
            ):
                raise ValueError(f"{name} must be a valid number")
        loss = value.get("loss")
        if loss is not None and loss not in {"log_loss", "squared_error"}:
            raise ValueError("SGD loss must be log_loss or squared_error")
        output = value.get("output")
        if not isinstance(output, Mapping) or not isinstance(
            output.get("bundle_uri"), str
        ):
            raise ValueError("SGD output.bundle_uri is required")
        return dict(value)


class SGDInputValidator:
    """Require one dense labeled tabular input."""

    api_version = 1
    schema_digest = "d" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        descriptors = value.get("descriptors")
        if not isinstance(bindings, list) or not isinstance(descriptors, Mapping):
            raise ValueError("SGD input requires bindings and descriptors")
        binding = next(
            (
                item
                for item in bindings
                if isinstance(item, Mapping) and item.get("name") == "train"
            ),
            None,
        )
        if not isinstance(binding, Mapping) or not binding.get("feature_names"):
            raise ValueError("SGD input requires train features")
        if not isinstance(binding.get("label_name"), str) or not binding["label_name"]:
            raise ValueError("SGD input requires a label")
        return value


class SGDOutputValidator:
    """Require successful Bundle-backed SGD output."""

    api_version = 1
    schema_digest = "e" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("SGD output requires successful execution")
        if not outputs.get("bundle_uri"):
            raise ValueError("SGD output requires bundle_uri")
        return value


class SGDCoverageValidator:
    """Require complete distributed synchronous update coverage."""

    api_version = 1
    schema_digest = "f" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("input_complete") is not True:
            raise ValueError("SGD input coverage is incomplete")
        if value.get("distributed") is not True:
            raise ValueError("SGD execution did not prove distribution")
        return value


__all__ = [
    "SGDConfigValidator",
    "SGDCoverageValidator",
    "SGDInputValidator",
    "SGDOutputValidator",
]
