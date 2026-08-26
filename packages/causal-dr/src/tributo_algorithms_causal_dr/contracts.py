"""Executable contracts for distributed doubly robust estimation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DRConfigValidator:
    api_version = 1
    schema_digest = "e" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"data", "model", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown DR config keys: {unknown}")
        for name in allowed:
            if not isinstance(value.get(name), Mapping):
                raise ValueError(f"{name} config is required")
        data = value["data"]
        for name in ("feature_columns", "treatment_col", "outcome_col"):
            if not data.get(name):
                raise ValueError(f"data.{name} is required")
        training = value["training"]
        folds = training.get("cross_fit_folds", 5)
        if (
            not isinstance(folds, int)
            or isinstance(folds, bool)
            or not 2 <= folds <= 20
        ):
            raise ValueError("training.cross_fit_folds must be between 2 and 20")
        if not value["ray"].get("storage_path") or not value["output"].get(
            "bundle_uri"
        ):
            raise ValueError("ray.storage_path and output.bundle_uri are required")
        return dict(value)


class DRInputValidator:
    api_version = 1
    schema_digest = "b" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("DR estimator requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or len(binding.get("feature_names", ())) < 2
            or not binding.get("label_name")
        ):
            raise ValueError("DR input requires features, treatment, and outcome")
        return value


class DROutputValidator:
    api_version = 1
    schema_digest = "c" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("DR estimation failed")
        if not outputs.get("bundle_uri") or not outputs.get("composition_digest"):
            raise ValueError("DR output requires Bundle and composition digest")
        return value


class DRCoverageValidator:
    api_version = 1
    schema_digest = "d" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("DR coverage is incomplete")
        state = value.get("state")
        details = state.get("details") if isinstance(state, Mapping) else None
        if not isinstance(details, Mapping):
            raise ValueError("DR stage evidence is missing")
        if details.get("component_stages") != "mu0,mu1,propensity":
            raise ValueError("DR stage set drifted")
        return value


__all__ = [
    "DRConfigValidator",
    "DRCoverageValidator",
    "DRInputValidator",
    "DROutputValidator",
]
