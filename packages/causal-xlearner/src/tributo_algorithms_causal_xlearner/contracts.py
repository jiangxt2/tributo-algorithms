"""Executable contracts for official X-Learner."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _validate_propensity_clip(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numeric values")
    lower = _finite_number(value[0], name)
    upper = _finite_number(value[1], name)
    if not 0.0 < lower < upper < 1.0:
        raise ValueError(f"{name} must satisfy 0 < lower < upper < 1")
    return lower, upper


def _validate_response_threshold(value: object, name: str) -> float:
    return _finite_number(value, name)


class XLearnerConfigValidator:
    api_version = 1
    schema_digest = "6" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"data", "model", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown X-Learner config keys: {unknown}")
        training = value.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("training config is required")
        folds = training.get("cross_fit_folds", 5)
        if (
            not isinstance(folds, int)
            or isinstance(folds, bool)
            or not 2 <= folds <= 20
        ):
            raise ValueError("training.cross_fit_folds must be between 2 and 20")
        _validate_propensity_clip(
            training.get("propensity_clip", (0.01, 0.99)),
            "training.propensity_clip",
        )
        _validate_response_threshold(
            training.get("response_threshold", 0.5),
            "training.response_threshold",
        )
        for name in allowed:
            if not isinstance(value.get(name), Mapping):
                raise ValueError(f"{name} config is required")
        data = value["data"]
        for name in ("feature_columns", "treatment_col", "outcome_col", "identity_col"):
            if not data.get(name):
                raise ValueError(f"data.{name} is required")
        if not value["ray"].get("storage_path") or not value["output"].get(
            "bundle_uri"
        ):
            raise ValueError("ray.storage_path and output.bundle_uri are required")
        return dict(value)


class XLearnerInputValidator:
    api_version = 1
    schema_digest = "2" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("X-Learner requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or len(binding.get("feature_names", ())) < 3
            or not binding.get("label_name")
        ):
            raise ValueError("X-Learner input roles are incomplete")
        return value


class XLearnerOutputValidator:
    api_version = 1
    schema_digest = "3" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("X-Learner execution failed")
        if not outputs.get("bundle_uri") or not outputs.get("composition_digest"):
            raise ValueError("X-Learner output requires Bundle and composition digest")
        return value


class XLearnerCoverageValidator:
    api_version = 1
    schema_digest = "4" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("X-Learner coverage is incomplete")
        state = value.get("state")
        details = state.get("details") if isinstance(state, Mapping) else None
        if not isinstance(details, Mapping):
            raise ValueError("X-Learner stage evidence is missing")
        expected = "mu0,mu1,tau0,tau1,propensity"
        if details.get("component_stages") != expected:
            raise ValueError("X-Learner stage set drifted")
        return value


__all__ = [
    "XLearnerConfigValidator",
    "XLearnerCoverageValidator",
    "XLearnerInputValidator",
    "XLearnerOutputValidator",
]
