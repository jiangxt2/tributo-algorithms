"""Executable contracts for official distributed XGBoost."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _validate_training_options(
    training: Mapping[str, Any], model: Mapping[str, Any]
) -> tuple[int, int | None, bool | None]:
    if unknown := sorted(
        set(training) - {"num_rounds", "early_stopping_rounds", "maximize"}
    ):
        raise ValueError(f"unknown XGBoost training keys: {unknown}")
    rounds = training.get("num_rounds", 10)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("XGBoost num_rounds/num_boost_round must be positive")
    patience = training.get("early_stopping_rounds")
    if patience is None:
        if "maximize" in training:
            raise ValueError("training.maximize requires early stopping")
        return rounds, None, None
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
        raise ValueError("training.early_stopping_rounds must be a positive integer")
    maximize = training.get("maximize")
    if not isinstance(maximize, bool):
        raise ValueError("early stopping requires an explicit training.maximize bool")
    metric = model.get("eval_metric")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("early stopping requires a single model.eval_metric string")
    if model.get("booster", "gbtree") == "gblinear":
        raise ValueError("early stopping best-model export requires a tree Booster")
    return rounds, patience, maximize


def _validate_training_metadata(value: object) -> dict[str, Any]:
    required = {"schema_version", "training_rounds", "model_rounds"}
    selection = {"best_iteration", "best_score", "selection_metric", "maximize"}
    if not isinstance(value, Mapping) or set(value) not in (
        required,
        required | selection,
    ):
        raise ValueError("XGBoost training metadata fields are invalid")
    for field in required:
        item = value[field]
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise ValueError(f"XGBoost training metadata {field} must be positive")
    if value["schema_version"] != 1:
        raise ValueError("unsupported XGBoost training metadata version")
    if value["model_rounds"] > value["training_rounds"]:
        raise ValueError("XGBoost model rounds exceed executed training rounds")
    if selection <= set(value):
        best = value["best_iteration"]
        score = value["best_score"]
        metric = value["selection_metric"]
        if (
            not isinstance(best, int)
            or isinstance(best, bool)
            or best < 0
            or best + 1 != value["model_rounds"]
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not isinstance(metric, str)
            or not metric.startswith("validation-")
            or len(metric) == len("validation-")
            or not isinstance(value["maximize"], bool)
        ):
            raise ValueError("XGBoost best-model metadata is inconsistent")
    return dict(value)


class XGBoostConfigValidator:
    api_version = 1
    schema_digest = "1" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"data", "model", "output", "ray", "resource", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown XGBoost config keys: {unknown}")
        for name in allowed:
            if name in value and not isinstance(value[name], Mapping):
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
        _validate_training_options(value.get("training", {}), value.get("model", {}))
        return dict(value)


class XGBoostInputValidator:
    api_version = 1
    schema_digest = "2" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) not in {1, 2}:
            raise ValueError("XGBoost requires train and optional validation bindings")
        by_role: dict[str, Mapping[str, Any]] = {}
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError("XGBoost input binding must be a mapping")
            role = binding.get("name")
            if (
                not isinstance(role, str)
                or role not in {"train", "validation"}
                or role in by_role
            ):
                raise ValueError("XGBoost input roles must be unique train/validation")
            features = binding.get("feature_names")
            label = binding.get("label_name")
            if (
                not isinstance(features, (list, tuple))
                or not features
                or any(not isinstance(name, str) or not name for name in features)
                or len(set(features)) != len(features)
                or not isinstance(label, str)
                or not label
            ):
                raise ValueError("XGBoost requires features and label")
            by_role[role] = binding
        if "train" not in by_role or value.get("primary_role", "train") != "train":
            raise ValueError("XGBoost requires train as the primary input role")
        train = by_role["train"]
        if "validation" in by_role:
            validation = by_role["validation"]
            if (
                tuple(validation["feature_names"]) != tuple(train["feature_names"])
                or validation["label_name"] != train["label_name"]
            ):
                raise ValueError(
                    "XGBoost validation feature order and label must match train"
                )
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
        metrics = value.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("XGBoost result metrics must be a mapping")
        fields = {
            "training_rounds",
            "model_rounds",
            "best_iteration",
            "best_score",
            "selection_metric",
            "maximize",
        }
        if set(metrics) & (fields - {"training_rounds"}):
            _validate_training_metadata(
                {
                    "schema_version": 1,
                    **{key: metrics[key] for key in fields if key in metrics},
                }
            )
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
