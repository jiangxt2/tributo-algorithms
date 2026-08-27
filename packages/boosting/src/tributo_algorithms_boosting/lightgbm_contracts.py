"""Contracts for the official LightGBM integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class LightGBMConfigValidator:
    """Validate the bounded LightGBM configuration envelope."""

    api_version = 1
    schema_digest = "5" * 64
    _MODEL_ALLOWED = {
        "bagging_fraction",
        "bagging_freq",
        "feature_fraction",
        "lambda_l1",
        "lambda_l2",
        "learning_rate",
        "max_bin",
        "max_depth",
        "min_data_in_leaf",
        "num_class",
        "num_leaves",
        "num_threads",
        "objective",
        "task",
        "verbosity",
    }

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"data", "model", "output", "ray", "resource", "training"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown LightGBM config keys: {unknown}")
        for name in allowed:
            item = value.get(name)
            if item is not None and not isinstance(item, Mapping):
                raise ValueError(f"{name} must be a mapping")
        data = value.get("data")
        if (
            not isinstance(data, Mapping)
            or not isinstance(data.get("label_col"), str)
            or not data["label_col"]
        ):
            raise ValueError("LightGBM data.label_col is required")
        feature_columns = data.get("feature_columns")
        if feature_columns is not None and (
            not isinstance(feature_columns, (list, tuple))
            or not feature_columns
            or any(not isinstance(name, str) or not name for name in feature_columns)
        ):
            raise ValueError("LightGBM data.feature_columns must be non-empty names")
        model = value.get("model", {})
        if not isinstance(model, Mapping):
            raise ValueError("LightGBM model must be a mapping")
        unknown_model = sorted(set(model) - self._MODEL_ALLOWED)
        if unknown_model:
            raise ValueError(f"unknown LightGBM model keys: {unknown_model}")
        task = model.get("task")
        if task is not None and task not in {"classification", "regression"}:
            raise ValueError("LightGBM model.task must be classification or regression")
        objective = model.get("objective")
        if objective is not None and objective not in {
            "binary",
            "multiclass",
            "multiclassova",
            "multiclassova2",
            "regression",
        }:
            raise ValueError("LightGBM model.objective is outside the supported subset")
        if task == "classification" and objective == "regression":
            raise ValueError(
                "LightGBM classification conflicts with regression objective"
            )
        if task == "regression" and objective in {
            "binary",
            "multiclass",
            "multiclassova",
            "multiclassova2",
        }:
            raise ValueError(
                "LightGBM regression conflicts with classification objective"
            )
        num_class = model.get("num_class")
        if num_class is not None and (
            not isinstance(num_class, int)
            or isinstance(num_class, bool)
            or num_class < 2
        ):
            raise ValueError("LightGBM model.num_class must be at least two")
        if objective in {"multiclass", "multiclassova", "multiclassova2"} and (
            num_class is None
        ):
            raise ValueError("multiclass LightGBM requires model.num_class")
        if objective == "binary" and num_class is not None and num_class != 2:
            raise ValueError("binary LightGBM requires model.num_class=2")
        training = value.get("training", {})
        if not isinstance(training, Mapping):
            raise ValueError("LightGBM training must be a mapping")
        num_rounds = training.get("num_rounds", 10)
        if (
            not isinstance(num_rounds, int)
            or isinstance(num_rounds, bool)
            or not 1 <= num_rounds <= 10000
        ):
            raise ValueError("LightGBM training.num_rounds must be in [1, 10000]")
        ray = value.get("ray")
        if (
            not isinstance(ray, Mapping)
            or not isinstance(ray.get("storage_path"), str)
            or not ray["storage_path"]
        ):
            raise ValueError("LightGBM ray.storage_path is required")
        output = value.get("output")
        if (
            not isinstance(output, Mapping)
            or not isinstance(output.get("bundle_uri"), str)
            or not output["bundle_uri"]
        ):
            raise ValueError("LightGBM output.bundle_uri is required")
        return dict(value)


class LightGBMInputValidator:
    """Require one labeled Ray Data tabular input."""

    api_version = 1
    schema_digest = "6" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        descriptors = value.get("descriptors")
        if not isinstance(bindings, list) or not isinstance(descriptors, Mapping):
            raise ValueError("LightGBM input requires bindings and descriptors")
        binding = next(
            (
                item
                for item in bindings
                if isinstance(item, Mapping) and item.get("name") == "train"
            ),
            None,
        )
        if not isinstance(binding, Mapping) or not binding.get("feature_names"):
            raise ValueError("LightGBM input requires train feature columns")
        if not isinstance(binding.get("label_name"), str) or not binding["label_name"]:
            raise ValueError("LightGBM input requires a label")
        return value


class LightGBMOutputValidator:
    """Require successful Bundle-backed LightGBM output."""

    api_version = 1
    schema_digest = "7" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("LightGBM output requires successful execution")
        if not outputs.get("bundle_uri"):
            raise ValueError("LightGBM output requires bundle_uri")
        return value


class LightGBMCoverageValidator:
    """Require complete framework-native distributed coverage."""

    api_version = 1
    schema_digest = "8" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("input_complete") is not True:
            raise ValueError("LightGBM input coverage is incomplete")
        if value.get("distributed") is not True:
            raise ValueError("LightGBM execution did not prove distribution")
        return value


__all__ = [
    "LightGBMCoverageValidator",
    "LightGBMConfigValidator",
    "LightGBMInputValidator",
    "LightGBMOutputValidator",
]
