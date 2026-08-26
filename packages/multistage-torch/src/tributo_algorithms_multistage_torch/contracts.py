"""Executable contracts for finite Teacher-to-Student distillation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DistillationConfigValidator:
    api_version = 1
    schema_digest = "a" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"model", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown distillation config keys: {unknown}")
        model = value.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("distillation model config is required")
        for name in ("input_features", "teacher_hidden", "student_hidden"):
            item = model.get(name)
            if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                raise ValueError(f"model.{name} must be positive")
        output = value.get("output")
        if not isinstance(output, Mapping) or not output.get("bundle_uri"):
            raise ValueError("output.bundle_uri is required")
        ray = value.get("ray")
        if not isinstance(ray, Mapping) or not ray.get("storage_path"):
            raise ValueError("ray.storage_path is required")
        return dict(value)


class DistillationInputValidator:
    api_version = 1
    schema_digest = "b" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("distillation requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or not binding.get("feature_names")
            or not binding.get("label_name")
        ):
            raise ValueError("distillation requires dense features and label")
        return value


class DistillationOutputValidator:
    api_version = 1
    schema_digest = "c" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("distillation failed")
        if not outputs.get("bundle_uri"):
            raise ValueError("distillation requires Student Bundle")
        return value


class DistillationCoverageValidator:
    api_version = 1
    schema_digest = "d" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("distillation coverage is incomplete")
        state = value.get("state")
        details = state.get("details") if isinstance(state, Mapping) else None
        if not isinstance(details, Mapping):
            raise ValueError("distillation stage evidence is missing")
        if details.get("component_stages") != "teacher,student":
            raise ValueError("distillation component stage set drifted")
        for stage in ("teacher", "student"):
            if int(details.get(f"stage.{stage}.rows", 0)) <= 0:
                raise ValueError(f"distillation {stage} coverage is missing")
        return value


class PretrainFinetuneConfigValidator:
    api_version = 1
    schema_digest = "e" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"model", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown pretrain-finetune config keys: {unknown}")
        for name in allowed:
            if not isinstance(value.get(name), Mapping):
                raise ValueError(f"{name} config is required")
        model = value["model"]
        for name in ("input_features", "hidden_features"):
            item = model.get(name)
            if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                raise ValueError(f"model.{name} must be positive")
        training = value["training"]
        for name in ("pretrain_epochs", "finetune_epochs"):
            item = training.get(name, 1)
            if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                raise ValueError(f"training.{name} must be positive")
        if not value["ray"].get("storage_path"):
            raise ValueError("ray.storage_path is required")
        if not value["output"].get("bundle_uri"):
            raise ValueError("output.bundle_uri is required")
        return dict(value)


class PretrainFinetuneInputValidator:
    api_version = 1
    schema_digest = "f" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("pretrain-finetune requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or not binding.get("feature_names")
            or not binding.get("label_name")
        ):
            raise ValueError("pretrain-finetune requires dense features and label")
        return value


class PretrainFinetuneOutputValidator:
    api_version = 1
    schema_digest = "0" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("pretrain-finetune failed")
        if not outputs.get("bundle_uri") or not outputs.get("composition_digest"):
            raise ValueError("pretrain-finetune requires Bundle and composition digest")
        return value


class PretrainFinetuneCoverageValidator:
    api_version = 1
    schema_digest = "1" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("pretrain-finetune coverage is incomplete")
        state = value.get("state")
        details = state.get("details") if isinstance(state, Mapping) else None
        if not isinstance(details, Mapping):
            raise ValueError("pretrain-finetune stage evidence is missing")
        if details.get("component_stages") != "pretrain,finetune":
            raise ValueError("pretrain-finetune component stage set drifted")
        for stage in ("pretrain", "finetune"):
            if int(details.get(f"stage.{stage}.rows", 0)) <= 0:
                raise ValueError(f"pretrain-finetune {stage} coverage is missing")
        return value


__all__ = [
    "DistillationConfigValidator",
    "DistillationCoverageValidator",
    "DistillationInputValidator",
    "DistillationOutputValidator",
    "PretrainFinetuneConfigValidator",
    "PretrainFinetuneCoverageValidator",
    "PretrainFinetuneInputValidator",
    "PretrainFinetuneOutputValidator",
]
