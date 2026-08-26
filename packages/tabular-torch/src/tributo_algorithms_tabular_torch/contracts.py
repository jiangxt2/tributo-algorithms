"""Executable contracts for official tabular PyTorch algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TabularTorchConfigValidator:
    api_version = 1
    schema_digest = "5" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"loss", "metrics", "model", "optimizer", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown tabular-torch config keys: {unknown}")
        normalized = dict(value)
        for name in allowed:
            item = normalized.get(name)
            if item is not None and not isinstance(item, Mapping):
                raise ValueError(f"{name} must be a mapping")
        output = normalized.get("output")
        if not isinstance(output, Mapping) or not output.get("bundle_uri"):
            raise ValueError("output.bundle_uri is required")
        return normalized


class PUConfigValidator(TabularTorchConfigValidator):
    api_version = 1
    schema_digest = "6" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized = dict(super().validate(value))
        loss = normalized.get("loss")
        if not isinstance(loss, Mapping) or "class_prior" not in loss:
            raise ValueError("PU loss.class_prior is required")
        prior = loss["class_prior"]
        if (
            not isinstance(prior, (int, float))
            or isinstance(prior, bool)
            or not 0 < float(prior) < 1
        ):
            raise ValueError("PU loss.class_prior must be in (0, 1)")
        if loss.get("type", "nnpu") not in {"nnpu", "upu"}:
            raise ValueError("PU loss.type must be nnpu or upu")
        return normalized


class LabeledDenseInputValidator:
    api_version = 1
    schema_digest = "7" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("tabular-torch requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or not binding.get("feature_names")
            or not binding.get("label_name")
        ):
            raise ValueError("tabular-torch requires dense features and one label")
        return value


class TabularTorchOutputValidator:
    api_version = 1
    schema_digest = "8" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("tabular-torch execution failed")
        if not outputs.get("bundle_uri"):
            raise ValueError("tabular-torch output requires Bundle")
        return value


class TabularTorchCoverageValidator:
    api_version = 1
    schema_digest = "9" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("tabular-torch distributed coverage is incomplete")
        return value


class PUCoverageValidator(TabularTorchCoverageValidator):
    api_version = 1
    schema_digest = "a" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = super().validate(value)
        workers = validated.get("workers")
        if not isinstance(workers, list):
            raise ValueError("PU coverage requires worker evidence")
        totals = {"positive": 0, "unlabeled": 0}
        for worker in workers:
            if not isinstance(worker, Mapping):
                raise ValueError("PU worker evidence is malformed")
            rows = worker.get("input_rows")
            if not isinstance(rows, Mapping):
                raise ValueError("PU worker group coverage is missing")
            train_rows = rows.get("train")
            worker_group_total = 0
            for group in totals:
                count = rows.get(f"coverage.{group}")
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(f"PU {group} coverage is malformed")
                totals[group] += count
                worker_group_total += count
            if (
                not isinstance(train_rows, int)
                or isinstance(train_rows, bool)
                or worker_group_total != train_rows
            ):
                raise ValueError("PU group coverage does not partition train rows")
        if any(count <= 0 for count in totals.values()):
            raise ValueError("PU execution did not cover both label groups")
        return validated


__all__ = [
    "LabeledDenseInputValidator",
    "PUConfigValidator",
    "PUCoverageValidator",
    "TabularTorchConfigValidator",
    "TabularTorchCoverageValidator",
    "TabularTorchOutputValidator",
]
