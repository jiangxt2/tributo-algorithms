"""Versioned contracts for official tabular Torch algorithms."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any


def _digest(name: str) -> str:
    return hashlib.sha256(f"tributo.tabular-torch.{name}.v2".encode()).hexdigest()


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
        if not isinstance(output, Mapping) or not isinstance(
            output.get("bundle_uri"), str
        ):
            raise ValueError("output.bundle_uri is required")
        optimizer = normalized.get("optimizer", {})
        if isinstance(optimizer, Mapping):
            learning_rate = optimizer.get("learning_rate", 0.001)
            accumulation = optimizer.get("accumulation_steps", 1)
            if (
                not isinstance(learning_rate, (int, float))
                or isinstance(learning_rate, bool)
                or not math.isfinite(float(learning_rate))
                or float(learning_rate) <= 0
            ):
                raise ValueError("optimizer.learning_rate must be positive and finite")
            if (
                not isinstance(accumulation, int)
                or isinstance(accumulation, bool)
                or accumulation < 1
            ):
                raise ValueError("optimizer.accumulation_steps must be positive")
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
            or not math.isfinite(float(prior))
            or not 0 < float(prior) < 1
        ):
            raise ValueError("PU loss.class_prior must be in (0, 1)")
        if loss.get("type", "nnpu") not in {"nnpu", "upu"}:
            raise ValueError("PU loss.type must be nnpu or upu")
        beta = loss.get("beta", 0.0)
        gamma = loss.get("gamma", 1.0)
        if (
            not isinstance(beta, (int, float))
            or isinstance(beta, bool)
            or not math.isfinite(float(beta))
            or float(beta) < 0
        ):
            raise ValueError("PU loss.beta must be finite and non-negative")
        if (
            not isinstance(gamma, (int, float))
            or isinstance(gamma, bool)
            or not math.isfinite(float(gamma))
            or not 0 <= float(gamma) <= 1
        ):
            raise ValueError("PU loss.gamma must be in [0, 1]")
        training = normalized.get("training", {})
        if isinstance(training, Mapping) and training.get("accumulation_steps", 1) != 1:
            raise ValueError("PU requires gradient_accumulation_steps=1")
        optimizer = normalized.get("optimizer", {})
        if (
            isinstance(optimizer, Mapping)
            and optimizer.get("accumulation_steps", 1) != 1
        ):
            raise ValueError("PU requires gradient_accumulation_steps=1")
        return normalized


class DNNTensorInputValidator:
    api_version = 1
    schema_digest = _digest("dnn-named-input")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("tabular-torch requires one train binding")
        binding = bindings[0]
        if not isinstance(binding, Mapping) or binding.get("role", "train") != "train":
            raise ValueError("tabular-torch requires a train binding")
        features = binding.get("feature_names")
        if (
            not isinstance(features, list)
            or not features
            or any(not isinstance(name, str) or not name for name in features)
        ):
            raise ValueError("tabular-torch requires named dense features")
        if not isinstance(binding.get("label_name"), str) or not binding["label_name"]:
            raise ValueError("tabular-torch requires one label")
        if binding.get("sample_weight_name") is not None:
            raise ValueError("tabular-torch does not support sample-weight binding")
        return value


class PUTensorInputValidator(DNNTensorInputValidator):
    api_version = 1
    schema_digest = _digest("pu-named-input")


class DNNTorchBundleOutputValidator:
    api_version = 1
    schema_digest = _digest("dnn-onnx-output")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("tabular-torch execution failed")
        if not isinstance(outputs.get("bundle_uri"), str) or not outputs["bundle_uri"]:
            raise ValueError("tabular-torch output requires Bundle")
        return value


class DNNTorchCoverageValidator:
    api_version = 1
    schema_digest = _digest("dnn-coverage")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("tabular-torch distributed coverage is incomplete")
        return value


class PUTorchBundleOutputValidator(DNNTorchBundleOutputValidator):
    api_version = 1
    schema_digest = _digest("pu-onnx-output")


class PUCoverageValidator(DNNTorchCoverageValidator):
    api_version = 1
    schema_digest = _digest("pu-coverage")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = super().validate(value)
        workers = validated.get("workers")
        if not isinstance(workers, list) or not workers:
            raise ValueError("PU coverage requires worker evidence")
        totals = {"positive": 0, "unlabeled": 0}
        for worker in workers:
            if not isinstance(worker, Mapping) or not isinstance(
                worker.get("input_rows"), Mapping
            ):
                raise ValueError("PU worker group coverage is missing")
            rows = worker["input_rows"]
            train_rows = rows.get("train")
            group_total = 0
            for group in totals:
                count = rows.get(f"coverage.{group}")
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(f"PU {group} coverage is malformed")
                totals[group] += count
                group_total += count
            if (
                not isinstance(train_rows, int)
                or isinstance(train_rows, bool)
                or group_total != train_rows
            ):
                raise ValueError("PU group coverage does not partition train rows")
        if any(count <= 0 for count in totals.values()):
            raise ValueError("PU execution did not cover both label groups")
        return validated


__all__ = [
    "DNNTensorInputValidator",
    "DNNTorchBundleOutputValidator",
    "DNNTorchCoverageValidator",
    "PUTensorInputValidator",
    "PUTorchBundleOutputValidator",
    "PUConfigValidator",
    "PUCoverageValidator",
    "TabularTorchConfigValidator",
]
