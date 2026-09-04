"""Executable contracts for the official Two-Tower Wheel."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def _digest(name: str) -> str:
    return hashlib.sha256(f"tributo.recsys-torch.{name}.v2".encode()).hexdigest()


class TwoTowerConfigValidator:
    api_version = 1
    schema_digest = "b" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"loss", "metrics", "model", "optimizer", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown Two-Tower config keys: {unknown}")
        normalized = dict(value)
        model = normalized.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("Two-Tower model config is required")
        for name in ("user_count", "item_count"):
            count = model.get(name)
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        output = normalized.get("output")
        if (
            not isinstance(output, Mapping)
            or not isinstance(output.get("bundle_uri"), str)
            or not output["bundle_uri"]
        ):
            raise ValueError("output.bundle_uri is required")
        return normalized


class PairInputValidator:
    api_version = 1
    schema_digest = "c" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or not 1 <= len(bindings) <= 3:
            raise ValueError("Two-Tower requires train and optional val/test bindings")
        if any(not isinstance(binding, Mapping) for binding in bindings):
            raise ValueError("Two-Tower bindings must be mappings")
        names = [binding.get("name") for binding in bindings]
        if any(not isinstance(name, str) for name in names):
            raise ValueError("Two-Tower input roles must be named")
        by_role = {str(binding["name"]): binding for binding in bindings}
        if (
            len(by_role) != len(bindings)
            or "train" not in by_role
            or not set(by_role).issubset({"train", "val", "test"})
            or value.get("primary_role") != "train"
        ):
            raise ValueError("Two-Tower input roles are invalid")
        train = by_role["train"]
        expected_features = train.get("feature_names")
        expected_label = train.get("label_name")
        for binding in by_role.values():
            if (
                binding.get("feature_names") != expected_features
                or not isinstance(expected_features, list)
                or len(expected_features) != 2
                or binding.get("label_name") != expected_label
                or not isinstance(expected_label, str)
                or binding.get("sample_weight_name") is not None
            ):
                raise ValueError(
                    "Two-Tower requires consistent user ID, item ID, and label columns"
                )
        return value


class TwoTowerOutputValidator:
    api_version = 1
    schema_digest = "d" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("Two-Tower execution failed")
        if not isinstance(outputs.get("bundle_uri"), str) or not outputs["bundle_uri"]:
            raise ValueError("Two-Tower output requires Bundle")
        return value


class PairCoverageValidator:
    api_version = 1
    schema_digest = "e" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("Two-Tower distributed coverage is incomplete")
        workers = value.get("workers")
        if not isinstance(workers, list):
            raise ValueError("Two-Tower coverage requires Worker evidence")
        totals = {"positive_pairs": 0, "negative_pairs": 0}
        for worker in workers:
            rows = worker.get("input_rows") if isinstance(worker, Mapping) else None
            if not isinstance(rows, Mapping):
                raise ValueError("Two-Tower pair coverage is missing")
            train_rows = rows.get("train")
            local_total = 0
            for name in totals:
                count = rows.get(f"coverage.{name}")
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(f"Two-Tower {name} coverage is malformed")
                totals[name] += count
                local_total += count
            if local_total != train_rows:
                raise ValueError("Two-Tower pair groups do not partition train rows")
        if any(count <= 0 for count in totals.values()):
            raise ValueError("Two-Tower requires positive and negative interactions")
        return value


class JaggedConfigValidator:
    api_version = 1
    schema_digest = "9" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"data", "model", "output", "ray", "training"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown jagged recommendation config keys: {unknown}")
        for name in allowed:
            if not isinstance(value.get(name), Mapping):
                raise ValueError(f"{name} config is required")
        model = value["model"]
        for name in ("user_count", "item_count", "embedding_dim"):
            item = model.get(name)
            if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                raise ValueError(f"model.{name} must be positive")
        data = value["data"]
        columns = [
            data.get(name)
            for name in ("user_col", "history_col", "candidate_col", "label_col")
        ]
        if any(not isinstance(name, str) or not name for name in columns):
            raise ValueError("jagged data column names are required")
        if len(set(columns)) != len(columns):
            raise ValueError("jagged data columns must be distinct")
        history_width = data.get("inference_history_width")
        if (
            not isinstance(history_width, int)
            or isinstance(history_width, bool)
            or history_width < 1
        ):
            raise ValueError("data.inference_history_width must be positive")
        if not value["ray"].get("storage_path"):
            raise ValueError("ray.storage_path is required")
        if (
            not isinstance(value["output"].get("bundle_uri"), str)
            or not value["output"]["bundle_uri"]
        ):
            raise ValueError("output.bundle_uri is required")
        return dict(value)


class JaggedInputValidator:
    api_version = 1
    schema_digest = "6" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("jagged recommendation requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or len(binding.get("feature_names", ())) != 3
            or not binding.get("label_name")
        ):
            raise ValueError(
                "jagged recommendation requires user, history, candidate, and label"
            )
        return value


class JaggedOutputValidator:
    api_version = 1
    schema_digest = "7" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("jagged recommendation failed")
        if not isinstance(outputs.get("bundle_uri"), str) or not outputs["bundle_uri"]:
            raise ValueError("jagged recommendation requires model Bundle")
        return value


class JaggedCoverageValidator:
    api_version = 1
    schema_digest = "8" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("jagged recommendation coverage is incomplete")
        workers = value.get("workers")
        if not isinstance(workers, list):
            raise ValueError("jagged recommendation requires Worker evidence")
        history_tokens = 0
        routed_tokens = 0
        pair_groups = {"positive_pairs": 0, "negative_pairs": 0}
        for worker in workers:
            rows = worker.get("input_rows") if isinstance(worker, Mapping) else None
            if not isinstance(rows, Mapping):
                raise ValueError("jagged Worker coverage is missing")
            history_tokens += int(rows.get("coverage.history_tokens", 0))
            routed_tokens += int(rows.get("coverage.routed_owned_tokens", 0))
            local_pairs = 0
            for name in pair_groups:
                count = int(rows.get(f"coverage.{name}", 0))
                pair_groups[name] += count
                local_pairs += count
            if local_pairs != rows.get("train"):
                raise ValueError("jagged pair groups do not partition train rows")
        if history_tokens < 1 or history_tokens != routed_tokens:
            raise ValueError("jagged All-to-All routing does not conserve sparse keys")
        if any(count < 1 for count in pair_groups.values()):
            raise ValueError("jagged training requires positive and negative pairs")
        state = value.get("state")
        details = state.get("details") if isinstance(state, Mapping) else None
        if (
            not isinstance(details, Mapping)
            or details.get("jagged") is not True
            or details.get("routing") != "all_to_all_single_owner_mod"
        ):
            raise ValueError("jagged All-to-All routing evidence is missing")
        return value


class JaggedTorchInputValidator(JaggedInputValidator):
    api_version = 1
    schema_digest = _digest("jagged-input-role-named")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = super().validate(value)
        bindings = validated.get("bindings")
        binding = bindings[0] if isinstance(bindings, list) else None
        if (
            not isinstance(binding, Mapping)
            or binding.get("name") != "train"
            or binding.get("sample_weight_name") is not None
            or not all(
                isinstance(name, str) and name
                for name in binding.get("feature_names", ())
            )
        ):
            raise ValueError(
                "jagged recommendation requires named, unweighted train input"
            )
        return validated


class JaggedTorchCoverageValidator(JaggedCoverageValidator):
    api_version = 1
    schema_digest = _digest("jagged-coverage")


class TwoTowerTensorInputValidator(PairInputValidator):
    api_version = 1
    schema_digest = _digest("two-tower-input-role-routed")


class TwoTowerTorchCoverageValidator(PairCoverageValidator):
    api_version = 1
    schema_digest = _digest("two-tower-coverage")


__all__ = [
    "PairCoverageValidator",
    "PairInputValidator",
    "TwoTowerTensorInputValidator",
    "TwoTowerTorchCoverageValidator",
    "TwoTowerConfigValidator",
    "TwoTowerOutputValidator",
    "JaggedConfigValidator",
    "JaggedCoverageValidator",
    "JaggedInputValidator",
    "JaggedOutputValidator",
    "JaggedTorchCoverageValidator",
    "JaggedTorchInputValidator",
]
