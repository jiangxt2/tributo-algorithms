"""Executable contracts for distributed causal estimation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ATEConfigValidator:
    api_version = 1
    schema_digest = "6" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {
            "confidence_z",
            "cross_fit_folds",
            "fold_column",
            "output",
            "policy_cost",
            "treatment_col",
        }
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown ATE config keys: {unknown}")
        treatment = value.get("treatment_col")
        if not isinstance(treatment, str) or not treatment:
            raise ValueError("treatment_col is required")
        output = value.get("output")
        if not isinstance(output, Mapping) or not output.get("bundle_uri"):
            raise ValueError("output.bundle_uri is required")
        folds = value.get("cross_fit_folds", 5)
        if (
            not isinstance(folds, int)
            or isinstance(folds, bool)
            or not 2 <= folds <= 20
        ):
            raise ValueError("cross_fit_folds must be between 2 and 20")
        return dict(value)


class IVConfigValidator(ATEConfigValidator):
    api_version = 1
    schema_digest = "5" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {
            "confidence_z",
            "instrument_col",
            "output",
            "policy_cost",
            "treatment_col",
        }
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown IV config keys: {unknown}")
        instrument = value.get("instrument_col")
        treatment = value.get("treatment_col")
        if not isinstance(instrument, str) or not instrument:
            raise ValueError("instrument_col is required")
        if not isinstance(treatment, str) or not treatment or instrument == treatment:
            raise ValueError("treatment_col must differ from instrument_col")
        output = value.get("output")
        if not isinstance(output, Mapping) or not output.get("bundle_uri"):
            raise ValueError("output.bundle_uri is required")
        return dict(value)


class CausalInputValidator:
    api_version = 1
    schema_digest = "2" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("causal estimation requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or not binding.get("feature_names")
            or not binding.get("label_name")
        ):
            raise ValueError("causal input requires treatment/features and outcome")
        return value


class CausalOutputValidator:
    api_version = 1
    schema_digest = "3" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("causal estimation failed")
        if not outputs.get("bundle_uri") or not outputs.get("report_artifact_sha256"):
            raise ValueError("causal output requires Bundle and report artifact")
        return value


class TreatmentCoverageValidator:
    api_version = 1
    schema_digest = "4" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("causal distributed coverage is incomplete")
        workers = value.get("workers")
        if not isinstance(workers, list):
            raise ValueError("causal coverage requires Worker evidence")
        totals = {"treated": 0, "control": 0}
        for worker in workers:
            rows = worker.get("input_rows") if isinstance(worker, Mapping) else None
            if not isinstance(rows, Mapping):
                raise ValueError("causal treatment coverage is missing")
            train_rows = rows.get("train")
            local_total = 0
            for group in totals:
                count = rows.get(f"coverage.{group}")
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise ValueError(f"causal {group} coverage is malformed")
                totals[group] += count
                local_total += count
            if local_total != train_rows:
                raise ValueError("treatment groups do not partition train rows")
        if any(count <= 0 for count in totals.values()):
            raise ValueError("causal estimation requires treatment overlap")
        return value


__all__ = [
    "ATEConfigValidator",
    "CausalInputValidator",
    "CausalOutputValidator",
    "IVConfigValidator",
    "TreatmentCoverageValidator",
]
