"""Executable contracts for causal discovery."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class PCConfigValidator:
    api_version = 1
    schema_digest = "6" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"alpha", "max_condition_set", "output", "vote_threshold"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown PC discovery config keys: {unknown}")
        alpha = value.get("alpha", 0.05)
        threshold = value.get("vote_threshold", 0.5)
        if (
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not 0 < float(alpha) < 1
        ):
            raise ValueError("alpha must be in (0, 1)")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.5 <= float(threshold) <= 1
        ):
            raise ValueError("vote_threshold must be in [0.5, 1]")
        output = value.get("output")
        if not isinstance(output, Mapping) or not output.get("bundle_uri"):
            raise ValueError("output.bundle_uri is required")
        return dict(value)


class DiscoveryInputValidator:
    api_version = 1
    schema_digest = "7" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("causal discovery requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or len(binding.get("feature_names", ())) < 2
            or binding.get("label_name") is not None
        ):
            raise ValueError(
                "causal discovery requires at least two unlabeled variables"
            )
        return value


class DiscoveryOutputValidator:
    api_version = 1
    schema_digest = "8" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("causal discovery failed")
        if not outputs.get("bundle_uri") or not outputs.get("report_artifact_sha256"):
            raise ValueError("causal discovery requires report Bundle output")
        return value


class DiscoveryCoverageValidator:
    api_version = 1
    schema_digest = "9" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("causal discovery coverage is incomplete")
        workers = value.get("workers")
        if not isinstance(workers, list):
            raise ValueError("causal discovery requires Worker evidence")
        for worker in workers:
            rows = worker.get("input_rows") if isinstance(worker, Mapping) else None
            if not isinstance(rows, Mapping):
                raise ValueError("causal discovery row evidence is missing")
            if rows.get("coverage.discovery_rows") != rows.get("train"):
                raise ValueError("causal discovery shard coverage drifted")
        return value


__all__ = [
    "DiscoveryCoverageValidator",
    "DiscoveryInputValidator",
    "DiscoveryOutputValidator",
    "PCConfigValidator",
]
