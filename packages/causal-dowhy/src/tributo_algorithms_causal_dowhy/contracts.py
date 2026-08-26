"""Executable contracts for distributed DoWhy adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DoWhyConfigValidator:
    api_version = 1
    schema_digest = "e" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"data", "output", "refutation", "runtime"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown DoWhy config keys: {unknown}")
        for name in allowed:
            if not isinstance(value.get(name), Mapping):
                raise ValueError(f"{name} config is required")
        data = value["data"]
        for name in ("common_causes", "treatment_col", "outcome_col"):
            if not data.get(name):
                raise ValueError(f"data.{name} is required")
        if not value["output"].get("bundle_uri"):
            raise ValueError("output.bundle_uri is required")
        return dict(value)


class DoWhyInputValidator:
    api_version = 1
    schema_digest = "f" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 1:
            raise ValueError("DoWhy requires one train binding")
        binding = bindings[0]
        if (
            not isinstance(binding, Mapping)
            or len(binding.get("feature_names", ())) < 2
            or not binding.get("label_name")
        ):
            raise ValueError("DoWhy input requires confounders, treatment, and outcome")
        return value


class DoWhyOutputValidator:
    api_version = 1
    schema_digest = "0" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("DoWhy execution failed")
        if not outputs.get("bundle_uri") or not outputs.get("composition_digest"):
            raise ValueError("DoWhy output requires Bundle and composition digest")
        return value


class DoWhyCoverageValidator:
    api_version = 1
    schema_digest = "1" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("DoWhy coverage is incomplete")
        state = value.get("state")
        details = state.get("details") if isinstance(state, Mapping) else None
        if (
            not isinstance(details, Mapping)
            or details.get("component_stages") != "estimate,refute"
        ):
            raise ValueError("DoWhy estimate/refute stage evidence is missing")
        return value


class GCMConfigValidator:
    api_version = 1
    schema_digest = "2" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"data", "gcm", "output", "runtime"}
        if unknown := sorted(set(value) - allowed):
            raise ValueError(f"unknown GCM config keys: {unknown}")
        for name in allowed:
            if not isinstance(value.get(name), Mapping):
                raise ValueError(f"{name} config is required")
        data = value["data"]
        nodes = data.get("nodes")
        edges = data.get("edges")
        target = data.get("target_node")
        if (
            not isinstance(nodes, list)
            or len(nodes) < 2
            or len(set(nodes)) != len(nodes)
            or any(not isinstance(name, str) or not name for name in nodes)
        ):
            raise ValueError("data.nodes requires unique node names")
        if not isinstance(edges, list) or not edges:
            raise ValueError("data.edges requires a non-empty edge list")
        for edge in edges:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or any(name not in nodes for name in edge)
                or edge[0] == edge[1]
            ):
                raise ValueError("data.edges must reference two distinct nodes")
        if target not in nodes:
            raise ValueError("data.target_node must reference a declared node")
        interventions = data.get("interventions", {})
        if not isinstance(interventions, Mapping) or any(
            name not in nodes
            or not isinstance(number, (int, float))
            or isinstance(number, bool)
            for name, number in interventions.items()
        ):
            raise ValueError("data.interventions must map declared nodes to numbers")
        gcm = value["gcm"]
        if gcm.get("quality", "good") not in {"good", "better", "best"}:
            raise ValueError("gcm.quality must be good, better, or best")
        for name in ("distribution_samples", "shapley_permutations"):
            item = gcm.get(name, 1)
            if not isinstance(item, int) or isinstance(item, bool) or item < 1:
                raise ValueError(f"gcm.{name} must be positive")
        if not value["output"].get("bundle_uri"):
            raise ValueError("output.bundle_uri is required")
        return dict(value)


class GCMInputValidator:
    api_version = 1
    schema_digest = "3" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or len(bindings) != 2:
            raise ValueError("GCM requires train and anomaly bindings")
        if value.get("primary_role") != "train":
            raise ValueError("GCM train binding must be primary")
        roles = {
            binding.get("name") for binding in bindings if isinstance(binding, Mapping)
        }
        if roles != {"train", "anomaly"}:
            raise ValueError("GCM input roles must be train and anomaly")
        for binding in bindings:
            if (
                not isinstance(binding, Mapping)
                or len(binding.get("feature_names", ())) < 2
                or binding.get("label_name") is not None
            ):
                raise ValueError("GCM roles require unlabeled causal variables")
        return value


class GCMOutputValidator:
    api_version = 1
    schema_digest = "4" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        outputs = value.get("outputs")
        if value.get("status") != "succeeded" or not isinstance(outputs, Mapping):
            raise ValueError("GCM execution failed")
        if not outputs.get("bundle_uri") or not outputs.get("composition_digest"):
            raise ValueError("GCM output requires Bundle and composition digest")
        return value


class GCMCoverageValidator:
    api_version = 1
    schema_digest = "5" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("GCM coverage is incomplete")
        workers = value.get("workers")
        if not isinstance(workers, list) or any(
            not isinstance(worker, Mapping)
            or not isinstance(worker.get("input_rows"), Mapping)
            or int(worker["input_rows"].get("coverage.anomaly", 0)) < 1
            for worker in workers
        ):
            raise ValueError("GCM anomaly coverage is missing")
        state = value.get("state")
        details = state.get("details") if isinstance(state, Mapping) else None
        if (
            not isinstance(details, Mapping)
            or details.get("component_stages") != "fit_gcm,attribute_root_cause"
        ):
            raise ValueError("GCM fit and root-cause stage evidence is missing")
        return value


__all__ = [
    "DoWhyConfigValidator",
    "DoWhyCoverageValidator",
    "DoWhyInputValidator",
    "DoWhyOutputValidator",
    "GCMConfigValidator",
    "GCMCoverageValidator",
    "GCMInputValidator",
    "GCMOutputValidator",
]
