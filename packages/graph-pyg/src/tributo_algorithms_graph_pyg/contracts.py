"""Executable contracts for the official homogeneous PyG graph Wheel."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def _digest(name: str) -> str:
    return hashlib.sha256(f"tributo.graph-pyg.{name}.v2".encode()).hexdigest()


class GraphConfigValidator:
    """Validate bounded full-neighborhood GraphSAGE configuration."""

    api_version = 1
    schema_digest = "1" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"model", "output", "ray", "training"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown graph config keys: {unknown}")
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
        training = normalized.get("training", {})
        if not isinstance(training, Mapping):
            raise ValueError("training must be a mapping")
        epochs = training.get("epochs", 2)
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError("training.epochs must be positive")
        return normalized


class RelationalGraphConfigValidator(GraphConfigValidator):
    api_version = 1
    schema_digest = "5" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized = dict(super().validate(value))
        model = normalized.get("model")
        relations = model.get("num_relations") if isinstance(model, Mapping) else None
        if (
            not isinstance(relations, int)
            or isinstance(relations, bool)
            or relations < 1
        ):
            raise ValueError("R-GCN model.num_relations must be positive")
        return normalized


class HomogeneousGraphInputValidator:
    """Require separate node, edge, and labeled seed roles."""

    api_version = 1
    schema_digest = "2" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        descriptors = value.get("descriptors")
        if not isinstance(bindings, list) or not isinstance(descriptors, Mapping):
            raise ValueError("graph input requires bindings and descriptors")
        if len(bindings) != 3 or any(
            not isinstance(item, Mapping) for item in bindings
        ):
            raise ValueError("graph input bindings must contain three mapping roles")
        names = [item.get("name") for item in bindings]
        if any(not isinstance(name, str) or not name for name in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("graph input binding roles must be unique names")
        by_role = {item["name"]: item for item in bindings}
        if set(by_role) != {"edges", "nodes", "train"}:
            raise ValueError(
                "graph input requires exactly nodes, edges, and train roles"
            )
        if value.get("primary_role") != "train":
            raise ValueError("graph train seeds must be the primary role")
        if set(descriptors) != set(by_role):
            raise ValueError("graph input descriptors do not match role bindings")
        if not by_role["train"].get("label_name"):
            raise ValueError("graph train seeds require labels")
        if len(by_role["nodes"].get("feature_names", ())) < 2:
            raise ValueError("node input requires node_id and at least one feature")
        if len(by_role["edges"].get("feature_names", ())) != 2:
            raise ValueError("edge input requires source and destination columns")
        return value


class RelationalGraphInputValidator(HomogeneousGraphInputValidator):
    api_version = 1
    schema_digest = "6" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        bindings = value.get("bindings")
        descriptors = value.get("descriptors")
        if not isinstance(bindings, list) or not isinstance(descriptors, Mapping):
            raise ValueError("relational graph input requires bindings and descriptors")
        if len(bindings) != 3 or any(
            not isinstance(item, Mapping) for item in bindings
        ):
            raise ValueError(
                "relational graph input bindings must contain three mapping roles"
            )
        names = [item.get("name") for item in bindings]
        if any(not isinstance(name, str) or not name for name in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("relational graph binding roles must be unique names")
        by_role = {item["name"]: item for item in bindings}
        if set(by_role) != {"edges", "nodes", "train"}:
            raise ValueError("R-GCN requires nodes, edges, and train roles")
        if value.get("primary_role") != "train" or set(descriptors) != set(by_role):
            raise ValueError("R-GCN role descriptors are inconsistent")
        if not by_role["train"].get("label_name"):
            raise ValueError("R-GCN train seeds require labels")
        if len(by_role["nodes"].get("feature_names", ())) < 2:
            raise ValueError("R-GCN nodes require node_id and features")
        if len(by_role["edges"].get("feature_names", ())) != 3:
            raise ValueError("R-GCN edges require source, destination, and relation")
        return value


class GraphOutputValidator:
    """Require a successfully published graph model Bundle."""

    api_version = 1
    schema_digest = "3" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("status") != "succeeded":
            raise ValueError("graph training did not succeed")
        outputs = value.get("outputs")
        if (
            not isinstance(outputs, Mapping)
            or not isinstance(outputs.get("bundle_uri"), str)
            or not outputs["bundle_uri"]
        ):
            raise ValueError("graph output requires bundle_uri")
        return value


class GraphCoverageValidator:
    """Require cross-worker seed coverage and synchronized topology state."""

    api_version = 1
    schema_digest = "4" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            value.get("input_complete") is not True
            or value.get("distributed") is not True
        ):
            raise ValueError("graph seed coverage is incomplete")
        workers = value.get("workers")
        state = value.get("state")
        if not isinstance(workers, list) or not isinstance(state, Mapping):
            raise ValueError("graph execution evidence is missing")
        if any(
            not isinstance(worker, Mapping)
            or not isinstance(worker.get("input_rows"), Mapping)
            or int(worker["input_rows"].get("train", 0)) <= 0
            or int(worker["input_rows"].get("nodes", 0)) <= 0
            or int(worker["input_rows"].get("edges", 0)) <= 0
            for worker in workers
        ):
            raise ValueError("graph role coverage is incomplete")
        details = state.get("details")
        if (
            not isinstance(details, Mapping)
            or details.get("sampling") != "full_neighborhood"
        ):
            raise ValueError("graph sampling evidence is missing")
        return value


class RelationalGraphCoverageValidator(GraphCoverageValidator):
    api_version = 1
    schema_digest = "7" * 64

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = super().validate(value)
        state = validated.get("state")
        details = state.get("details") if isinstance(state, Mapping) else None
        if (
            not isinstance(details, Mapping)
            or details.get("topology_kind") != "relational"
        ):
            raise ValueError("R-GCN relational topology evidence is missing")
        return validated


class GraphSAGETorchInputValidator(HomogeneousGraphInputValidator):
    api_version = 1
    schema_digest = _digest("graphsage-input")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = super().validate(value)
        bindings = validated.get("bindings")
        if not isinstance(bindings, list):
            raise ValueError("graph input bindings are malformed")
        for binding in bindings:
            if (
                isinstance(binding, Mapping)
                and binding.get("sample_weight_name") is not None
            ):
                raise ValueError("GraphSAGE does not support sample-weight binding")
        return validated


class RGCNTorchInputValidator(RelationalGraphInputValidator):
    api_version = 1
    schema_digest = _digest("rgcn-input")

    def validate(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = super().validate(value)
        bindings = validated.get("bindings")
        if not isinstance(bindings, list):
            raise ValueError("R-GCN input bindings are malformed")
        for binding in bindings:
            if (
                isinstance(binding, Mapping)
                and binding.get("sample_weight_name") is not None
            ):
                raise ValueError("R-GCN does not support sample-weight binding")
        return validated


class GraphSAGETorchCoverageValidator(GraphCoverageValidator):
    api_version = 1
    schema_digest = _digest("graphsage-coverage")


class RGCNTorchCoverageValidator(RelationalGraphCoverageValidator):
    api_version = 1
    schema_digest = _digest("rgcn-coverage")


__all__ = [
    "GraphConfigValidator",
    "GraphSAGETorchCoverageValidator",
    "GraphSAGETorchInputValidator",
    "GraphCoverageValidator",
    "GraphOutputValidator",
    "HomogeneousGraphInputValidator",
    "RelationalGraphConfigValidator",
    "RelationalGraphCoverageValidator",
    "RelationalGraphInputValidator",
    "RGCNTorchCoverageValidator",
    "RGCNTorchInputValidator",
]
