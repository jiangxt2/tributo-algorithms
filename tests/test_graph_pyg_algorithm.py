"""Tests for the official homogeneous PyG GraphSAGE algorithm."""

from __future__ import annotations

import torch
from tributo.algorithms.api import DistributionStrategy
from tributo.algorithms.spi import FrameworkNativeAlgorithm
from tributo_algorithms_graph_pyg import (
    GRAPHSAGE_DESCRIPTOR,
    RGCN_DESCRIPTOR,
    DistributedGraphSAGE,
    DistributedRGCN,
)
from tributo_algorithms_graph_pyg.algorithm import _build_model
from tributo_algorithms_graph_pyg.contracts import (
    GraphCoverageValidator,
    HomogeneousGraphInputValidator,
    RelationalGraphCoverageValidator,
    RelationalGraphInputValidator,
)


def test_graphsage_descriptor_uses_framework_native_contract() -> None:
    registration = GRAPHSAGE_DESCRIPTOR.registration

    assert GRAPHSAGE_DESCRIPTOR.api_version == 2
    assert issubclass(DistributedGraphSAGE, FrameworkNativeAlgorithm)
    assert registration.contract_bindings is not None
    assert registration.distribution_spec is not None
    assert (
        registration.distribution_spec.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    )
    assert registration.implementation.runtime_id == "tributo.framework_native"


def test_rgcn_descriptor_uses_framework_native_contract() -> None:
    registration = RGCN_DESCRIPTOR.registration
    assert issubclass(DistributedRGCN, DistributedGraphSAGE)
    assert registration.contract_bindings is not None
    assert registration.distribution_spec is not None
    assert (
        registration.distribution_spec.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    )


def test_graph_input_contract_requires_nodes_edges_and_seed_labels() -> None:
    value = {
        "primary_role": "train",
        "bindings": [
            {
                "name": "nodes",
                "feature_names": ["node_id", "feature_0", "feature_1"],
                "label_name": None,
            },
            {
                "name": "edges",
                "feature_names": ["source", "destination"],
                "label_name": None,
            },
            {
                "name": "train",
                "feature_names": ["node_id"],
                "label_name": "label",
            },
        ],
        "descriptors": {"nodes": {}, "edges": {}, "train": {}},
    }

    assert HomogeneousGraphInputValidator().validate(value) == value


def test_graphsage_model_runs_on_homogeneous_graph() -> None:
    model = _build_model(input_features=2, hidden_features=4, num_classes=2)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 0, 1]], dtype=torch.long
    )

    output = model(features, edge_index)

    assert output.shape == (4, 2)


def test_rgcn_model_runs_on_relational_graph() -> None:
    model = _build_model(
        input_features=2,
        hidden_features=4,
        num_classes=2,
        model_kind="rgcn",
        num_relations=2,
    )
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    assert model(features, edge_index, edge_type).shape == (4, 2)


def test_relational_contract_requires_relation_and_topology_evidence() -> None:
    input_value = {
        "primary_role": "train",
        "bindings": [
            {
                "name": "nodes",
                "feature_names": ["node_id", "f0", "f1"],
                "label_name": None,
            },
            {
                "name": "edges",
                "feature_names": ["source", "destination", "relation"],
                "label_name": None,
            },
            {
                "name": "train",
                "feature_names": ["node_id"],
                "label_name": "label",
            },
        ],
        "descriptors": {"nodes": {}, "edges": {}, "train": {}},
    }
    assert RelationalGraphInputValidator().validate(input_value) == input_value
    coverage = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {"input_rows": {"train": 2, "nodes": 4, "edges": 4}},
            {"input_rows": {"train": 2, "nodes": 4, "edges": 4}},
        ],
        "state": {
            "details": {
                "sampling": "full_neighborhood",
                "topology_kind": "relational",
            }
        },
    }
    assert RelationalGraphCoverageValidator().validate(coverage) == coverage


def test_graph_coverage_contract_requires_all_roles() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {
                "input_rows": {"train": 2, "nodes": 4, "edges": 6},
            },
            {
                "input_rows": {"train": 2, "nodes": 4, "edges": 6},
            },
        ],
        "state": {"details": {"sampling": "full_neighborhood"}},
    }

    assert GraphCoverageValidator().validate(value) == value
