"""Tests for the official homogeneous PyG GraphSAGE algorithm."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from ray.train import Checkpoint
from tributo.algorithms.api import AlgorithmExecutionError, DistributionStrategy
from tributo.algorithms.spi import FrameworkNativeAlgorithm
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.runtime import BundleModelLoader
from tributo_algorithms_graph_pyg import (
    GRAPHSAGE_DESCRIPTOR,
    RGCN_DESCRIPTOR,
    DistributedGraphSAGE,
    DistributedRGCN,
)
from tributo_algorithms_graph_pyg.algorithm import (
    _build_model,
    _graph_identity_digest,
    _graph_source_fingerprint,
    _node_lookup_model,
    _scaling_config,
    _state_digest,
    _tensor_digest,
    export_result,
)
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


def test_graph_scaling_config_spreads_workers() -> None:
    plan = cast(
        Any,
        SimpleNamespace(
            runtime=SimpleNamespace(
                worker_count=2,
                num_cpus=1.0,
                num_gpus=0.0,
                custom_resources={"graph": 0.25},
            )
        ),
    )
    scaling = cast(Any, _scaling_config(plan))
    assert scaling.num_workers == 2
    assert scaling.placement_strategy == "SPREAD"
    assert scaling.resources_per_worker == {"CPU": 1.0, "GPU": 0.0, "graph": 0.25}


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


def test_graph_lookup_returns_validity_without_negative_index_wraparound() -> None:
    model = _build_model(input_features=2, hidden_features=4, num_classes=2)
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]])
    lookup = cast(Any, _node_lookup_model(model, logits))
    output = lookup(torch.tensor([-1, 2, 3], dtype=torch.int64))
    assert tuple(lookup.state_dict()) == tuple(cast(Any, model).state_dict())
    torch.testing.assert_close(output[1, :2], logits[2])
    torch.testing.assert_close(output[[0, 2], :2], torch.zeros((2, 2)))
    torch.testing.assert_close(output[:, 2], torch.tensor([0.0, 1.0, 0.0]))


def _export_graph_checkpoint(
    tmp_path: Path,
    *,
    model_config_updates: Mapping[str, object] | None = None,
    removed_config_keys: tuple[str, ...] = (),
    inference_logits: object | None = None,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    model = cast(torch.nn.Module, _build_model(2, 4, 2))
    torch.save(model.state_dict(), checkpoint_root / "model.pt")
    logits = (
        torch.tensor([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]])
        if inference_logits is None
        else inference_logits
    )
    torch.save(logits, checkpoint_root / "inference_logits.pt")
    topology_digest = "a" * 64
    node_feature_digest = "b" * 64
    config: dict[str, object] = {
        "input_features": 2,
        "hidden_features": 4,
        "num_classes": 2,
        "node_features": ["f0", "f1"],
        "model_kind": "graphsage",
        "num_relations": 1,
        "graph_identity_version": 1,
        "topology_digest": topology_digest,
        "node_feature_digest": node_feature_digest,
        "graph_identity_digest": _graph_identity_digest(
            topology_digest=topology_digest,
            node_feature_digest=node_feature_digest,
            node_feature_names=("f0", "f1"),
        ),
        "inference_logits_digest": (
            _tensor_digest(logits) if isinstance(logits, torch.Tensor) else "c" * 64
        ),
    }
    if model_config_updates is not None:
        config.update(model_config_updates)
    for key in removed_config_keys:
        config.pop(key)
    (checkpoint_root / "model_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    export_result(
        result=SimpleNamespace(metrics={"loss": 0.1}),
        checkpoint=Checkpoint.from_directory(str(checkpoint_root)),
        plan=cast(
            Any,
            SimpleNamespace(
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}},
                resolution=SimpleNamespace(
                    implementation_id="tributo.official.graph_pyg.graphsage"
                ),
            ),
        ),
        run_id="graph-export-invalid-checkpoint-test",
    )


@pytest.mark.parametrize("version", [True, 1.0, "1", None])
def test_graph_export_rejects_non_integer_identity_version(
    tmp_path: Path,
    version: object,
) -> None:
    with pytest.raises(
        AlgorithmExecutionError, match="identity version is unsupported"
    ):
        _export_graph_checkpoint(
            tmp_path,
            model_config_updates={"graph_identity_version": version},
        )


@pytest.mark.parametrize(
    ("model_config_updates", "removed_config_keys", "message"),
    [
        ({"topology_digest": "invalid"}, (), "topology_digest is invalid"),
        ({}, ("node_feature_digest",), "node_feature_digest is invalid"),
        ({"graph_identity_digest": "c" * 64}, (), "identity digest mismatched"),
        ({"inference_logits_digest": "d" * 64}, (), "logits digest mismatched"),
    ],
)
def test_graph_export_rejects_corrupt_identity_metadata(
    tmp_path: Path,
    model_config_updates: Mapping[str, object],
    removed_config_keys: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(AlgorithmExecutionError, match=message):
        _export_graph_checkpoint(
            tmp_path,
            model_config_updates=model_config_updates,
            removed_config_keys=removed_config_keys,
        )


@pytest.mark.parametrize("invalid_logits", ["wrong_class_width", "non_finite"])
def test_graph_export_rejects_invalid_inference_logits(
    tmp_path: Path,
    invalid_logits: str,
) -> None:
    logits = (
        torch.ones((3, 3))
        if invalid_logits == "wrong_class_width"
        else torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    )
    with pytest.raises(
        AlgorithmExecutionError,
        match="inference logits do not match the finite class contract",
    ):
        _export_graph_checkpoint(tmp_path, inference_logits=logits)


def test_graph_export_publishes_sanitized_transductive_inference(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    model = cast(torch.nn.Module, _build_model(2, 4, 2))
    torch.save(model.state_dict(), checkpoint_root / "model.pt")
    inference_logits = torch.tensor([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]])
    torch.save(inference_logits, checkpoint_root / "inference_logits.pt")
    topology_digest = "a" * 64
    node_feature_digest = "b" * 64
    graph_identity_digest = _graph_identity_digest(
        topology_digest=topology_digest,
        node_feature_digest=node_feature_digest,
        node_feature_names=("f0", "f1"),
    )
    inference_logits_digest = _tensor_digest(inference_logits)
    (checkpoint_root / "model_config.json").write_text(
        json.dumps(
            {
                "input_features": 2,
                "hidden_features": 4,
                "num_classes": 2,
                "node_features": ["f0", "f1"],
                "model_kind": "graphsage",
                "num_relations": 1,
                "graph_identity_version": 1,
                "topology_digest": topology_digest,
                "node_feature_digest": node_feature_digest,
                "graph_identity_digest": graph_identity_digest,
                "inference_logits_digest": inference_logits_digest,
            }
        ),
        encoding="utf-8",
    )
    execution = export_result(
        result=SimpleNamespace(metrics={"loss": 0.1}),
        checkpoint=Checkpoint.from_directory(str(checkpoint_root)),
        plan=cast(
            Any,
            SimpleNamespace(
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}},
                resolution=SimpleNamespace(
                    implementation_id="tributo.official.graph_pyg.graphsage"
                ),
            ),
        ),
        run_id="graph-export-test",
    )
    bundle_uri = cast(str, execution.outputs["bundle_uri"])
    manifest = BundleReader().read_manifest(bundle_uri)
    assert manifest.source_info.source_fingerprint == _graph_source_fingerprint(
        model_digest=_state_digest(cast(Mapping[str, object], model.state_dict())),
        graph_identity_digest=graph_identity_digest,
        inference_logits_digest=inference_logits_digest,
    )
    runtime = BundleModelLoader().open(
        bundle_uri,
        role="inference",
        use_case="batch",
    )
    try:
        outputs = runtime.predict(
            {"node_id": np.asarray([0, 2, -1, 3], dtype=np.int64)}
        )
    finally:
        runtime.close()
    assert outputs["output"].shape == (4, 3)
    np.testing.assert_array_equal(outputs["output"][:, 2], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(outputs["output"][[2, 3], :2], 0.0)
    config_paths = tuple(Path(bundle_uri).rglob("model_config.json"))
    assert config_paths
    for config_path in config_paths:
        published_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert "node_values" not in published_config
        assert "edge_index" not in published_config
        assert "edge_type" not in published_config
        assert published_config["graph_identity_version"] == 1
        assert published_config["topology_digest"] == topology_digest
        assert published_config["node_feature_digest"] == node_feature_digest
        assert published_config["graph_identity_digest"] == graph_identity_digest
        assert published_config["inference_logits_digest"] == inference_logits_digest


def test_graph_source_fingerprint_includes_derived_logits() -> None:
    first = _graph_source_fingerprint(
        model_digest="a" * 64,
        graph_identity_digest="b" * 64,
        inference_logits_digest="c" * 64,
    )
    second = _graph_source_fingerprint(
        model_digest="a" * 64,
        graph_identity_digest="b" * 64,
        inference_logits_digest="d" * 64,
    )

    assert first != second


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
