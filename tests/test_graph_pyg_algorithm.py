"""Tests for GraphSAGE and R-GCN RayTorchAdapters."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
import torch
from tributo.algorithms import (
    DistributionStrategy,
    RayTorchAdapter,
    TorchCheckpointRef,
)
from tributo.algorithms.api import AlgorithmConfigurationError, AlgorithmExecutionError
from tributo.algorithms.spi import (
    TorchArtifactContext,
    TorchRuntimeContext,
    TorchStageContext,
)
from tributo_algorithms_graph_pyg import (
    GRAPHSAGE_DESCRIPTOR,
    RGCN_DESCRIPTOR,
    DistributedGraphSAGE,
    DistributedRGCN,
)
from tributo_algorithms_graph_pyg.algorithm import (
    _build_model,
    _gradient_clip_norm,
    _graph_identity_digest,
    _graph_source_fingerprint,
    _integer_values,
    _node_lookup_model,
    _state_digest,
    _tensor_digest,
)
from tributo_algorithms_graph_pyg.contracts import (
    GraphConfigValidator,
    GraphOutputValidator,
    GraphSAGETorchCoverageValidator,
    GraphSAGETorchInputValidator,
    RelationalGraphConfigValidator,
    RGCNTorchCoverageValidator,
    RGCNTorchInputValidator,
)


def test_graph_descriptors_use_ray_torch_adapter_and_budgets() -> None:
    for descriptor in (GRAPHSAGE_DESCRIPTOR, RGCN_DESCRIPTOR):
        registration = descriptor.registration
        assert (
            registration.distribution_spec.strategy
            is DistributionStrategy.RAY_TRAIN_TORCH
        )
        policy = registration.distribution_spec.policy
        assert policy.loop_owner == "adapter"
        assert policy.max_replicated_bytes_per_worker == 536_870_912
        assert all(
            route.mode in {"split_exact", "replicate"}
            for route in policy.dataset_routing
        )


def test_graph_adapters_implement_public_contract() -> None:
    assert issubclass(DistributedGraphSAGE, RayTorchAdapter)
    assert issubclass(DistributedRGCN, RayTorchAdapter)
    assert DistributedGraphSAGE().model_kind == "graphsage"
    assert DistributedRGCN().model_kind == "rgcn"


@pytest.mark.parametrize("adapter", [DistributedGraphSAGE(), DistributedRGCN()])
def test_graph_artifact_plan_declares_targets_roles_and_signature(adapter) -> None:
    runtime = TorchRuntimeContext(
        {"model": {"num_classes": 3}},
        "example.graph",
        1,
        "a" * 64,
        "b" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    plan = adapter.artifact_plan(TorchArtifactContext(stage))
    assert plan.source_kind == "torch_module"
    assert plan.input_signature == (
        {"name": "node_id", "dtype": "int64", "shape": ("batch",)},
    )
    assert plan.output_signature == (
        {"name": "output", "dtype": "float32", "shape": ("batch", 4)},
    )
    assert tuple(target["name"] for target in plan.targets) == (
        "graph-model",
        "graph-inference",
    )
    assert tuple(target["format"] for target in plan.targets) == (
        "safetensors",
        "onnx",
    )
    assert plan.roles == {"model": "graph-model", "inference": "graph-inference"}


def test_graph_worker_config_is_derived_from_typed_bindings() -> None:
    runtime = TorchRuntimeContext(
        algorithm_config={"model": {}, "ray": {"storage_path": "/hidden"}},
        implementation_id="tributo.official.graph_pyg.graphsage",
        world_size=2,
        policy_digest="a" * 64,
        execution_plan_digest="b" * 64,
        input_bindings={
            "nodes": {"feature_names": ["node_id", "f0", "f1"]},
            "edges": {"feature_names": ["source", "destination"]},
            "train": {"feature_names": ["seed_id"], "label_name": "label"},
        },
    )
    context = TorchStageContext(runtime, "train", 0, True, ("train", "nodes", "edges"))
    config = DistributedGraphSAGE().worker_config(context)
    assert "ray" not in config
    assert config["columns"] == {
        "node_id": "node_id",
        "node_features": ["f0", "f1"],
        "edge_source": "source",
        "edge_destination": "destination",
        "seed_node_id": "seed_id",
        "seed_label": "label",
    }


def test_graph_models_run_on_homogeneous_and_relational_graphs() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    assert _build_model(2, 4, 2)(features, edge_index).shape == (3, 2)
    relational = _build_model(2, 4, 2, model_kind="rgcn", num_relations=2)
    edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
    assert relational(features, edge_index, edge_type).shape == (3, 2)


def test_graph_lookup_marks_invalid_node_ids_without_wraparound() -> None:
    model = _build_model(2, 4, 2)
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.25], [-0.5, 2.0]])
    lookup = _node_lookup_model(model, logits)
    output = lookup(torch.tensor([-1, 2, 3], dtype=torch.int64))
    assert output.shape == (3, 3)
    torch.testing.assert_close(output[1, :2], logits[2])
    torch.testing.assert_close(output[[0, 2], :2], torch.zeros((2, 2)))
    torch.testing.assert_close(output[:, 2], torch.tensor([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="integer values"):
        lookup(torch.tensor([1.5]))


@pytest.mark.parametrize("values", [[1.5], [True]])
def test_graph_rejects_non_integer_ids(values: list[object]) -> None:
    with pytest.raises(AlgorithmExecutionError, match="integer values"):
        _integer_values(values, "graph node IDs")


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_graph_rejects_invalid_gradient_clip_norm(value: float) -> None:
    with pytest.raises(AlgorithmConfigurationError, match="positive and finite"):
        _gradient_clip_norm({"max_gradient_norm": value})


def test_graph_identity_and_source_fingerprint_bind_all_state() -> None:
    topology = "a" * 64
    features = "b" * 64
    identity = _graph_identity_digest(
        topology_digest=topology,
        node_feature_digest=features,
        node_feature_names=("f0", "f1"),
    )
    fingerprint = _graph_source_fingerprint(
        model_digest="c" * 64,
        graph_identity_digest=identity,
        inference_logits_digest="d" * 64,
    )
    assert len(identity) == len(fingerprint) == 64
    assert _tensor_digest(torch.ones((2, 2))) != _tensor_digest(torch.zeros((2, 2)))
    assert len(_state_digest(_build_model(2, 4, 2).state_dict())) == 64


def test_graph_contracts_require_all_roles_and_evidence() -> None:
    value = {
        "primary_role": "train",
        "bindings": [
            {"name": "nodes", "feature_names": ["node_id", "f0", "f1"]},
            {"name": "edges", "feature_names": ["source", "destination"]},
            {"name": "train", "feature_names": ["node_id"], "label_name": "label"},
        ],
        "descriptors": {"nodes": {}, "edges": {}, "train": {}},
    }
    assert GraphSAGETorchInputValidator().validate(value) == value
    assert RGCNTorchInputValidator().validate(
        {
            **value,
            "bindings": [
                value["bindings"][0],
                {
                    "name": "edges",
                    "feature_names": ["source", "destination", "relation"],
                },
                value["bindings"][2],
            ],
        }
    )
    evidence = {
        "input_complete": True,
        "distributed": True,
        "workers": [{"input_rows": {"train": 1, "nodes": 2, "edges": 2}}],
        "state": {
            "details": {"sampling": "full_neighborhood", "topology_kind": "homogeneous"}
        },
    }
    assert GraphSAGETorchCoverageValidator().validate(evidence) == evidence
    assert RGCNTorchCoverageValidator().validate(
        {
            **evidence,
            "state": {
                "details": {
                    "sampling": "full_neighborhood",
                    "topology_kind": "relational",
                }
            },
        }
    )


def test_graph_v2_inputs_reject_sample_weights() -> None:
    value = {
        "primary_role": "train",
        "bindings": [
            {"name": "nodes", "feature_names": ["node_id", "f0"]},
            {
                "name": "edges",
                "feature_names": ["source", "destination"],
                "sample_weight_name": "weight",
            },
            {"name": "train", "feature_names": ["node_id"], "label_name": "label"},
        ],
        "descriptors": {"nodes": {}, "edges": {}, "train": {}},
    }
    with pytest.raises(ValueError, match="sample-weight"):
        GraphSAGETorchInputValidator().validate(value)


@pytest.mark.parametrize(
    ("validator", "bindings"),
    [
        (
            GraphSAGETorchInputValidator,
            [
                {"name": "nodes", "feature_names": ["node_id", "f0"]},
                {"name": "edges", "feature_names": ["source", "destination"]},
                {"name": "train", "feature_names": ["node_id"], "label_name": "label"},
            ],
        ),
        (
            RGCNTorchInputValidator,
            [
                {"name": "nodes", "feature_names": ["node_id", "f0"]},
                {
                    "name": "edges",
                    "feature_names": ["source", "destination", "relation"],
                },
                {"name": "train", "feature_names": ["node_id"], "label_name": "label"},
            ],
        ),
    ],
)
def test_graph_inputs_reject_duplicate_or_non_mapping_bindings(
    validator, bindings
) -> None:
    base = {
        "primary_role": "train",
        "bindings": bindings,
        "descriptors": {"nodes": {}, "edges": {}, "train": {}},
    }
    for candidate in (
        bindings + [bindings[0]],
        bindings[:2] + ["not-a-binding"],
    ):
        value = dict(base)
        value["bindings"] = candidate
        with pytest.raises(ValueError, match="binding"):
            validator().validate(value)


@pytest.mark.parametrize(
    ("validator", "model"),
    [
        (GraphConfigValidator, {}),
        (RelationalGraphConfigValidator, {"num_relations": 1}),
    ],
)
def test_graph_configs_reject_zero_epochs(validator, model) -> None:
    with pytest.raises(ValueError, match="training.epochs"):
        validator().validate(
            {
                "model": model,
                "training": {"epochs": 0},
                "output": {"bundle_uri": "/tmp/unused"},
            }
        )


def test_graph_output_contract_rejects_failed_or_missing_bundle() -> None:
    for value in (
        {"status": "failed", "outputs": {"bundle_uri": "/tmp/model"}},
        {"status": "succeeded", "outputs": {}},
    ):
        with pytest.raises(ValueError):
            GraphOutputValidator().validate(value)


def test_graph_export_source_preserves_identity_and_checkpoint_contract(
    tmp_path,
) -> None:
    model = _build_model(2, 4, 2)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    torch.save(logits, tmp_path / "inference_logits.pt")
    topology_digest = "a" * 64
    node_feature_digest = "b" * 64
    node_features = ["f0", "f1"]
    graph_identity_digest = _graph_identity_digest(
        topology_digest=topology_digest,
        node_feature_digest=node_feature_digest,
        node_feature_names=tuple(node_features),
    )
    model_config_path = tmp_path / "model_config.json"
    model_config_path.write_text(
        json.dumps(
            {
                "input_features": 2,
                "hidden_features": 4,
                "num_classes": 2,
                "node_features": node_features,
                "model_kind": "graphsage",
                "num_relations": 1,
                "graph_identity_version": 1,
                "topology_digest": topology_digest,
                "node_feature_digest": node_feature_digest,
                "graph_identity_digest": graph_identity_digest,
                "inference_logits_digest": _tensor_digest(logits),
            }
        ),
        encoding="utf-8",
    )

    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield str(tmp_path)

    runtime = TorchRuntimeContext(
        {"model": {"num_classes": 2}}, "example.graph", 1, "c" * 64, "d" * 64
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    ref = TorchCheckpointRef(Checkpoint())
    context = TorchArtifactContext(stage, ref)
    with DistributedGraphSAGE().open_export_source(ref, context) as source:
        assert source.checkpoint_contract is not None
        assert source.feature_schema["graph_identity_digest"] == graph_identity_digest
        predictions = source.model_object(source.sample_inputs["node_id"])
        assert predictions.shape == (2, 3)
        torch.testing.assert_close(predictions[:, -1], torch.ones(2))

    outside_model = tmp_path.parent / f"{tmp_path.name}-outside-model.pt"
    outside_model.write_bytes(b"not-a-model")
    (tmp_path / "model.pt").unlink()
    (tmp_path / "model.pt").symlink_to(outside_model)
    with pytest.raises(AlgorithmExecutionError, match="missing payloads"):
        with DistributedGraphSAGE().open_export_source(ref, context):
            pass
    (tmp_path / "model.pt").unlink()
    torch.save(model.state_dict(), tmp_path / "model.pt")

    corrupted = json.loads(model_config_path.read_text(encoding="utf-8"))
    corrupted["graph_identity_digest"] = "c" * 64
    model_config_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(AlgorithmExecutionError, match="identity digest mismatched"):
        with DistributedGraphSAGE().open_export_source(ref, context):
            pass


def test_rgcn_export_source_preserves_typed_signature_and_inference(tmp_path) -> None:
    model = _build_model(2, 4, 2, model_kind="rgcn", num_relations=2)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    torch.save(logits, tmp_path / "inference_logits.pt")
    node_features = ["f0", "f1"]
    topology_digest = "a" * 64
    node_feature_digest = "b" * 64
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "input_features": 2,
                "hidden_features": 4,
                "num_classes": 2,
                "node_features": node_features,
                "model_kind": "rgcn",
                "num_relations": 2,
                "graph_identity_version": 1,
                "topology_digest": topology_digest,
                "node_feature_digest": node_feature_digest,
                "graph_identity_digest": _graph_identity_digest(
                    topology_digest=topology_digest,
                    node_feature_digest=node_feature_digest,
                    node_feature_names=tuple(node_features),
                ),
                "inference_logits_digest": _tensor_digest(logits),
            }
        ),
        encoding="utf-8",
    )

    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield str(tmp_path)

    runtime = TorchRuntimeContext(
        {"model": {"num_classes": 2}},
        "example.rgcn",
        1,
        "c" * 64,
        "d" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    ref = TorchCheckpointRef(Checkpoint())
    context = TorchArtifactContext(stage, ref)
    with DistributedRGCN().open_export_source(ref, context) as source:
        assert source.checkpoint_contract is not None
        assert source.checkpoint_contract.output_schema[0].shape == ("batch", 3)
        predictions = source.model_object(source.sample_inputs["node_id"])
        assert predictions.shape == (2, 3)


def test_graph_checkpoint_source_requires_a_checkpoint() -> None:
    with pytest.raises(AlgorithmExecutionError, match="no checkpoint"):
        DistributedGraphSAGE().checkpoint_source(type("Result", (), {})(), object())
