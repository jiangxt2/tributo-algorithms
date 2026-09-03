"""Tests for the jagged-history RayTorchAdapter."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pandas as pd
import pytest
import torch
from tributo.algorithms import (
    DistributionStrategy,
    RayTorchAdapter,
    TorchCheckpointRef,
)
from tributo.algorithms.api import AlgorithmExecutionError
from tributo.algorithms.spi import (
    TorchArtifactContext,
    TorchRuntimeContext,
    TorchStageContext,
    TorchWorkerCheckpointContext,
)
from tributo_algorithms_recsys_torch import (
    JAGGED_DESCRIPTOR,
    DistributedJaggedEmbedding,
)
from tributo_algorithms_recsys_torch.contracts import (
    JaggedOutputValidator,
    JaggedTorchCoverageValidator,
    JaggedTorchInputValidator,
)
from tributo_algorithms_recsys_torch.jagged import (
    _jagged_tensors,
    _load_retry_state,
    _model,
    _padded_inference_model,
    _state_digest,
)


def test_jagged_descriptor_uses_ray_torch_adapter() -> None:
    registration = JAGGED_DESCRIPTOR.registration
    assert (
        registration.distribution_spec.strategy is DistributionStrategy.RAY_TRAIN_TORCH
    )
    assert registration.distribution_spec.policy.loop_owner == "adapter"
    assert issubclass(DistributedJaggedEmbedding, RayTorchAdapter)


def test_jagged_batch_builds_embedding_bag_offsets() -> None:
    frame = pd.DataFrame(
        {"user": [0, 1], "history": [[1, 2], [2]], "item": [3, 4], "label": [1.0, 0.0]}
    )
    user, values, offsets, candidate, labels, token_count = _jagged_tensors(
        frame,
        user_col="user",
        history_col="history",
        candidate_col="item",
        label_col="label",
        item_count=8,
        device=torch.device("cpu"),
    )
    assert user.shape == candidate.shape == labels.reshape(-1).shape == (2,)
    assert values.tolist() == [1, 2, 2]
    assert offsets.tolist() == [0, 2]
    assert token_count == 3


def test_padded_jagged_model_emits_validity_column() -> None:
    model = _model(user_count=4, item_count=8, embedding_dim=3)
    padded = _padded_inference_model(model)
    output = padded(
        torch.tensor([0, -1]), torch.tensor([[1, -1], [1, 2]]), torch.tensor([3, 99])
    )
    assert output.shape == (2, 2)
    assert output[:, 1].tolist() == [1.0, 0.0]
    assert len(_state_digest(model.state_dict())) == 64


def test_jagged_coverage_requires_token_conservation() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {
                "input_rows": {
                    "train": 2,
                    "coverage.history_tokens": 3,
                    "coverage.routed_owned_tokens": 3,
                    "coverage.positive_pairs": 1,
                    "coverage.negative_pairs": 1,
                }
            }
        ],
        "state": {
            "details": {"jagged": True, "routing": "all_to_all_single_owner_mod"}
        },
    }
    assert JaggedTorchCoverageValidator().validate(value) == value


def test_jagged_v2_input_rejects_sample_weights() -> None:
    with pytest.raises(ValueError, match="named, unweighted"):
        JaggedTorchInputValidator().validate(
            {
                "bindings": [
                    {
                        "role": "train",
                        "feature_names": ["user_id", "item_history", "item_id"],
                        "label_name": "label",
                        "sample_weight_name": "weight",
                    }
                ]
            }
        )


def test_jagged_output_contract_rejects_failed_or_missing_bundle() -> None:
    for value in (
        {"status": "failed", "outputs": {"bundle_uri": "/tmp/model"}},
        {"status": "succeeded", "outputs": {}},
    ):
        with pytest.raises(ValueError):
            JaggedOutputValidator().validate(value)


def test_jagged_export_source_preserves_inference_contract(tmp_path) -> None:
    model = _model(user_count=4, item_count=8, embedding_dim=3)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    metadata = {
        "model": {"user_count": 4, "item_count": 8, "embedding_dim": 3},
        "data": {
            "user_col": "user_id",
            "history_col": "item_history",
            "candidate_col": "item_id",
            "label_col": "label",
            "inference_history_width": 2,
        },
    }
    (tmp_path / "model_config.json").write_text(json.dumps(metadata), encoding="utf-8")

    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield str(tmp_path)

    runtime = TorchRuntimeContext(
        {
            "model": {"user_count": 4, "item_count": 8, "embedding_dim": 3},
            "data": {"inference_history_width": 2},
        },
        "example.jagged",
        1,
        "a" * 64,
        "b" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    ref = TorchCheckpointRef(Checkpoint())
    adapter = DistributedJaggedEmbedding()
    artifact_context = TorchArtifactContext(stage, ref)
    plan = adapter.artifact_plan(artifact_context)
    assert plan.source_kind == "torch_module"
    assert tuple(field["name"] for field in plan.input_signature) == (
        "user_id",
        "item_history",
        "item_id",
    )
    assert plan.output_signature == (
        {"name": "output", "dtype": "float32", "shape": ("batch", 2)},
    )
    assert tuple(target["name"] for target in plan.targets) == (
        "jagged-ranking-model",
        "jagged-ranking-inference",
    )
    assert plan.roles == {
        "model": "jagged-ranking-model",
        "inference": "jagged-ranking-inference",
    }
    with adapter.open_export_source(ref, artifact_context) as source:
        assert source.checkpoint_contract is not None
        assert source.checkpoint_contract.output_schema[0].shape == ("batch", 2)
        predictions = source.model_object(
            source.sample_inputs["user_id"],
            source.sample_inputs["item_history"],
            source.sample_inputs["item_id"],
        )
        assert predictions.shape == (2, 2)
        torch.testing.assert_close(predictions[:, 1], torch.ones(2))

    outside_model = tmp_path.parent / f"{tmp_path.name}-outside-model.pt"
    outside_model.write_bytes(b"not-a-model")
    (tmp_path / "model.pt").unlink()
    (tmp_path / "model.pt").symlink_to(outside_model)
    with pytest.raises(AlgorithmExecutionError, match="missing payloads"):
        with adapter.open_export_source(ref, artifact_context):
            pass
    (tmp_path / "model.pt").unlink()
    with pytest.raises(AlgorithmExecutionError, match="missing payloads"):
        with adapter.open_export_source(ref, artifact_context):
            pass

    torch.save(
        _model(user_count=4, item_count=8, embedding_dim=4).state_dict(),
        tmp_path / "model.pt",
    )
    with pytest.raises(AlgorithmExecutionError, match="payload is incompatible"):
        with adapter.open_export_source(ref, artifact_context):
            pass


def test_jagged_checkpoint_source_requires_a_checkpoint() -> None:
    with pytest.raises(AlgorithmExecutionError, match="no checkpoint"):
        DistributedJaggedEmbedding().checkpoint_source(
            type("Result", (), {})(), object()
        )


def test_jagged_retry_checkpoint_requires_optimizer_state(tmp_path) -> None:
    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield str(tmp_path)

    runtime = TorchRuntimeContext({}, "example.jagged", 1, "a" * 64, "b" * 64)
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    context = TorchWorkerCheckpointContext(
        stage, "ray_failure_retry", TorchCheckpointRef(Checkpoint())
    )
    model = _model(user_count=4, item_count=8, embedding_dim=3)
    optimizer = torch.optim.Adam(model.parameters())
    torch.save(model.state_dict(), tmp_path / "model.pt")
    with pytest.raises(AlgorithmExecutionError, match="model/optimizer state"):
        _load_retry_state(context, model, optimizer, rank=0)


def test_jagged_retry_checkpoint_restores_optimizer_and_rng_state(tmp_path) -> None:
    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield str(tmp_path)

    runtime = TorchRuntimeContext({}, "example.jagged", 1, "a" * 64, "b" * 64)
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    context = TorchWorkerCheckpointContext(
        stage, "ray_failure_retry", TorchCheckpointRef(Checkpoint())
    )
    torch.manual_seed(23)
    saved_model = _model(user_count=4, item_count=8, embedding_dim=3)
    saved_optimizer = torch.optim.Adam(saved_model.parameters())
    sum(parameter.square().sum() for parameter in saved_model.parameters()).backward()
    saved_optimizer.step()
    expected_state = {
        name: value.detach().clone() for name, value in saved_model.state_dict().items()
    }
    expected_rng = torch.get_rng_state().clone()
    torch.save(saved_model.state_dict(), tmp_path / "model.pt")
    torch.save(saved_optimizer.state_dict(), tmp_path / "optimizer.pt")
    torch.save({"states": [expected_rng.numpy().tobytes()]}, tmp_path / "rng_state.pt")

    target_model = _model(user_count=4, item_count=8, embedding_dim=3)
    target_optimizer = torch.optim.Adam(target_model.parameters())
    torch.manual_seed(99)
    assert _load_retry_state(context, target_model, target_optimizer, rank=0) == 0
    for name, value in target_model.state_dict().items():
        torch.testing.assert_close(value, expected_state[name])
    assert target_optimizer.state_dict()["state"]
    torch.testing.assert_close(torch.get_rng_state(), expected_rng)

    wrong_source = TorchWorkerCheckpointContext(
        stage, "stage_dependency", TorchCheckpointRef(Checkpoint())
    )
    with pytest.raises(AlgorithmExecutionError, match="only Ray failure retry"):
        _load_retry_state(wrong_source, target_model, target_optimizer, rank=0)
