"""Tests for the pretrain/finetune RayTorchAdapter component plan."""

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
from tributo.algorithms.api import AlgorithmExecutionError
from tributo.algorithms.spi import (
    TorchArtifactContext,
    TorchRuntimeContext,
    TorchStageContext,
    TorchWorkerCheckpointContext,
)
from tributo_algorithms_multistage_torch import (
    PRETRAIN_FINETUNE_DESCRIPTOR,
    DistributedPretrainFinetune,
)
from tributo_algorithms_multistage_torch.contracts import (
    PretrainFinetuneOutputValidator,
    PretrainFinetuneTorchCoverageValidator,
)
from tributo_algorithms_multistage_torch.pretrain import (
    _finetune_model,
    _load_retry_state,
    _pretrain_model,
    _state_digest,
)


def test_pretrain_finetune_descriptor_declares_two_stages() -> None:
    registration = PRETRAIN_FINETUNE_DESCRIPTOR.registration
    assert (
        registration.distribution_spec.strategy is DistributionStrategy.RAY_TRAIN_TORCH
    )
    policy = registration.distribution_spec.policy
    assert policy.loop_owner == "adapter"
    assert [stage.stage_id for stage in policy.execution_plan.stages] == [
        "pretrain",
        "finetune",
    ]
    assert policy.execution_plan.final_stage_id == "finetune"
    assert issubclass(DistributedPretrainFinetune, RayTorchAdapter)


def test_pretraining_and_finetuning_models_have_expected_heads() -> None:
    pretrain = _pretrain_model(2, 4)
    finetune = _finetune_model(2, 4)
    assert hasattr(pretrain, "decoder")
    assert hasattr(finetune, "classifier")
    assert len(_state_digest(pretrain.state_dict())) == 64


def test_pretrain_finetune_coverage_requires_both_stages() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "state": {
            "details": {
                "component_stages": "pretrain,finetune",
                "stage.pretrain.rows": 2,
                "stage.finetune.rows": 2,
            }
        },
    }
    assert PretrainFinetuneTorchCoverageValidator().validate(value) == value


def test_pretrain_finetune_output_contract_rejects_failed_or_incomplete_bundle() -> (
    None
):
    for value in (
        {"status": "failed", "outputs": {"bundle_uri": "/tmp/model"}},
        {"status": "succeeded", "outputs": {"bundle_uri": "/tmp/model"}},
    ):
        with pytest.raises(ValueError):
            PretrainFinetuneOutputValidator().validate(value)


def test_pretrain_worker_config_uses_input_binding_names() -> None:
    runtime = TorchRuntimeContext(
        algorithm_config={"model": {}, "ray": {"storage_path": "/hidden"}},
        implementation_id="tributo.official.multistage_torch.pretrain_finetune",
        world_size=2,
        policy_digest="a" * 64,
        execution_plan_digest="b" * 64,
        input_bindings={
            "train": {
                "feature_names": ["x0", "x1"],
                "label_name": "target",
            }
        },
    )
    context = TorchStageContext(runtime, "pretrain", 0, False, ("train",))
    config = DistributedPretrainFinetune().worker_config(context)
    assert config["feature_names"] == ("x0", "x1")
    assert config["label_name"] == "target"
    assert "ray" not in config


def test_pretrain_export_source_preserves_inference_contract(tmp_path) -> None:
    model = _finetune_model(2, 4)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "input_features": 2,
                "hidden_features": 4,
                "feature_names": ["x0", "x1"],
                "stage": "finetune",
            }
        ),
        encoding="utf-8",
    )

    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield str(tmp_path)

    runtime = TorchRuntimeContext(
        {"model": {"input_features": 2}},
        "example.pretrain",
        1,
        "a" * 64,
        "b" * 64,
    )
    stage = TorchStageContext(runtime, "finetune", 1, True, ("train",))
    ref = TorchCheckpointRef(Checkpoint())
    context = TorchArtifactContext(stage, ref)
    adapter = DistributedPretrainFinetune()
    plan = adapter.artifact_plan(context)
    assert plan.source_kind == "torch_module"
    assert plan.input_signature == (
        {"name": "float_input", "dtype": "float32", "shape": ("batch", 2)},
    )
    assert plan.output_signature == (
        {"name": "output", "dtype": "float32", "shape": ("batch", 1)},
    )
    assert tuple(target["name"] for target in plan.targets) == (
        "finetuned-model",
        "finetuned-inference",
    )
    assert plan.roles == {
        "model": "finetuned-model",
        "inference": "finetuned-inference",
    }
    with adapter.open_export_source(ref, context) as source:
        assert source.checkpoint_contract is not None
        assert source.checkpoint_contract.input_schema[0].name == "float_input"
        predictions = source.model_object(source.sample_inputs["float_input"])
        assert predictions.shape == (2, 1)

    outside_model = tmp_path.parent / f"{tmp_path.name}-outside-model.pt"
    outside_model.write_bytes(b"not-a-model")
    (tmp_path / "model.pt").unlink()
    (tmp_path / "model.pt").symlink_to(outside_model)
    with pytest.raises(AlgorithmExecutionError, match="missing payloads"):
        with adapter.open_export_source(ref, context):
            pass
    (tmp_path / "model.pt").unlink()
    with pytest.raises(AlgorithmExecutionError, match="missing payloads"):
        with adapter.open_export_source(ref, context):
            pass

    torch.save(_finetune_model(2, 5).state_dict(), tmp_path / "model.pt")
    with pytest.raises(AlgorithmExecutionError, match="payload is incompatible"):
        with adapter.open_export_source(ref, context):
            pass


def test_pretrain_checkpoint_source_requires_a_checkpoint() -> None:
    with pytest.raises(AlgorithmExecutionError, match="no checkpoint"):
        DistributedPretrainFinetune().checkpoint_source(
            type("Result", (), {})(), object()
        )


def test_pretrain_retry_checkpoint_requires_optimizer_state(tmp_path) -> None:
    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield str(tmp_path)

    runtime = TorchRuntimeContext({}, "example.pretrain", 1, "a" * 64, "b" * 64)
    stage = TorchStageContext(runtime, "pretrain", 0, True, ("train",))
    context = TorchWorkerCheckpointContext(
        stage, "ray_failure_retry", TorchCheckpointRef(Checkpoint())
    )
    model = _pretrain_model(2, 4)
    optimizer = torch.optim.Adam(model.parameters())
    torch.save(model.state_dict(), tmp_path / "model.pt")
    with pytest.raises(AlgorithmExecutionError, match="model/optimizer state"):
        _load_retry_state(context, model, optimizer, rank=0)


def test_pretrain_retry_checkpoint_restores_optimizer_and_rng_state(tmp_path) -> None:
    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield str(tmp_path)

    runtime = TorchRuntimeContext({}, "example.pretrain", 1, "a" * 64, "b" * 64)
    stage = TorchStageContext(runtime, "pretrain", 0, True, ("train",))
    context = TorchWorkerCheckpointContext(
        stage, "ray_failure_retry", TorchCheckpointRef(Checkpoint())
    )
    torch.manual_seed(31)
    saved_model = _pretrain_model(2, 4)
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

    target_model = _pretrain_model(2, 4)
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
