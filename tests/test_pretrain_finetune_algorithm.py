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
from tributo.algorithms.api import AlgorithmConfigurationError, AlgorithmExecutionError
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
    _gradient_clip_norm,
    _load_encoder_checkpoint,
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


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_pretrain_finetune_rejects_invalid_gradient_clip_norm(value: float) -> None:
    with pytest.raises(AlgorithmConfigurationError, match="positive and finite"):
        _gradient_clip_norm({"max_gradient_norm": value})


def test_finetune_stage_loads_pretrain_stage_dependency(tmp_path) -> None:
    source = _pretrain_model(2, 4)
    target = _finetune_model(2, 4)
    torch.save(source.encoder.state_dict(), tmp_path / "encoder.pt")

    class Checkpoint:
        @contextmanager
        def as_directory(self):
            yield str(tmp_path)

    runtime = TorchRuntimeContext({}, "example.pretrain", 1, "a" * 64, "b" * 64)
    stage = TorchStageContext(
        runtime,
        "finetune",
        1,
        True,
        ("train",),
        predecessor_stage_id="pretrain",
    )
    checkpoint = TorchWorkerCheckpointContext(
        stage,
        "stage_dependency",
        TorchCheckpointRef(Checkpoint()),
    )

    _load_encoder_checkpoint(checkpoint, target)

    assert all(
        torch.equal(
            source.encoder.state_dict()[name], target.encoder.state_dict()[name]
        )
        for name in source.encoder.state_dict()
    )


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
