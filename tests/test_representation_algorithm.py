"""Tests for the typed self-supervised Autoencoder TorchRecipe."""

from __future__ import annotations

import hashlib

import pytest
import torch
from tributo.algorithms import (
    DistributionStrategy,
    TorchBatchContext,
    TorchBuildContext,
    TorchRuntimeContext,
    TorchStageContext,
    TorchStageRunIdentity,
    TorchStepContext,
)
from tributo_algorithms_representation import TABULAR_AUTOENCODER_DESCRIPTOR
from tributo_algorithms_representation.contracts import RepresentationConfigValidator
from tributo_algorithms_representation.recipe import TabularAutoencoderRecipe


def _context() -> TorchBuildContext:
    policy = TABULAR_AUTOENCODER_DESCRIPTOR.registration.distribution_spec.policy
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "ae",
        "ae",
        hashlib.sha256(b"ae").hexdigest(),
        policy.digest,
        policy.execution_plan.digest,
    )
    runtime = TorchRuntimeContext(
        {"model": {"input_features": 3, "latent_features": 2}},
        "ae",
        1,
        policy.digest,
        policy.execution_plan.digest,
        identity,
        input_binding_digest="1" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    return TorchBuildContext(runtime, stage)


def test_autoencoder_descriptor_uses_torch_runtime() -> None:
    registration = TABULAR_AUTOENCODER_DESCRIPTOR.registration
    assert (
        registration.distribution_spec.strategy is DistributionStrategy.RAY_TRAIN_TORCH
    )
    assert registration.contract_bindings is not None


def test_autoencoder_uses_element_normalizer() -> None:
    recipe = TabularAutoencoderRecipe()
    build = _context()
    batch_context = TorchBatchContext(build.stage, ("f0", "f1", "f2"))
    modules = recipe.build_modules(build)
    adapted = recipe.adapt_batch(
        {
            "f0": torch.tensor([0.0, 1.0]),
            "f1": torch.tensor([1.0, 0.0]),
            "f2": torch.tensor([0.5, 0.5]),
        },
        batch_context,
    )
    result = recipe.training_step(modules, adapted, TorchStepContext(build.stage, 0, 0))
    assert result.outputs["output"].shape == (2, 3)
    assert result.loss.normalizer == 6
    assert result.metrics["reconstruction_mse"].normalizer == 6


def test_autoencoder_config_accepts_positive_accumulation() -> None:
    value = {
        "optimizer": {"accumulation_steps": 2},
        "output": {"bundle_uri": "/tmp/model"},
    }
    assert RepresentationConfigValidator().validate(value) == value


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_autoencoder_config_rejects_invalid_accumulation(value: object) -> None:
    with pytest.raises(ValueError, match="accumulation_steps"):
        RepresentationConfigValidator().validate(
            {
                "optimizer": {"accumulation_steps": value},
                "output": {"bundle_uri": "/tmp/model"},
            }
        )
