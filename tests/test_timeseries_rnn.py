"""Tests for fixed-window LSTM and GRU TorchRecipes."""

from __future__ import annotations

import hashlib

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
from tributo_algorithms_timeseries.rnn_descriptor import GRU_DESCRIPTOR, LSTM_DESCRIPTOR
from tributo_algorithms_timeseries.rnn_recipe import GRURecipe, LSTMRecipe

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "lag_0": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "lag_1": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "lag_2": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "lag_3": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }


def _context(descriptor: object) -> TorchBuildContext:
    policy = descriptor.registration.distribution_spec.policy
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "rnn",
        "rnn",
        hashlib.sha256(b"rnn").hexdigest(),
        policy.digest,
        policy.execution_plan.digest,
    )
    runtime = TorchRuntimeContext(
        {"model": {"input_features": 4}},
        "rnn",
        1,
        policy.digest,
        policy.execution_plan.digest,
        identity,
        input_binding_digest="1" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    return TorchBuildContext(runtime, stage)


def test_rnn_descriptors_use_unified_torch_runtime() -> None:
    for descriptor in (LSTM_DESCRIPTOR, GRU_DESCRIPTOR):
        assert (
            descriptor.registration.distribution_spec.strategy
            is DistributionStrategy.RAY_TRAIN_TORCH
        )
        assert descriptor.registration.implementation.version == "2.0.0"


def test_lstm_and_gru_recipes_produce_fixed_window_logits() -> None:
    for descriptor, recipe in (
        (LSTM_DESCRIPTOR, LSTMRecipe()),
        (GRU_DESCRIPTOR, GRURecipe()),
    ):
        build = _context(descriptor)
        stage = build.stage
        batch_context = TorchBatchContext(
            stage, ("lag_0", "lag_1", "lag_2", "lag_3"), "label"
        )
        modules = recipe.build_modules(build)
        adapted = recipe.adapt_batch(_batch(), batch_context)
        result = recipe.training_step(modules, adapted, TorchStepContext(stage, 0, 0))
        assert tuple(result.outputs["output"].shape) == (4, 1)
        assert result.loss.normalizer == 4
        assert recipe.configure_optimizers(modules, build).optimizer is not None
