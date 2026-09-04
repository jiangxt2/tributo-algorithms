"""Tests for the typed temporal convolution TorchRecipe."""

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
from tributo_algorithms_timeseries import TEMPORAL_CONV_DESCRIPTOR
from tributo_algorithms_timeseries.recipe import TemporalConvRecipe


def _contexts(
    optimizer: dict[str, object] | None = None,
) -> tuple[TorchBuildContext, TorchBatchContext, TorchStepContext]:
    policy = TEMPORAL_CONV_DESCRIPTOR.registration.distribution_spec.policy
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "tcn",
        "tcn",
        hashlib.sha256(b"tcn").hexdigest(),
        policy.digest,
        policy.execution_plan.digest,
    )
    runtime = TorchRuntimeContext(
        {
            "model": {"input_features": 4},
            "optimizer": (
                {"accumulation_steps": 2} if optimizer is None else optimizer
            ),
        },
        "tcn",
        1,
        policy.digest,
        policy.execution_plan.digest,
        identity,
        input_binding_digest="1" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    return (
        TorchBuildContext(runtime, stage),
        TorchBatchContext(
            stage, ("lag_3", "lag_2", "lag_1", "lag_0"), label_name="label"
        ),
        TorchStepContext(stage, 0, 0),
    )


def test_descriptor_uses_unified_torch_runtime_and_contracts() -> None:
    registration = TEMPORAL_CONV_DESCRIPTOR.registration
    assert (
        registration.distribution_spec.strategy is DistributionStrategy.RAY_TRAIN_TORCH
    )
    assert registration.implementation.runtime_id == "tributo.ray_train_torch"
    assert registration.implementation.version == "2.0.0"
    assert registration.contract_bindings is not None


def test_temporal_recipe_returns_sum_count_loss() -> None:
    recipe = TemporalConvRecipe()
    build, batch_context, step_context = _contexts()
    modules = recipe.build_modules(build)
    batch = {
        "lag_3": torch.tensor([-2.0, 0.5]),
        "lag_2": torch.tensor([-1.0, 1.0]),
        "lag_1": torch.tensor([-0.5, 1.5]),
        "lag_0": torch.tensor([-0.1, 2.0]),
        "label": torch.tensor([0.0, 1.0]),
    }
    adapted = recipe.adapt_batch(batch, batch_context)
    result = recipe.training_step(modules, adapted, step_context)
    assert result.outputs["output"].shape == (2, 1)
    assert result.loss.normalizer == 2
    assert result.metrics["accuracy"].normalizer == 2
    assert recipe.configure_optimizers(modules, build).gradient_accumulation_steps == 2


@pytest.mark.parametrize(
    ("optimizer", "message"),
    [
        ({"accumulation_steps": 1.5}, "gradient_accumulation_steps"),
        ({"max_gradient_norm": True}, "max_gradient_norm"),
    ],
)
def test_temporal_recipe_preserves_optimizer_value_types(
    optimizer: dict[str, object], message: str
) -> None:
    recipe = TemporalConvRecipe()
    build, _, _ = _contexts(optimizer)
    modules = recipe.build_modules(build)
    with pytest.raises(ValueError, match=message):
        recipe.configure_optimizers(modules, build)
