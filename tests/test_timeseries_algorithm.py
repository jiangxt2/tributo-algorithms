"""Tests for the official TrainingRecipeV2 time-series implementation."""

from __future__ import annotations

import torch
from tributo.algorithms import TrainingRecipeV2, TrainingStepResult
from tributo.algorithms.api import DistributionStrategy
from tributo_algorithms_timeseries import TEMPORAL_CONV_DESCRIPTOR
from tributo_algorithms_timeseries.recipe import TemporalConvRecipe


def test_descriptor_uses_recipe_v2_runtime_and_contracts() -> None:
    registration = TEMPORAL_CONV_DESCRIPTOR.registration

    assert TEMPORAL_CONV_DESCRIPTOR.api_version == 2
    assert registration.contract_bindings is not None
    assert registration.distribution_spec is not None
    assert (
        registration.distribution_spec.strategy
        is DistributionStrategy.RAY_TRAIN_RECIPE_V2
    )
    assert registration.implementation.runtime_id == "tributo.ray_train_recipe_v2"


def test_temporal_recipe_implements_only_public_recipe_v2_math() -> None:
    recipe = TemporalConvRecipe()
    assert isinstance(recipe, TrainingRecipeV2)
    modules = recipe.build_modules({"model": {"input_features": 4, "channels": 4}})
    batch = {
        "lag_3": torch.tensor([-2.0, 0.5]),
        "lag_2": torch.tensor([-1.0, 1.0]),
        "lag_1": torch.tensor([-0.5, 1.5]),
        "lag_0": torch.tensor([-0.1, 2.0]),
        "label": torch.tensor([0.0, 1.0]),
    }
    features, targets, weights, rows = recipe.batch_adapter(
        batch,
        feature_names=("lag_3", "lag_2", "lag_1", "lag_0"),
        label_name="label",
        weight_name=None,
        config={},
    )
    result = recipe.training_step(modules, features, targets, weights, {})
    plan = recipe.optimization_plan(
        modules["model"],
        {"learning_rate": 0.01, "accumulation_steps": 2},
    )

    assert isinstance(result, TrainingStepResult)
    assert result.predictions.shape == (2, 1)
    assert result.loss.ndim == 0
    assert rows == 2
    assert plan.gradient_accumulation_steps == 2
    assert recipe.metric_plan({}).factories.keys() == {"accuracy"}
