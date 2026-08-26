"""Tests for the official self-supervised Autoencoder."""

from __future__ import annotations

import torch
from tributo.algorithms import TrainingStepResult
from tributo.algorithms.api import DistributionStrategy
from tributo_algorithms_representation import TABULAR_AUTOENCODER_DESCRIPTOR
from tributo_algorithms_representation.recipe import TabularAutoencoderRecipe


def test_autoencoder_descriptor_uses_recipe_v2() -> None:
    registration = TABULAR_AUTOENCODER_DESCRIPTOR.registration
    assert registration.distribution_spec is not None
    assert (
        registration.distribution_spec.strategy
        is DistributionStrategy.RAY_TRAIN_RECIPE_V2
    )
    assert registration.contract_bindings is not None


def test_autoencoder_batch_adapter_requires_no_label() -> None:
    recipe = TabularAutoencoderRecipe()
    modules = recipe.build_modules(
        {"model": {"input_features": 3, "latent_features": 2}}
    )
    batch = {
        "f0": torch.tensor([0.0, 1.0]),
        "f1": torch.tensor([1.0, 0.0]),
        "f2": torch.tensor([0.5, 0.5]),
    }
    features, targets, weights, rows = recipe.batch_adapter(
        batch,
        feature_names=("f0", "f1", "f2"),
        label_name=None,
        weight_name=None,
        config={},
    )
    step = recipe.training_step(modules, features, targets, weights, {})

    assert isinstance(step, TrainingStepResult)
    assert features.shape == targets.shape == (2, 3)
    assert step.loss.ndim == 0
    assert rows == 2
