"""Tests for the official pre-tokenized Transformer classifier."""

from __future__ import annotations

import torch
from tributo.algorithms import TrainingStepResult
from tributo.algorithms.api import DistributionStrategy
from tributo_algorithms_transformers_nlp import TOKEN_TRANSFORMER_DESCRIPTOR
from tributo_algorithms_transformers_nlp.recipe import TokenTransformerRecipe


def test_transformer_descriptor_uses_recipe_v2() -> None:
    distribution = TOKEN_TRANSFORMER_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.RAY_TRAIN_RECIPE_V2
    assert TOKEN_TRANSFORMER_DESCRIPTOR.registration.contract_bindings is not None


def test_transformer_recipe_consumes_ordered_token_columns() -> None:
    recipe = TokenTransformerRecipe()
    modules = recipe.build_modules(
        {
            "model": {
                "vocab_size": 32,
                "sequence_length": 4,
                "hidden_size": 8,
                "heads": 2,
            }
        }
    )
    batch = {
        "token_0": torch.tensor([1, 4]),
        "token_1": torch.tensor([2, 5]),
        "token_2": torch.tensor([3, 0]),
        "token_3": torch.tensor([0, 0]),
        "label": torch.tensor([1.0, 0.0]),
    }
    features, targets, weights, rows = recipe.batch_adapter(
        batch,
        feature_names=("token_0", "token_1", "token_2", "token_3"),
        label_name="label",
        weight_name=None,
        config={},
    )
    step = recipe.training_step(modules, features, targets, weights, {})

    assert isinstance(step, TrainingStepResult)
    assert features.shape == (2, 4)
    assert step.predictions.shape == (2, 1)
    assert step.loss.ndim == 0
    assert rows == 2
