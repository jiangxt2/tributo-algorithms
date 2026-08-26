"""Tests for the official distributed Two-Tower recommender."""

from __future__ import annotations

import torch
from tributo.algorithms import TrainingRecipeV2
from tributo.algorithms.api import DistributionStrategy
from tributo_algorithms_recsys_torch import TWO_TOWER_DESCRIPTOR, TwoTowerRecipe
from tributo_algorithms_recsys_torch.contracts import PairCoverageValidator


def test_two_tower_descriptor_uses_recipe_v2() -> None:
    registration = TWO_TOWER_DESCRIPTOR.registration
    assert registration.contract_bindings is not None
    assert registration.distribution_spec is not None
    assert (
        registration.distribution_spec.strategy
        is DistributionStrategy.RAY_TRAIN_RECIPE_V2
    )


def test_two_tower_recipe_trains_embedding_pairs() -> None:
    recipe = TwoTowerRecipe()
    assert isinstance(recipe, TrainingRecipeV2)
    modules = recipe.build_modules(
        {"model": {"user_count": 4, "item_count": 5, "embedding_dim": 3}}
    )
    batch = {
        "user_id": torch.tensor([0, 1, 2, 3]),
        "item_id": torch.tensor([1, 2, 3, 4]),
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }
    features, targets, weights, rows = recipe.batch_adapter(
        batch,
        feature_names=("user_id", "item_id"),
        label_name="label",
        weight_name=None,
        config={},
    )
    result = recipe.training_step(modules, features, targets, weights, {})
    result.loss.backward()

    assert rows == 4
    assert result.predictions.shape == (4, 1)
    assert result.coverage_counts == {
        "positive_pairs": 2,
        "negative_pairs": 2,
    }


def test_pair_coverage_contract_partitions_interactions() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {
                "input_rows": {
                    "train": 4,
                    "coverage.positive_pairs": 2,
                    "coverage.negative_pairs": 2,
                }
            },
            {
                "input_rows": {
                    "train": 4,
                    "coverage.positive_pairs": 2,
                    "coverage.negative_pairs": 2,
                }
            },
        ],
    }
    assert PairCoverageValidator().validate(value) == value
