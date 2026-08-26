"""Tests for official dense DNN and PU TrainingRecipeV2 implementations."""

from __future__ import annotations

import pytest
import torch
from tributo.algorithms import TrainingRecipeV2
from tributo.algorithms.api import DistributionStrategy
from tributo_algorithms_tabular_torch import (
    DNN_DESCRIPTOR,
    PU_DESCRIPTOR,
    DNNRecipe,
    PURecipe,
)
from tributo_algorithms_tabular_torch.contracts import (
    PUConfigValidator,
    PUCoverageValidator,
)


@pytest.mark.parametrize("descriptor", [DNN_DESCRIPTOR, PU_DESCRIPTOR])
def test_descriptor_uses_recipe_v2(descriptor: object) -> None:
    registration = descriptor.registration
    assert registration.contract_bindings is not None
    assert registration.distribution_spec is not None
    assert (
        registration.distribution_spec.strategy
        is DistributionStrategy.RAY_TRAIN_RECIPE_V2
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "x0": torch.tensor([-2.0, -1.0, 1.0, 2.0]),
        "x1": torch.tensor([-1.0, -0.5, 0.5, 1.0]),
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }


def test_dnn_recipe_produces_scalar_supervised_loss() -> None:
    recipe = DNNRecipe()
    assert isinstance(recipe, TrainingRecipeV2)
    modules = recipe.build_modules(
        {"model": {"input_features": 2, "hidden_units": [4]}}
    )
    features, targets, weights, rows = recipe.batch_adapter(
        _batch(),
        feature_names=("x0", "x1"),
        label_name="label",
        weight_name=None,
        config={},
    )
    result = recipe.training_step(modules, features, targets, weights, {})
    assert result.predictions.shape == (4, 1)
    assert result.loss.ndim == 0
    assert rows == 4
    assert recipe.metric_plan({}).factories.keys() == {"accuracy"}


@pytest.mark.parametrize("loss_type", ["nnpu", "upu"])
def test_pu_recipe_preserves_positive_unlabeled_risk(loss_type: str) -> None:
    recipe = PURecipe()
    modules = recipe.build_modules(
        {
            "model": {"input_features": 2, "hidden_units": [4]},
            "loss": {"type": loss_type, "class_prior": 0.4},
        }
    )
    features, targets, weights, _ = recipe.batch_adapter(
        _batch(),
        feature_names=("x0", "x1"),
        label_name="label",
        weight_name=None,
        config={},
    )
    result = recipe.training_step(modules, features, targets, weights, {})
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert result.coverage_counts == {"positive": 2, "unlabeled": 2}
    assert recipe.metric_plan({}).factories.keys() == {"observed_positive_recall"}


def test_pu_contract_rejects_missing_class_prior() -> None:
    with pytest.raises(ValueError, match="class_prior"):
        PUConfigValidator().validate(
            {
                "loss": {"type": "nnpu"},
                "output": {"bundle_uri": "/tmp/unused"},
            }
        )


def test_pu_recipe_reconstructs_model_without_training_loss_config() -> None:
    modules = PURecipe().build_modules(
        {"model": {"input_features": 2, "hidden_units": [4]}}
    )
    assert isinstance(modules["model"], torch.nn.Module)


def test_pu_coverage_contract_proves_group_partition() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {
                "input_rows": {
                    "train": 4,
                    "coverage.positive": 1,
                    "coverage.unlabeled": 3,
                }
            },
            {
                "input_rows": {
                    "train": 4,
                    "coverage.positive": 3,
                    "coverage.unlabeled": 1,
                }
            },
        ],
    }
    assert PUCoverageValidator().validate(value) == value
