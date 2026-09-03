"""Tests for the typed Two-Tower TorchRecipe."""

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
from tributo_algorithms_recsys_torch import TWO_TOWER_DESCRIPTOR, TwoTowerRecipe
from tributo_algorithms_recsys_torch.contracts import (
    PairCoverageValidator,
    TwoTowerConfigValidator,
    TwoTowerTensorInputValidator,
)


def _context() -> TorchBuildContext:
    policy = TWO_TOWER_DESCRIPTOR.registration.distribution_spec.policy
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "two-tower",
        "two-tower",
        hashlib.sha256(b"two-tower").hexdigest(),
        policy.digest,
        policy.execution_plan.digest,
    )
    runtime = TorchRuntimeContext(
        {"model": {"user_count": 4, "item_count": 5, "embedding_dim": 3}},
        "two-tower",
        1,
        policy.digest,
        policy.execution_plan.digest,
        identity,
        input_binding_digest="1" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    return TorchBuildContext(runtime, stage)


def test_two_tower_descriptor_uses_torch_runtime() -> None:
    registration = TWO_TOWER_DESCRIPTOR.registration
    assert (
        registration.distribution_spec.strategy is DistributionStrategy.RAY_TRAIN_TORCH
    )
    assert registration.implementation.version == "2.0.0"


def test_two_tower_uses_two_named_id_tensors() -> None:
    recipe = TwoTowerRecipe()
    build = _context()
    modules = recipe.build_modules(build)
    adapted = recipe.adapt_batch(
        {
            "user_id": torch.tensor([0, 1, 2, 3]),
            "item_id": torch.tensor([1, 2, 3, 4]),
            "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        },
        TorchBatchContext(build.stage, ("user_id", "item_id"), "label"),
    )
    result = recipe.training_step(modules, adapted, TorchStepContext(build.stage, 0, 0))
    assert set(adapted.keyword) == {"user_id", "item_id"}
    assert result.outputs["output"].shape == (4, 1)
    assert result.loss.normalizer == 4
    assert result.coverage_counts["coverage.positive_pairs"] == 2


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


def test_two_tower_uses_new_typed_input_and_coverage_contracts() -> None:
    contracts = TWO_TOWER_DESCRIPTOR.registration.contract_bindings
    assert contracts is not None
    assert str(contracts.input.validator_ref).endswith(":TwoTowerTensorInputValidator")
    assert str(contracts.coverage.validator_ref).endswith(
        ":TwoTowerTorchCoverageValidator"
    )
    assert TwoTowerConfigValidator.schema_digest == "b" * 64


def test_two_tower_v2_input_rejects_sample_weights() -> None:
    value = {
        "bindings": [
            {
                "role": "train",
                "feature_names": ["user_id", "item_id"],
                "label_name": "label",
                "sample_weight_name": "weight",
            }
        ]
    }
    with pytest.raises(ValueError, match="user ID"):
        TwoTowerTensorInputValidator().validate(value)
