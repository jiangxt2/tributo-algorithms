"""Tests for distributed self-supervised pretraining and finetuning."""

from __future__ import annotations

import torch
from tributo.algorithms.api import DistributionStrategy, FrameworkNativePolicy
from tributo_algorithms_multistage_torch import (
    PRETRAIN_FINETUNE_DESCRIPTOR,
    DistributedPretrainFinetune,
)
from tributo_algorithms_multistage_torch.contracts import (
    PretrainFinetuneCoverageValidator,
)
from tributo_algorithms_multistage_torch.pretrain import (
    _finetune_model,
    _pretrain_model,
)


def test_pretrain_finetune_descriptor_declares_two_stages() -> None:
    distribution = PRETRAIN_FINETUNE_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    assert isinstance(distribution.policy, FrameworkNativePolicy)
    assert distribution.policy.component_stages == ("pretrain", "finetune")
    assert issubclass(DistributedPretrainFinetune, object)


def test_pretraining_reconstructs_and_finetuning_classifies() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pretrain = _pretrain_model(2, 3)
    finetune = _finetune_model(2, 3)
    assert pretrain(features).shape == (2, 2)
    assert finetune(features).shape == (2, 1)


def test_pretrain_finetune_coverage_requires_both_stages() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "state": {
            "details": {
                "component_stages": "pretrain,finetune",
                "stage.pretrain.rows": 16,
                "stage.finetune.rows": 16,
            }
        },
    }
    assert PretrainFinetuneCoverageValidator().validate(value) == value
