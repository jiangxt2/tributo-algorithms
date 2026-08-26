"""Tests for distributed jagged-history recommendation."""

from __future__ import annotations

import pandas as pd
import torch
from tributo.algorithms.api import DistributionStrategy, FrameworkNativePolicy
from tributo_algorithms_recsys_torch import (
    JAGGED_DESCRIPTOR,
    DistributedJaggedEmbedding,
)
from tributo_algorithms_recsys_torch.contracts import JaggedCoverageValidator
from tributo_algorithms_recsys_torch.jagged import (
    _jagged_tensors,
    _model,
    _route_sparse_keys,
)


def test_jagged_descriptor_uses_framework_native_alltoall() -> None:
    distribution = JAGGED_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    assert isinstance(distribution.policy, FrameworkNativePolicy)
    assert distribution.policy.framework == "pytorch-ddp-jagged-alltoall"
    assert issubclass(DistributedJaggedEmbedding, object)


def test_jagged_batch_uses_embedding_bag_offsets() -> None:
    frame = pd.DataFrame(
        {
            "user": [0, 1, 2],
            "history": [[0], [1, 2], [2, 3, 4]],
            "candidate": [1, 2, 3],
            "label": [1.0, 0.0, 1.0],
        }
    )
    user, values, offsets, candidate, labels, token_count = _jagged_tensors(
        frame,
        user_col="user",
        history_col="history",
        candidate_col="candidate",
        label_col="label",
        item_count=8,
        device=torch.device("cpu"),
    )
    model = _model(user_count=4, item_count=8, embedding_dim=3)
    output = model(user, values, offsets, candidate)
    assert output.shape == (3, 1)
    assert offsets.tolist() == [0, 1, 3]
    assert token_count == 6
    assert _route_sparse_keys(values, rank=0, world_size=1) == 6
    assert labels.shape == (3, 1)


def test_jagged_coverage_proves_alltoall_token_conservation() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {
                "input_rows": {
                    "train": 2,
                    "coverage.history_tokens": 3,
                    "coverage.routed_owned_tokens": 4,
                    "coverage.positive_pairs": 1,
                    "coverage.negative_pairs": 1,
                }
            },
            {
                "input_rows": {
                    "train": 2,
                    "coverage.history_tokens": 5,
                    "coverage.routed_owned_tokens": 4,
                    "coverage.positive_pairs": 1,
                    "coverage.negative_pairs": 1,
                }
            },
        ],
        "state": {
            "details": {
                "jagged": True,
                "routing": "all_to_all_single_owner_mod",
            }
        },
    }
    assert JaggedCoverageValidator().validate(value) == value
