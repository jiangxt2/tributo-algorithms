"""Tests for distributed jagged-history recommendation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from ray.train import Checkpoint
from tributo.algorithms.api import DistributionStrategy, FrameworkNativePolicy
from tributo.exporting.runtime import BundleModelLoader
from tributo_algorithms_recsys_torch import (
    JAGGED_DESCRIPTOR,
    DistributedJaggedEmbedding,
)
from tributo_algorithms_recsys_torch.contracts import (
    JaggedConfigValidator,
    JaggedCoverageValidator,
)
from tributo_algorithms_recsys_torch.jagged import (
    JaggedResult,
    _jagged_tensors,
    _model,
    _padded_inference_model,
    _route_sparse_keys,
    export_jagged_result,
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
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "from tributo_algorithms_recsys_torch.jagged import _model; "
                "model = _model(user_count=4, item_count=8, embedding_dim=3); "
                "output = model("
                "torch.tensor([0, 1, 2]), "
                "torch.tensor([0, 1, 2, 2, 3, 4]), "
                "torch.tensor([0, 1, 3]), "
                "torch.tensor([1, 2, 3])"
                "); "
                "assert output.shape == (3, 1)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert offsets.tolist() == [0, 1, 3]
    assert token_count == 6
    assert _route_sparse_keys(values, rank=0, world_size=1) == 6
    assert labels.shape == (3, 1)


def test_jagged_config_requires_explicit_inference_history_width() -> None:
    value = {
        "data": {
            "user_col": "user_id",
            "history_col": "history",
            "candidate_col": "item_id",
            "label_col": "label",
            "inference_history_width": 3,
        },
        "model": {"user_count": 4, "item_count": 8, "embedding_dim": 3},
        "output": {"bundle_uri": "/tmp/bundle"},
        "ray": {"storage_path": "/tmp/ray"},
        "training": {},
    }
    assert JaggedConfigValidator().validate(value) == value
    invalid = {**value, "data": {**value["data"], "inference_history_width": 0}}
    with pytest.raises(ValueError, match="inference_history_width must be positive"):
        JaggedConfigValidator().validate(invalid)


def test_padded_jagged_inference_is_exact_and_fail_closed() -> None:
    model = _model(user_count=4, item_count=8, embedding_dim=3)
    padded = cast(Any, _padded_inference_model(model))
    users = torch.tensor([0, 1], dtype=torch.int64)
    histories = torch.tensor([[0, -1, -1], [1, 2, -1]], dtype=torch.int64)
    candidates = torch.tensor([1, 2], dtype=torch.int64)
    base = cast(Any, model)
    expected_history = torch.stack(
        (
            base.history_embedding.weight[0],
            base.history_embedding.weight[[1, 2]].mean(dim=0),
        )
    )
    expected = (
        (base.user_embedding(users) + expected_history)
        * base.candidate_embedding(candidates)
    ).sum(dim=1, keepdim=True) + base.bias
    output = padded(users, histories, candidates)
    torch.testing.assert_close(output[:, :1], expected)
    torch.testing.assert_close(output[:, 1], torch.tensor([1.0, 1.0]))

    invalid_output = padded(
        torch.tensor([-1, 4, 0]),
        torch.tensor([[0, -1, -1], [1, 2, -1], [8, -1, -1]]),
        torch.tensor([1, 2, 1]),
    )
    torch.testing.assert_close(invalid_output, torch.zeros((3, 2)))


def test_jagged_export_publishes_typed_padded_inference(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    model = cast(torch.nn.Module, _model(user_count=4, item_count=8, embedding_dim=3))
    torch.save(model.state_dict(), checkpoint_root / "model.pt")
    (checkpoint_root / "model_config.json").write_text(
        json.dumps(
            {
                "model": {"user_count": 4, "item_count": 8, "embedding_dim": 3},
                "data": {
                    "user_col": "user_id",
                    "history_col": "item_history",
                    "candidate_col": "item_id",
                    "label_col": "label",
                    "inference_history_width": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    execution = export_jagged_result(
        result=JaggedResult(
            checkpoint=Checkpoint.from_directory(str(checkpoint_root)),
            metrics={"loss": 0.1},
            evidence={},
        ),
        checkpoint=Checkpoint.from_directory(str(checkpoint_root)),
        plan=cast(
            Any,
            SimpleNamespace(
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}},
                primary_input_binding=SimpleNamespace(
                    feature_names=("user_id", "item_history", "item_id")
                ),
                resolution=SimpleNamespace(
                    implementation_id="tributo.official.recsys_torch.jagged_embedding"
                ),
            ),
        ),
        run_id="jagged-export-test",
    )
    runtime = BundleModelLoader().open(
        cast(str, execution.outputs["bundle_uri"]),
        role="inference",
        use_case="batch",
    )
    try:
        outputs = runtime.predict(
            {
                "user_id": np.asarray([0, -1], dtype=np.int64),
                "item_history": np.asarray([[0, -1, -1], [1, 2, -1]], dtype=np.int64),
                "item_id": np.asarray([1, 2], dtype=np.int64),
            }
        )
    finally:
        runtime.close()
    assert outputs["output"].shape == (2, 2)
    np.testing.assert_array_equal(outputs["output"][:, 1], [1.0, 0.0])
    np.testing.assert_allclose(outputs["output"][1], 0.0)


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
