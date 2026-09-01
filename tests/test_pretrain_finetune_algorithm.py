"""Tests for distributed self-supervised pretraining and finetuning."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from ray.train import Checkpoint
from tributo.algorithms.api import DistributionStrategy, FrameworkNativePolicy
from tributo.exporting.runtime import BundleModelLoader
from tributo_algorithms_multistage_torch import (
    PRETRAIN_FINETUNE_DESCRIPTOR,
    DistributedPretrainFinetune,
)
from tributo_algorithms_multistage_torch.contracts import (
    PretrainFinetuneCoverageValidator,
)
from tributo_algorithms_multistage_torch.pretrain import (
    PretrainFinetuneResult,
    _finetune_model,
    _pretrain_model,
    export_pretrain_finetune_result,
)


def test_pretrain_finetune_descriptor_declares_two_stages() -> None:
    distribution = PRETRAIN_FINETUNE_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    assert isinstance(distribution.policy, FrameworkNativePolicy)
    assert distribution.policy.component_stages == ("pretrain", "finetune")
    assert (
        PRETRAIN_FINETUNE_DESCRIPTOR.registration.implementation.flavor_id
        == "onnx-runtime-v1"
    )
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


def test_pretrain_finetune_export_publishes_onnx_inference(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    model = cast(torch.nn.Module, _finetune_model(2, 4))
    torch.save(model.state_dict(), checkpoint_root / "model.pt")
    (checkpoint_root / "model_config.json").write_text(
        json.dumps(
            {
                "input_features": 2,
                "hidden_features": 4,
                "feature_names": ["x0", "x1"],
            }
        ),
        encoding="utf-8",
    )
    execution = export_pretrain_finetune_result(
        result=PretrainFinetuneResult(
            checkpoint=Checkpoint.from_directory(str(checkpoint_root)),
            metrics={"loss": 0.1},
            stages={},
            composition_digest="b" * 64,
        ),
        checkpoint=Checkpoint.from_directory(str(checkpoint_root)),
        plan=cast(
            Any,
            SimpleNamespace(
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}},
                resolution=SimpleNamespace(
                    implementation_id=(
                        "tributo.official.multistage_torch.pretrain_finetune"
                    )
                ),
            ),
        ),
        run_id="pretrain-finetune-export-test",
    )
    runtime = BundleModelLoader().open(
        cast(str, execution.outputs["bundle_uri"]),
        role="inference",
        use_case="batch",
    )
    try:
        outputs = runtime.predict(
            {"float_input": np.asarray([[0.0, 1.0]], dtype=np.float32)}
        )
    finally:
        runtime.close()
    assert outputs["output"].shape == (1, 1)
