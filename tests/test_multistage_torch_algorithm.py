"""Tests for finite Teacher-to-Student distillation."""

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
    DISTILLATION_DESCRIPTOR,
    DistributedDistillation,
)
from tributo_algorithms_multistage_torch.algorithm import (
    DistillationResult,
    _distillation_scaling_config,
    _model,
    export_result,
)
from tributo_algorithms_multistage_torch.contracts import (
    DistillationCoverageValidator,
)


def test_distillation_descriptor_declares_two_component_stages() -> None:
    distribution = DISTILLATION_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    assert isinstance(distribution.policy, FrameworkNativePolicy)
    assert distribution.policy.component_stages == ("teacher", "student")
    assert (
        DISTILLATION_DESCRIPTOR.registration.implementation.flavor_id
        == "onnx-runtime-v1"
    )
    assert issubclass(DistributedDistillation, object)


def test_distillation_scaling_config_spreads_workers() -> None:
    plan = cast(
        Any,
        SimpleNamespace(
            runtime=SimpleNamespace(
                worker_count=2,
                num_cpus=1.0,
                num_gpus=0.0,
                custom_resources={},
            )
        ),
    )
    scaling = cast(Any, _distillation_scaling_config(plan))
    assert scaling.num_workers == 2
    assert scaling.placement_strategy == "SPREAD"


def test_teacher_and_student_models_have_distinct_capacity() -> None:
    teacher = _model(2, 8)
    student = _model(2, 3)
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert teacher(values).shape == (2, 1)
    assert student(values).shape == (2, 1)
    assert sum(item.numel() for item in teacher.parameters()) > sum(
        item.numel() for item in student.parameters()
    )


def test_distillation_coverage_requires_both_stages() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "state": {
            "details": {
                "component_stages": "teacher,student",
                "stage.teacher.rows": 16,
                "stage.student.rows": 16,
            }
        },
    }
    assert DistillationCoverageValidator().validate(value) == value


def test_distillation_export_publishes_student_onnx_inference(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    model = cast(torch.nn.Module, _model(2, 3))
    torch.save(model.state_dict(), checkpoint_root / "model.pt")
    (checkpoint_root / "model_config.json").write_text(
        json.dumps(
            {
                "input_features": 2,
                "teacher_hidden": 8,
                "student_hidden": 3,
                "feature_names": ["x0", "x1"],
            }
        ),
        encoding="utf-8",
    )
    execution = export_result(
        result=DistillationResult(
            checkpoint=Checkpoint.from_directory(str(checkpoint_root)),
            metrics={"loss": 0.1},
            stages={},
            composition_digest="a" * 64,
        ),
        checkpoint=Checkpoint.from_directory(str(checkpoint_root)),
        plan=cast(
            Any,
            SimpleNamespace(
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}},
                resolution=SimpleNamespace(
                    implementation_id="tributo.official.multistage_torch.distillation"
                ),
            ),
        ),
        run_id="distillation-export-test",
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
