"""Tests for finite Teacher-to-Student distillation."""

from __future__ import annotations

import torch
from tributo.algorithms.api import DistributionStrategy, FrameworkNativePolicy
from tributo_algorithms_multistage_torch import (
    DISTILLATION_DESCRIPTOR,
    DistributedDistillation,
)
from tributo_algorithms_multistage_torch.algorithm import _model
from tributo_algorithms_multistage_torch.contracts import (
    DistillationCoverageValidator,
)


def test_distillation_descriptor_declares_two_component_stages() -> None:
    distribution = DISTILLATION_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    assert isinstance(distribution.policy, FrameworkNativePolicy)
    assert distribution.policy.component_stages == ("teacher", "student")
    assert issubclass(DistributedDistillation, object)


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
