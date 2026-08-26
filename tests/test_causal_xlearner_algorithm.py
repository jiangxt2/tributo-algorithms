"""Tests for official distributed X-Learner."""

from __future__ import annotations

from tributo_algorithms_causal_xlearner import X_LEARNER_DESCRIPTOR
from tributo_algorithms_causal_xlearner.contracts import XLearnerConfigValidator
from tributo_algorithms_causal_xlearner.exporter import XLearnerONNXExporter
from tributo_algorithms_causal_xlearner.model import STAGES


def test_xlearner_owns_stable_algorithm_spec_and_stages() -> None:
    assert X_LEARNER_DESCRIPTOR.name == "x_learner"
    assert X_LEARNER_DESCRIPTOR.package_name == ("tributo-algorithms-causal-xlearner")
    distribution = X_LEARNER_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.policy.component_stages == STAGES


def test_xlearner_config_contract_requires_five_stage_inputs() -> None:
    value = {
        "data": {
            "feature_columns": ["x0", "x1"],
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            "identity_col": "identity",
        },
        "model": {},
        "training": {},
        "ray": {"storage_path": "/tmp/xlearner"},
        "output": {"bundle_uri": "/tmp/xlearner-bundle"},
    }
    assert XLearnerConfigValidator().validate(value) == value


def test_xlearner_onnx_exporter_uses_core_v2_and_runtime_validation() -> None:
    assert XLearnerONNXExporter.api_version == 2
    assert XLearnerONNXExporter.output_format == "onnx"
    assert XLearnerONNXExporter.output_flavor_id == "onnx-runtime-v1"
    assert {
        binding.validator_id for binding in XLearnerONNXExporter.validator_bindings
    } == {
        "structure-v1",
        "onnx-runtime-v1",
    }
