"""Tests for official distributed doubly robust estimation."""

from __future__ import annotations

from tributo.algorithms.api import DistributionStrategy
from tributo_algorithms_causal_dr import DR_DESCRIPTOR, DistributedDRLearner
from tributo_algorithms_causal_dr.contracts import DRConfigValidator


def test_dr_descriptor_declares_three_framework_stages() -> None:
    distribution = DR_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    assert distribution.policy.component_stages == ("mu0", "mu1", "propensity")
    assert issubclass(DistributedDRLearner, object)


def test_dr_config_contract_requires_nuisance_inputs() -> None:
    value = {
        "data": {
            "feature_columns": ["x0", "x1"],
            "treatment_col": "treatment",
            "outcome_col": "outcome",
        },
        "model": {},
        "training": {},
        "ray": {"storage_path": "/tmp/dr"},
        "output": {"bundle_uri": "/tmp/dr-bundle"},
    }
    assert DRConfigValidator().validate(value) == value


def test_dr_config_defaults_to_five_fold_cross_fitting() -> None:
    value = {
        "data": {
            "feature_columns": ["x0", "x1"],
            "treatment_col": "treatment",
            "outcome_col": "outcome",
        },
        "model": {},
        "training": {},
        "ray": {"storage_path": "/tmp/dr"},
        "output": {"bundle_uri": "/tmp/dr-bundle"},
    }
    normalized = DRConfigValidator().validate(value)
    assert normalized["training"] == {}
