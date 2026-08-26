"""Tests for official distributed DoWhy adapters."""

from __future__ import annotations

from tributo.algorithms.api import DistributionStrategy
from tributo_algorithms_causal_dowhy import (
    DOWHY_DESCRIPTOR,
    DistributedDoWhyRefutation,
)
from tributo_algorithms_causal_dowhy.contracts import DoWhyConfigValidator


def test_dowhy_descriptor_declares_estimate_refute_stages() -> None:
    distribution = DOWHY_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    assert distribution.policy.component_stages == ("estimate", "refute")
    assert issubclass(DistributedDoWhyRefutation, object)


def test_dowhy_config_contract_requires_causal_problem() -> None:
    value = {
        "data": {
            "common_causes": ["x0"],
            "treatment_col": "treatment",
            "outcome_col": "outcome",
        },
        "refutation": {"seed": 7},
        "runtime": {},
        "output": {"bundle_uri": "/tmp/dowhy-bundle"},
    }
    assert DoWhyConfigValidator().validate(value) == value
