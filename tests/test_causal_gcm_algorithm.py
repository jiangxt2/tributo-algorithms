"""Tests for official distributed DoWhy GCM root-cause analysis."""

from __future__ import annotations

from tributo.algorithms.api import DistributionStrategy, FrameworkNativePolicy
from tributo_algorithms_causal_dowhy import (
    GCM_DESCRIPTOR,
    DistributedGCMRootCause,
)
from tributo_algorithms_causal_dowhy.contracts import (
    GCMConfigValidator,
    GCMCoverageValidator,
    GCMInputValidator,
)


def test_gcm_descriptor_declares_root_cause_stages() -> None:
    distribution = GCM_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    assert isinstance(distribution.policy, FrameworkNativePolicy)
    assert distribution.policy.component_stages == (
        "fit_gcm",
        "attribute_root_cause",
    )
    assert issubclass(DistributedGCMRootCause, object)


def test_gcm_contracts_require_causal_graph_and_two_roles() -> None:
    config = {
        "data": {
            "nodes": ["x", "y"],
            "edges": [["x", "y"]],
            "target_node": "y",
            "interventions": {"x": 0.0},
        },
        "gcm": {
            "quality": "good",
            "distribution_samples": 50,
            "shapley_permutations": 3,
        },
        "runtime": {},
        "output": {"bundle_uri": "/tmp/gcm-bundle"},
    }
    inputs = {
        "primary_role": "train",
        "bindings": [
            {"name": "train", "feature_names": ["x", "y"], "label_name": None},
            {
                "name": "anomaly",
                "feature_names": ["x", "y"],
                "label_name": None,
            },
        ],
    }
    assert GCMConfigValidator().validate(config) == config
    assert GCMInputValidator().validate(inputs) == inputs


def test_gcm_coverage_requires_anomaly_and_stage_evidence() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {"input_rows": {"train": 8, "coverage.anomaly": 2}},
            {"input_rows": {"train": 8, "coverage.anomaly": 2}},
        ],
        "state": {
            "details": {
                "component_stages": "fit_gcm,attribute_root_cause",
            }
        },
    }
    assert GCMCoverageValidator().validate(value) == value
