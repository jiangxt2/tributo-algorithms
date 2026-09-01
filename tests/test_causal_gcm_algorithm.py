"""Tests for official distributed DoWhy GCM root-cause analysis."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from tributo.algorithms.api import DistributionStrategy, FrameworkNativePolicy
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.runtime import BundleModelLoader
from tributo_algorithms_causal_dowhy import (
    GCM_DESCRIPTOR,
    DistributedGCMRootCause,
)
from tributo_algorithms_causal_dowhy.contracts import (
    GCMConfigValidator,
    GCMCoverageValidator,
    GCMInputValidator,
)
from tributo_algorithms_causal_dowhy.gcm import (
    GCMRootCauseResult,
    _report_query_onnx,
    export_gcm_result,
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
    assert GCM_DESCRIPTOR.registration.implementation.flavor_id == "onnx-runtime-v1"
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


def test_gcm_report_query_is_exact_and_marks_invalid_nodes() -> None:
    import onnxruntime as ort

    report = {
        "nodes": ["x", "y"],
        "root_cause_attribution": {
            "x": {"mean_signed": -0.25, "mean_absolute": 0.5},
            "y": {"mean_signed": 0.75, "mean_absolute": 1.0},
        },
    }
    session = ort.InferenceSession(
        _report_query_onnx(report),
        providers=["CPUExecutionProvider"],
    )
    signed, absolute, valid = session.run(
        None,
        {"node_id": np.asarray([0, 1, -1, 2], dtype=np.int64)},
    )
    np.testing.assert_allclose(signed, [-0.25, 0.75, 0.0, 0.0])
    np.testing.assert_allclose(absolute, [0.5, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(valid, [True, True, False, False])


def test_gcm_export_publishes_exact_query_and_report_roles(tmp_path: Path) -> None:
    report = {
        "nodes": ["x", "y"],
        "root_cause_attribution": {
            "x": {"mean_signed": -0.25, "mean_absolute": 0.5},
            "y": {"mean_signed": 0.75, "mean_absolute": 1.0},
        },
        "counterfactual_target_delta": -0.4,
        "counterfactual_target_absolute_delta": 0.4,
        "limitations": ["report query only"],
    }
    execution = export_gcm_result(
        result=GCMRootCauseResult(
            report=report,
            stage_evidence={},
            composition_digest="c" * 64,
        ),
        checkpoint=object(),
        plan=cast(
            Any,
            SimpleNamespace(
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}},
                resolution=SimpleNamespace(
                    implementation_id="tributo.official.causal_dowhy.gcm_root_cause"
                ),
            ),
        ),
        run_id="gcm-export-test",
    )
    bundle_uri = cast(str, execution.outputs["bundle_uri"])
    manifest = BundleReader().read_manifest(bundle_uri)
    assert set(manifest.roles) == {"inference", "report"}
    runtime = BundleModelLoader().open(bundle_uri, role="inference", use_case="batch")
    try:
        outputs = runtime.predict({"node_id": np.asarray([0, -1], dtype=np.int64)})
    finally:
        runtime.close()
    np.testing.assert_array_equal(outputs["valid"], [True, False])
