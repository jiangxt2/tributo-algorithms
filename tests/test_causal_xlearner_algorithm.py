"""Tests for official distributed X-Learner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tributo.algorithms.api import AlgorithmExecutionError
from tributo.exporting.models import (
    ExportContext,
    ExportSource,
    ExportTarget,
    PlannedTarget,
)
from tributo_algorithms_causal_xlearner import X_LEARNER_DESCRIPTOR
from tributo_algorithms_causal_xlearner.algorithm import (
    XLearnerResult,
    _validate_treatment_batch,
)
from tributo_algorithms_causal_xlearner.contracts import XLearnerConfigValidator
from tributo_algorithms_causal_xlearner.exporter import XLearnerONNXExporter
from tributo_algorithms_causal_xlearner.model import (
    STAGE_OBJECTIVES,
    STAGES,
    XLearnerModel,
)


def test_xlearner_owns_stable_algorithm_spec_and_stages() -> None:
    assert X_LEARNER_DESCRIPTOR.name == "x_learner"
    assert X_LEARNER_DESCRIPTOR.package_name == ("tributo-algorithms")
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


@pytest.mark.parametrize(
    "propensity_graph", ["valid", "missing_probability", "wrong_operator"]
)
def test_xlearner_onnx_export_round_trip_uses_opset_ir(
    tmp_path: Path, propensity_graph: str
) -> None:
    # Isolate the native XGBoost/ONNX runtime from other framework tests.
    script = """
import runpy
import sys
from pathlib import Path
import pytest

check = runpy.run_path(sys.argv[1])["_check_xlearner_onnx_export"]
with pytest.MonkeyPatch.context() as patch:
    check(Path(sys.argv[2]), patch, sys.argv[3])
"""
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-W",
            "ignore::DeprecationWarning",
            "-c",
            script,
            str(Path(__file__).resolve()),
            str(tmp_path),
            propensity_graph,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_xlearner_onnx_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, propensity_graph: str
) -> None:
    import onnx
    import onnxmltools
    import onnxruntime as ort
    import xgboost

    monkeypatch.setattr(onnx, "IR_VERSION", 999)
    if propensity_graph != "valid":
        original_convert = onnxmltools.convert_xgboost

        def convert_with_invalid_propensity(booster: Any, **kwargs: Any) -> Any:
            model = original_convert(booster, **kwargs)
            node = model.graph.node[0]
            if node.op_type == "TreeEnsembleClassifier":
                if propensity_graph == "missing_probability":
                    del node.output[1:]
                else:
                    node.op_type = "Identity"
            return model

        monkeypatch.setattr(
            onnxmltools, "convert_xgboost", convert_with_invalid_propensity
        )
    features = np.asarray(
        [[-2.0, -1.0], [-1.0, 0.5], [-0.5, 1.0], [0.5, -1.0], [1.0, 0.5], [2.0, 1.0]],
        dtype=np.float32,
    )
    boosters = {}
    for index, stage in enumerate(STAGES):
        labels = (
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32)
            if stage == "propensity"
            else (index + 1) * features[:, 0] + 0.25 * features[:, 1]
        )
        boosters[stage] = xgboost.train(
            {
                "objective": STAGE_OBJECTIVES[stage],
                "base_score": 0.5,
                "max_depth": 2,
                "min_child_weight": 0,
                "eta": 0.5,
                "nthread": 1,
                "seed": 7,
            },
            xgboost.DMatrix(features, label=labels),
            num_boost_round=4,
        )
    result = XLearnerResult(
        booster_raw={
            stage: bytes(booster.save_raw(raw_format="ubj"))
            for stage, booster in boosters.items()
        },
        stage_evidence={},
        feature_names=("x0", "x1"),
        response_threshold=0.5,
        propensity_clip=(0.4, 0.6),
        metrics={},
        composition_digest="test-composition",
    )
    context = ExportContext(
        execution_id="xlearner-round-trip",
        node_id="cate",
        artifact_dir=tmp_path,
    )
    source = ExportSource(source_kind="official_x_learner", model_object=result)
    target = PlannedTarget(
        target=ExportTarget(name="cate", format="onnx"),
        exporter_id=XLearnerONNXExporter.exporter_id,
    )
    exporter = XLearnerONNXExporter()
    if propensity_graph != "valid":
        with pytest.raises(AlgorithmExecutionError, match="label and probability"):
            exporter.export(context, source, {}, target)
        assert not (tmp_path / "model.onnx").exists()
        return
    draft = exporter.export(context, source, {}, target)
    path = tmp_path / draft.entrypoint
    model = onnx.load(path)
    assert model.ir_version == onnx.helper.find_min_ir_version_for(model.opset_import)
    assert {node.name for node in model.graph.node} >= set(STAGES)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    native = XLearnerModel.from_raw(
        result.booster_raw,
        feature_names=result.feature_names,
        response_threshold=result.response_threshold,
        propensity_clip=result.propensity_clip,
    )
    raw_propensity = boosters["propensity"].predict(xgboost.DMatrix(features))
    assert np.any(raw_propensity < result.propensity_clip[0])
    assert np.any(raw_propensity > result.propensity_clip[1])
    for batch_size in (1, 3, len(features)):
        batch = features[:batch_size]
        actual = session.run(["cate"], {"float_input": batch})[0]
        assert actual.shape == (batch_size, 1)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(
            actual[:, 0], native.predict(batch).cate, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, "0", "1"])
def test_xlearner_treatment_validation_normalizes_binary_values(value: object) -> None:
    result = _validate_treatment_batch(
        pd.DataFrame({"treatment": [value]}), treatment_name="treatment"
    )

    assert result["treatment"].tolist() == [int(value)]
    assert result["treatment"].dtype == np.dtype("int8")


@pytest.mark.parametrize(
    "value",
    [0.5, 1.5, -1, 2, np.nan, np.inf, -np.inf, None, "invalid"],
)
def test_xlearner_treatment_validation_rejects_non_binary_or_non_finite_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="treatment values"):
        _validate_treatment_batch(
            pd.DataFrame({"treatment": [value]}), treatment_name="treatment"
        )


def test_xlearner_treatment_validation_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="column.*missing"):
        _validate_treatment_batch(
            pd.DataFrame({"outcome": [1.0]}), treatment_name="treatment"
        )
