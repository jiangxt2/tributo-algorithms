"""Tests for official distributed X-Learner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tributo.algorithms.api import AlgorithmExecutionError
from tributo.exceptions import ModelLoadError
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
from tributo_algorithms_causal_xlearner.flavor import XLearnerFlavor
from tributo_algorithms_causal_xlearner.metrics import (
    _PREFIX_RECORD,
    _RECORD_TYPE_COLUMN,
    _SUMMARY_RECORD,
    CATE_COLUMN,
    CONTRIBUTION_COLUMN,
    IDENTITY_COLUMN,
    MU0_COLUMN,
    MU1_COLUMN,
    OUTCOME_COLUMN,
    QUADRANT_COLUMN,
    TREATMENT_COLUMN,
    _attach_stable_tie_breaker,
    _build_report,
    _OrderedMetricScan,
    _summarize_batch,
)
from tributo_algorithms_causal_xlearner.model import (
    FORMULA,
    QUADRANT_CODES,
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


def _valid_xlearner_config() -> dict[str, Any]:
    return {
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


def test_xlearner_config_contract_requires_five_stage_inputs() -> None:
    value = _valid_xlearner_config()

    assert XLearnerConfigValidator().validate(value) == value


@pytest.mark.parametrize(
    "propensity_clip",
    [
        [],
        [0.2],
        [0.2, 0.8, 0.9],
        "0.2,0.8",
        [0.0, 0.8],
        [0.2, 1.0],
        [0.8, 0.2],
        [0.5, 0.5],
        [np.nan, 0.8],
        [0.2, np.inf],
        [True, 0.8],
    ],
)
def test_xlearner_config_rejects_invalid_propensity_clip(
    propensity_clip: object,
) -> None:
    value = _valid_xlearner_config()
    value["training"]["propensity_clip"] = propensity_clip

    with pytest.raises(ValueError, match="training.propensity_clip"):
        XLearnerConfigValidator().validate(value)


@pytest.mark.parametrize("response_threshold", [np.nan, np.inf, -np.inf, True, "0.5"])
def test_xlearner_config_rejects_invalid_response_threshold(
    response_threshold: object,
) -> None:
    value = _valid_xlearner_config()
    value["training"]["response_threshold"] = response_threshold

    with pytest.raises(ValueError, match="training.response_threshold"):
        XLearnerConfigValidator().validate(value)


@pytest.mark.parametrize("response_threshold", [-1.5, 1.5])
def test_xlearner_config_accepts_finite_thresholds_outside_probability_range(
    response_threshold: float,
) -> None:
    value = _valid_xlearner_config()
    value["training"].update(
        propensity_clip=[0.2, 0.8], response_threshold=response_threshold
    )

    assert XLearnerConfigValidator().validate(value) == value


@pytest.mark.parametrize(
    "propensity_clip",
    [
        (0.0, 0.8),
        (0.2, 1.0),
        (0.8, 0.2),
        (0.5, 0.5),
        (np.nan, 0.8),
        (0.2, np.inf),
        (True, 0.8),
    ],
)
def test_xlearner_model_rejects_invalid_propensity_clip(
    propensity_clip: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="propensity_clip"):
        XLearnerModel(
            {stage: object() for stage in STAGES},
            feature_names=("x0",),
            response_threshold=0.5,
            propensity_clip=propensity_clip,
        )


@pytest.mark.parametrize("response_threshold", [np.nan, np.inf, -np.inf, True])
def test_xlearner_model_rejects_invalid_response_threshold(
    response_threshold: float,
) -> None:
    with pytest.raises(ValueError, match="response_threshold"):
        XLearnerModel(
            {stage: object() for stage in STAGES},
            feature_names=("x0",),
            response_threshold=response_threshold,
            propensity_clip=(0.2, 0.8),
        )


@pytest.mark.parametrize("response_threshold", [-1.5, 1.5])
def test_xlearner_model_accepts_finite_thresholds_outside_probability_range(
    response_threshold: float,
) -> None:
    model = XLearnerModel(
        {stage: object() for stage in STAGES},
        feature_names=("x0",),
        response_threshold=response_threshold,
        propensity_clip=(0.2, 0.8),
    )

    assert model.response_threshold == response_threshold
    assert model.propensity_clip == (0.2, 0.8)


def test_xlearner_flavor_rejects_boolean_response_threshold_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeBooster:
        def load_model(self, path: str) -> None:
            self.stage = Path(path).stem

        def save_config(self) -> str:
            return json.dumps(
                {"learner": {"objective": {"name": STAGE_OBJECTIVES[self.stage]}}}
            )

    fake_xgboost = ModuleType("xgboost")
    fake_xgboost.Booster = FakeBooster
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost)

    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "api_version": 1,
                "formula": FORMULA,
                "quadrant_codes": QUADRANT_CODES,
                "components": {stage: f"{stage}.ubj" for stage in STAGES},
                "feature_names": ["x0"],
                "response_threshold": True,
                "propensity_clip": [0.2, 0.8],
            }
        ),
        encoding="utf-8",
    )
    artifact: Any = SimpleNamespace(
        entrypoint_path=metadata_path,
        path_for=lambda name: tmp_path / name,
    )

    with pytest.raises(ModelLoadError, match="artifact is invalid"):
        XLearnerFlavor().load(artifact, role="prediction")


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


class _Booster:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def predict(self, _matrix: object) -> np.ndarray:
        return self.values


def test_xlearner_quadrants_preserve_response_uplift_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "xgboost",
        SimpleNamespace(DMatrix=lambda values, **_kwargs: values),
    )
    model = XLearnerModel(
        {
            "mu0": _Booster([0.1, 0.1, 0.8, 0.8]),
            "mu1": _Booster([0.2, 0.2, 0.9, 0.9]),
            "tau0": _Booster([-0.2, 0.2, -0.2, 0.2]),
            "tau1": _Booster([-0.2, 0.2, -0.2, 0.2]),
            "propensity": _Booster([0.5, 0.5, 0.5, 0.5]),
        },
        feature_names=("x0",),
        response_threshold=0.5,
        propensity_clip=(0.01, 0.99),
    )

    result = model.predict(np.arange(4, dtype=np.float32).reshape(-1, 1))

    assert result.quadrant.tolist() == [
        "low_response_low_uplift",
        "low_response_high_uplift",
        "high_response_low_uplift",
        "high_response_high_uplift",
    ]
    assert QUADRANT_CODES == {
        "low_response_low_uplift": 0,
        "low_response_high_uplift": 1,
        "high_response_low_uplift": 2,
        "high_response_high_uplift": 3,
    }


def test_xlearner_report_contains_complete_uplift_evaluation() -> None:
    summary: dict[str, object] = {
        "treated_outcome_sum": 1.0,
        "control_outcome_sum": 1.0,
        "positive_contribution_count": 1,
        "zero_contribution_count": 2,
        "negative_contribution_count": 1,
    }
    for name, treatment, outcome in (
        ("low_response_high_uplift", 1, 1),
        ("high_response_high_uplift", 0, 0),
        ("low_response_low_uplift", 1, 0),
        ("high_response_low_uplift", 0, 1),
    ):
        summary[f"quadrant.{name}.count"] = 1
        summary[f"quadrant.{name}.treated_count"] = treatment
        summary[f"quadrant.{name}.control_count"] = 1 - treatment
        summary[f"quadrant.{name}.treated_outcome_sum"] = outcome if treatment else 0
        summary[f"quadrant.{name}.control_outcome_sum"] = (
            outcome if not treatment else 0
        )
    prefixes = {
        1: {
            "gain": 1.0,
            "cate_sum": 0.4,
            "treated_count": 1.0,
            "control_count": 0.0,
            "treated_outcome_sum": 1.0,
            "control_outcome_sum": 0.0,
        },
        2: {
            "gain": 1.0,
            "cate_sum": 0.6,
            "treated_count": 1.0,
            "control_count": 1.0,
            "treated_outcome_sum": 1.0,
            "control_outcome_sum": 0.0,
        },
        3: {
            "gain": 1.0,
            "cate_sum": 0.5,
            "treated_count": 2.0,
            "control_count": 1.0,
            "treated_outcome_sum": 1.0,
            "control_outcome_sum": 0.0,
        },
        4: {
            "gain": 0.0,
            "cate_sum": 0.2,
            "treated_count": 2.0,
            "control_count": 2.0,
            "treated_outcome_sum": 1.0,
            "control_outcome_sum": 1.0,
        },
    }

    report = _build_report(
        summaries=[summary],
        prefixes=prefixes,
        rows=4,
        treated_rows=2,
        control_rows=2,
        fold_count=2,
        n_points=5,
        n_bins=2,
    )

    assert report["ate"] == 0.05
    assert report["upliftCurve"]["modelGain"] == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert report["evaluation"] == {
        "method": "cross_fitted_oof",
        "sampleSize": 4,
    }
    assert report["calibration"]["points"] == [
        {"predicted": -0.2, "actual": -1.0, "count": 2, "valid": True},
        {"predicted": 0.3, "actual": 1.0, "count": 2, "valid": True},
    ]
    assert [segment["label"] for segment in report["quadrant"]["segments"]] == [
        0,
        1,
        2,
        3,
    ]
    assert [segment["type"] for segment in report["quadrant"]["segments"]] == list(
        QUADRANT_CODES
    )
    assert report["metric_version"] == "xlearner-oof-normalized-v1"
    assert report["metric_status"]["auuc"] == "undefined_zero_normalizer"


def _metric_row(
    *,
    identity: int,
    cate: float,
    treatment: int,
    outcome: float,
    contribution: float,
    quadrant: str,
) -> dict[str, object]:
    return {
        IDENTITY_COLUMN: identity,
        TREATMENT_COLUMN: treatment,
        OUTCOME_COLUMN: outcome,
        MU0_COLUMN: 0.1,
        MU1_COLUMN: 0.9,
        CATE_COLUMN: cate,
        QUADRANT_COLUMN: quadrant,
        CONTRIBUTION_COLUMN: contribution,
    }


def test_oof_summary_identifies_continuous_outcomes_for_binary_reference() -> None:
    frame = pd.DataFrame(
        [
            _metric_row(
                identity=1,
                cate=0.1,
                treatment=1,
                outcome=1.5,
                contribution=1.5,
                quadrant="low_response_high_uplift",
            ),
            _metric_row(
                identity=2,
                cate=-0.1,
                treatment=0,
                outcome=0.0,
                contribution=0.0,
                quadrant="low_response_low_uplift",
            ),
        ]
    )
    assert _summarize_batch(frame)["non_binary_outcome_count"] == 1


def test_xlearner_tie_breaker_is_stable_across_input_order() -> None:
    import pandas as pd

    rows = [
        _metric_row(
            identity=7,
            cate=0.5,
            treatment=0,
            outcome=0.0,
            contribution=0.0,
            quadrant="low_response_low_uplift",
        ),
        _metric_row(
            identity=7,
            cate=0.5,
            treatment=1,
            outcome=1.0,
            contribution=1.0,
            quadrant="low_response_high_uplift",
        ),
        _metric_row(
            identity=7,
            cate=0.5,
            treatment=0,
            outcome=1.0,
            contribution=-1.0,
            quadrant="high_response_low_uplift",
        ),
    ]
    forward = _attach_stable_tie_breaker(pd.DataFrame(rows))
    reverse = _attach_stable_tie_breaker(pd.DataFrame(list(reversed(rows))))
    sort_columns = [
        CATE_COLUMN,
        IDENTITY_COLUMN,
        "__tributo_xlearner_tie_breaker_high",
        "__tributo_xlearner_tie_breaker_low",
    ]

    forward = forward.sort_values(sort_columns).reset_index(drop=True)
    reverse = reverse.sort_values(sort_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(forward, reverse)


def test_xlearner_ordered_scan_is_linear_for_many_identical_physical_blocks() -> None:
    import pandas as pd

    block_rows = 32
    block_count = 128
    total_rows = block_rows * block_count
    frame = pd.DataFrame(
        [
            _metric_row(
                identity=1,
                cate=0.5,
                treatment=1,
                outcome=1.0,
                contribution=1.0,
                quadrant="low_response_high_uplift",
            ),
        ]
        * block_rows
    )
    scan = _OrderedMetricScan(target_ranks=list(range(1, total_rows + 1)))
    outputs = [scan(frame) for _ in range(block_count)]

    emitted_rows = sum(len(output) for output in outputs)
    prefix_ranks = [
        int(record["rank"])
        for output in outputs
        for record in output.to_dict("records")
        if record[_RECORD_TYPE_COLUMN] == _PREFIX_RECORD
    ]
    summary_count = sum(
        int((output[_RECORD_TYPE_COLUMN] == _SUMMARY_RECORD).sum())
        for output in outputs
    )
    assert emitted_rows == total_rows + block_count
    assert prefix_ranks == list(range(1, total_rows + 1))
    assert summary_count == block_count
    assert outputs[-1]["gain"].iloc[-1] == total_rows


def _normalization_report(
    treatment: list[int], outcome: list[float], *, batch_size: int
) -> dict[str, Any]:
    """Feed actual prefix-scan records into reporting without starting Ray."""
    rows = len(treatment)
    treated_rows = sum(treatment)
    control_rows = rows - treated_rows
    ratio = treated_rows / control_rows
    frame = pd.DataFrame(
        [
            _metric_row(
                identity=index,
                cate=float(rows - index),
                treatment=assigned,
                outcome=response,
                contribution=response if assigned else -ratio * response,
                quadrant="low_response_high_uplift",
            )
            for index, (assigned, response) in enumerate(zip(treatment, outcome))
        ]
    )
    scan = _OrderedMetricScan(target_ranks=list(range(1, rows + 1)))
    records = [
        row
        for start in range(0, rows, batch_size)
        for row in scan(frame.iloc[start : start + batch_size]).to_dict("records")
    ]
    summaries = [row for row in records if row[_RECORD_TYPE_COLUMN] == _SUMMARY_RECORD]
    prefixes = {
        int(row["rank"]): row
        for row in records
        if row[_RECORD_TYPE_COLUMN] == _PREFIX_RECORD
    }
    return _build_report(
        summaries=summaries,
        prefixes=prefixes,
        rows=rows,
        treated_rows=treated_rows,
        control_rows=control_rows,
        fold_count=2,
        n_points=rows + 1,
        n_bins=2,
    )


@pytest.mark.parametrize("batch_size", [1, 3, 32])
@pytest.mark.parametrize("treated_rows,control_rows", [(1, 3), (4, 12), (3, 1), (2, 2)])
@pytest.mark.parametrize("constant_outcome", [0.0, 1.0])
def test_oof_zero_net_gain_has_undefined_auuc_across_batch_layouts(
    batch_size: int, treated_rows: int, control_rows: int, constant_outcome: float
) -> None:
    rows = treated_rows + control_rows
    report = _normalization_report(
        [1] * treated_rows + [0] * control_rows,
        [constant_outcome] * rows,
        batch_size=batch_size,
    )
    # Every arm has the same outcome: exact net gain is zero for any ratio.
    assert report["auuc"] == 0.5
    assert report["metric_status"]["auuc"] == "undefined_zero_normalizer"
    assert report["upliftCurve"]["modelGain"][-1] == 0.0
    assert report["upliftCurve"]["randomGain"] == [0.0] * (rows + 1)


@pytest.mark.parametrize("batch_size", [1, 3, 32])
@pytest.mark.parametrize(
    "outcome,gain,auuc",
    [
        ([1.0, 0.0, 0.0, 0.0], 1.0, 0.875),
        ([0.0, 1.0, 1.0, 1.0], -1.0, 0.375),
        ([1e-20, 0.0, 0.0, 0.0], 1e-20, 0.875),
    ],
)
def test_oof_nonzero_gain_preserves_scale_and_normalized_area(
    batch_size: int, outcome: list[float], gain: float, auuc: float
) -> None:
    report = _normalization_report([1, 0, 0, 0], outcome, batch_size=batch_size)
    assert report["metric_status"]["auuc"] == "ok"
    assert report["auuc"] == auuc
    assert report["normalization"]["full_cohort_gain"] == gain
    assert report["normalization"]["zero_tolerance"] < abs(gain)


@pytest.mark.parametrize("extra_response,undefined", [(4e-16, True), (1e-12, False)])
def test_oof_zero_test_is_relative_to_the_outcome_total_scale(
    extra_response: float, undefined: bool
) -> None:
    report = _normalization_report(
        [1, 0, 0, 0], [1.0 + extra_response, 1.0, 1.0, 1.0], batch_size=4
    )
    if undefined:
        assert report["metric_status"]["auuc"] == "undefined_zero_normalizer"
        assert report["auuc"] == 0.5
        assert report["normalization"]["full_cohort_gain"] == 0.0
    else:
        assert report["metric_status"]["auuc"] == "ok"
        assert (
            report["normalization"]["full_cohort_gain"] == (1.0 + extra_response) - 1.0
        )
