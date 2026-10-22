"""Independent reference values and failure boundaries for held-out RCT metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from tributo_algorithms_causal_xlearner import evaluate_rct_predictions


def _inputs() -> dict[str, Any]:
    return {
        "prediction_ids": list(range(1, 9)),
        "cate_predictions": list(range(8, 0, -1)),
        "observation_ids": list(range(1, 9)),
        "treatment": [1, 0, 1, 0, 1, 0, 1, 0],
        "outcome": [1, 0, 1, 1, 0, 0, 0, 1],
        "assignment_probability": 0.5,
        "experiment_id": "fixed-randomized-fixture",
        "policy_coverage": 0.5,
        "n_bins": 2,
        "n_bootstrap": 200,
        "seed": 17,
    }


def test_rct_curve_policy_and_bins_match_hand_calculated_values() -> None:
    result = evaluate_rct_predictions(**_inputs())
    assert result["schema_version"] == 1
    assert result["metric_version"] == "rct-ipw-v1"
    assert result["mode"] == "rct"
    assert result["normalization"] == "none"
    assert result["metric_units"] == "outcome_probability"
    assert result["metrics"] == {
        "predicted_ate": 4.5,
        "observed_ate": 0.0,
        "ipw_ate": 0.0,
        "auuc_ipw": 0.25,
        "qini_ipw": 0.25,
    }
    assert result["curve"]["coverage"] == [index / 8 for index in range(9)]
    assert result["curve"]["cumulative_gain"] == [
        0,
        0.25,
        0.25,
        0.5,
        0.25,
        0.25,
        0.25,
        0.25,
        0,
    ]
    assert result["curve"]["random_gain"] == [0] * 9
    assert result["policy"] == {
        "rule": "top_coverage",
        "requested_coverage": 0.5,
        "threshold": None,
        "selected_rows": 4,
        "coverage": 0.5,
        "value": 0.75,
        "no_treatment_value": 0.5,
        "all_treatment_value": 0.5,
        "random_value": 0.5,
        "incremental_value": 0.25,
        "gain_over_random": 0.25,
    }
    assert [row["observed_uplift"] for row in result["bins"]] == [0.5, -0.5]
    assert [row["predicted_cate"] for row in result["bins"]] == [6.5, 2.5]
    assert [row["confidence_interval"] for row in result["bins"]] == [[0, 1], [-1, 0]]
    assert all(row["status"] == "ok" for row in result["bins"])
    json.dumps(result, allow_nan=False)


def test_known_assignment_probability_is_not_replaced_by_empirical_frequency() -> None:
    values = _inputs()
    values["assignment_probability"] = 0.25
    result = evaluate_rct_predictions(**values)
    assert result["metrics"]["observed_ate"] == 0
    assert result["metrics"]["ipw_ate"] == pytest.approx(2 / 3)
    assert result["curve"]["cumulative_gain"][-1] == pytest.approx(2 / 3)
    # The random line has area endpoint / 2; no division by net gain occurs.
    assert result["metrics"]["qini_ipw"] == pytest.approx(
        result["metrics"]["auuc_ipw"] - 1 / 3
    )


def test_rct_metrics_are_invariant_to_both_input_orders_with_cate_ties() -> None:
    values = _inputs()
    values["cate_predictions"] = [1.0] * 8
    expected = evaluate_rct_predictions(**values)
    for columns in (
        ("prediction_ids", "cate_predictions"),
        ("observation_ids", "treatment", "outcome"),
    ):
        for column in columns:
            values[column] = list(reversed(values[column]))
    assert evaluate_rct_predictions(**values) == expected
    assert expected["policy"]["value"] == 0.75  # IDs 1..4 win ties.


def test_threshold_policy_uses_inclusive_boundary_and_realized_budget() -> None:
    values = _inputs()
    values.update(policy_coverage=None, policy_threshold=6)
    result = evaluate_rct_predictions(**values)
    assert result["policy"]["selected_rows"] == 3
    assert result["policy"]["coverage"] == 3 / 8
    assert result["policy"]["value"] == 1.0
    assert result["policy"]["incremental_value"] == 0.5
    assert result["policy"]["gain_over_random"] == 0.5


@pytest.mark.parametrize("coverage,selected", [(0, 0), (1, 8), (0.26, 2)])
def test_coverage_uses_floor_and_supports_no_or_all_treatment(
    coverage: float, selected: int
) -> None:
    values = _inputs()
    values["policy_coverage"] = coverage
    result = evaluate_rct_predictions(**values)
    assert result["policy"]["selected_rows"] == selected
    assert result["policy"]["coverage"] == selected / 8
    if coverage in (0, 1):
        assert result["policy"]["gain_over_random"] == 0
        assert result["confidence_intervals"]["gain_over_random"] == [0, 0]


def test_zero_outcomes_and_negative_ranking_are_valid_results() -> None:
    values = _inputs()
    values["outcome"] = [0] * 8
    zero = evaluate_rct_predictions(**values)
    assert zero["metrics"]["auuc_ipw"] == 0
    assert zero["metrics"]["qini_ipw"] == 0
    assert zero["policy"]["value"] == 0
    assert all(interval == [0, 0] for interval in zero["confidence_intervals"].values())
    values = _inputs()
    values["cate_predictions"] = list(reversed(values["cate_predictions"]))
    negative = evaluate_rct_predictions(**values)
    assert negative["metrics"]["qini_ipw"] == -0.25
    assert negative["policy"]["gain_over_random"] == -0.25


def test_bootstrap_matches_independent_reference_with_frozen_actions() -> None:
    values = _inputs()
    result = evaluate_rct_predictions(**values)
    treated = np.asarray([0, 2, 4, 6])
    control = np.asarray([1, 3, 5, 7])
    assignment = np.asarray(values["treatment"])
    outcome = np.asarray(values["outcome"])
    action = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
    rng = np.random.default_rng(17)
    reference = []
    selected_counts = set()
    for _ in range(200):
        indices = np.r_[rng.choice(treated, 4), rng.choice(control, 4)]
        t, y, d = assignment[indices], outcome[indices], action[indices]
        selected_counts.add(int(d.sum()))
        effect = 2 * t * y - 2 * (1 - t) * y
        reference.append(
            [
                float(y[t == 1].mean() - y[t == 0].mean()),
                float(effect.mean()),
                float((2 * d * t * y + 2 * (1 - d) * (1 - t) * y).mean()),
                float((d * effect).mean()),
                float(y.mean()),
                float(((d - 0.5) * effect).mean()),
            ]
        )
    assert len(selected_counts) > 1  # Resampling does not select a new top half.
    for index, name in enumerate(
        (
            "observed_ate",
            "ipw_ate",
            "policy_value",
            "incremental_value",
            "random_value",
            "gain_over_random",
        )
    ):
        expected = np.quantile(np.asarray(reference)[:, index], [0.025, 0.975])
        assert result["confidence_intervals"][name] == pytest.approx(expected)
    assert result["uncertainty"]["policy_actions"] == "fixed_on_original_cohort"
    assert evaluate_rct_predictions(**values) == result


def test_small_or_single_arm_bins_are_retained_with_explicit_status() -> None:
    values = _inputs()
    values["n_bins"] = 8
    result = evaluate_rct_predictions(**values)
    assert len(result["bins"]) == 8
    assert all(row["status"] == "insufficient_group_rows" for row in result["bins"])
    assert all(row["observed_uplift"] is None for row in result["bins"])
    assert all(row["confidence_interval"] is None for row in result["bins"])


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("prediction_ids", [1] * 8, "duplicate"),
        ("observation_ids", [1] * 8, "duplicate"),
        ("prediction_ids", [True] + list(range(2, 9)), "IDs"),
        ("observation_ids", list(range(2, 10)), "exact coverage"),
        ("observation_ids", [str(i) for i in range(1, 9)], "consistent"),
        ("prediction_ids", ["one"] + list(range(2, 9)), "consistent"),
        ("cate_predictions", [0] * 7, "lengths"),
        ("outcome", [0] * 7, "lengths"),
        ("treatment", [0] * 7, "lengths"),
        ("cate_predictions", [float("nan")] * 8, "finite"),
        ("cate_predictions", [complex(1, 1)] * 8, "real"),
        ("treatment", [2] * 8, "binary"),
        ("treatment", [0] * 8, "each treatment arm"),
        ("treatment", [1] + [0] * 7, "each treatment arm"),
        ("outcome", [0.5] * 8, "binary"),
        ("outcome", [float("inf")] * 8, "finite"),
        ("assignment_probability", 0, "between"),
        ("assignment_probability", 1, "between"),
        ("assignment_probability", True, "finite number"),
        ("assignment_probability", float("nan"), "finite number"),
        ("confidence_level", 1, "between"),
        ("confidence_level", float("inf"), "finite number"),
        ("policy_coverage", -0.1, "between"),
        ("policy_coverage", 1.1, "between"),
        ("policy_coverage", True, "finite number"),
        ("policy_coverage", None, "exactly one"),
        ("n_bins", 0, "range"),
        ("n_bins", 101, "range"),
        ("n_bins", True, "integer"),
        ("n_bootstrap", 1, "range"),
        ("n_bootstrap", 1001, "range"),
        ("n_bootstrap", 3.5, "integer"),
        ("seed", -1, "range"),
        ("seed", False, "integer"),
        ("experiment_id", " ", "experiment_id"),
        ("experiment_id", None, "experiment_id"),
    ],
)
def test_rct_rejects_invalid_inputs(name: str, value: object, message: str) -> None:
    values = _inputs()
    values[name] = value
    with pytest.raises(ValueError, match=message):
        evaluate_rct_predictions(**values)


def test_rct_rejects_ambiguous_policy_or_nonfinite_threshold() -> None:
    values = _inputs()
    values["policy_threshold"] = 0
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_rct_predictions(**values)
    values.update(policy_coverage=None, policy_threshold=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        evaluate_rct_predictions(**values)


def test_rct_rejects_empty_multidimensional_and_oversized_cohorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tributo_algorithms_causal_xlearner import rct_metrics

    values = _inputs()
    values["prediction_ids"] = []
    with pytest.raises(ValueError, match="empty"):
        evaluate_rct_predictions(**values)
    values["prediction_ids"] = [[i] for i in range(1, 9)]
    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_rct_predictions(**values)
    monkeypatch.setattr(rct_metrics, "MAX_RCT_ROWS", 7)
    with pytest.raises(ValueError, match="at most 7"):
        evaluate_rct_predictions(**_inputs())


def test_rct_rejects_nonfinite_derived_ipw_weights() -> None:
    values = _inputs()
    values["assignment_probability"] = float(np.nextafter(0, 1))
    with pytest.raises(ValueError, match="IPW contributions"):
        evaluate_rct_predictions(**values)


def test_public_rct_import_does_not_load_frameworks() -> None:
    script = """
import sys
from tributo_algorithms_causal_xlearner import RCTEvaluationResult, evaluate_rct_predictions
assert callable(evaluate_rct_predictions)
assert not {'ray', 'xgboost', 'torch'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
