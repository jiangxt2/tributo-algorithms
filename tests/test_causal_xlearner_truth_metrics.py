"""Tests for truth-based held-out X-Learner effect metrics."""

from __future__ import annotations

import numpy as np
import pytest
from tributo_algorithms_causal_xlearner import evaluate_cate_predictions


def test_truth_metrics_align_by_id_and_report_project_metric_semantics() -> None:
    result = evaluate_cate_predictions(
        prediction_ids=["sample-b", "sample-a"],
        cate_predictions=[1.0, 3.0],
        truth_ids=["sample-a", "sample-b"],
        mu0=[0.0, 1.0],
        mu1=[4.0, 3.0],
    )

    assert result["sqrt_PEHE"] == pytest.approx(1.0)
    assert result["ATE_error"] == pytest.approx(1.0)
    assert result["sample_count"] == 2
    assert result["evaluated_estimand"] == "population_ATE_over_evaluated_cohort"
    assert result["metric_units"] == "outcome_units"
    assert result["normalization"] == "none"
    assert result["metric_definitions"] == {
        "ATE_error": "abs(mean(cate_pred) - mean(mu1 - mu0))",
        "sqrt_PEHE": "sqrt(mean((cate_pred - (mu1 - mu0)) ** 2))",
    }


def test_truth_metrics_are_invariant_to_row_order() -> None:
    forward = evaluate_cate_predictions(
        prediction_ids=[1, 2, 3],
        cate_predictions=[0.5, 1.5, 2.5],
        truth_ids=[1, 2, 3],
        mu0=[0.0, 1.0, 2.0],
        mu1=[1.0, 3.0, 2.0],
    )
    reverse = evaluate_cate_predictions(
        prediction_ids=[3, 2, 1],
        cate_predictions=[2.5, 1.5, 0.5],
        truth_ids=[2, 3, 1],
        mu0=[1.0, 2.0, 0.0],
        mu1=[3.0, 2.0, 1.0],
    )

    assert reverse["sqrt_PEHE"] == pytest.approx(forward["sqrt_PEHE"])
    assert reverse["ATE_error"] == pytest.approx(forward["ATE_error"])
    assert reverse["sample_count"] == forward["sample_count"]


@pytest.mark.parametrize(
    ("prediction_ids", "truth_ids", "message"),
    [
        ([1, 1], [1, 2], "prediction_ids contains duplicate IDs"),
        ([1, 2], [1, 1], "truth_ids contains duplicate IDs"),
        ([1, 2], [1, 3], "exact coverage"),
    ],
)
def test_truth_metrics_reject_duplicate_or_mismatched_ids(
    prediction_ids: list[int], truth_ids: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_cate_predictions(
            prediction_ids=prediction_ids,
            cate_predictions=[0.0, 1.0],
            truth_ids=truth_ids,
            mu0=[0.0, 0.0],
            mu1=[1.0, 1.0],
        )


@pytest.mark.parametrize(
    ("prediction_ids", "cate", "truth_ids", "mu0", "mu1"),
    [
        ([1, 2, 3], [0.0, 1.0, 2.0], [1, 2], [0.0, 0.0], [1.0, 1.0]),
        ([1, 2], [0.0, 1.0], [1, 2, 3], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    ],
)
def test_truth_metrics_reject_missing_or_extra_ids(
    prediction_ids: list[int],
    cate: list[float],
    truth_ids: list[int],
    mu0: list[float],
    mu1: list[float],
) -> None:
    with pytest.raises(ValueError, match="exact coverage"):
        evaluate_cate_predictions(prediction_ids, cate, truth_ids, mu0, mu1)


@pytest.mark.parametrize(
    ("prediction_ids", "cate", "truth_ids", "mu0", "mu1", "message"),
    [
        ([1, 2], [0.0], [1, 2], [0.0, 0.0], [1.0, 1.0], "length"),
        ([1, 2], [0.0, 1.0], [1, 2], [0.0], [1.0, 1.0], "length"),
        ([1, 2], [0.0, np.nan], [1, 2], [0.0, 0.0], [1.0, 1.0], "finite"),
        ([1, 2], [0.0, 1.0], [1, 2], [0.0, np.inf], [1.0, 1.0], "finite"),
        ([1, 2], [0.0, 1.0], [1, 2], [0.0, 0.0], [1.0, -np.inf], "finite"),
    ],
)
def test_truth_metrics_reject_invalid_values_and_lengths(
    prediction_ids: list[int],
    cate: list[float],
    truth_ids: list[int],
    mu0: list[float],
    mu1: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_cate_predictions(prediction_ids, cate, truth_ids, mu0, mu1)


def test_truth_metrics_reject_empty_and_non_vector_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_cate_predictions([], [], [], [], [])

    with pytest.raises(ValueError, match="one-dimensional"):
        evaluate_cate_predictions(
            prediction_ids=[[1, 2]],
            cate_predictions=[0.0, 1.0],
            truth_ids=[1, 2],
            mu0=[0.0, 0.0],
            mu1=[1.0, 1.0],
        )

    with pytest.raises(ValueError, match="non-empty strings or integers"):
        evaluate_cate_predictions(
            prediction_ids=[True, 2],
            cate_predictions=[0.0, 1.0],
            truth_ids=[True, 2],
            mu0=[0.0, 0.0],
            mu1=[1.0, 1.0],
        )
