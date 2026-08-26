"""Tests for official PU evaluation metrics."""

from __future__ import annotations

import numpy as np
from tributo_algorithms_tabular_torch.pu_metrics import (
    compute_pu_metrics,
    pu_auc_score,
    pu_calibration,
    pu_f1_score,
    pu_precision_score,
)


def test_pu_metrics_are_finite_and_bounded() -> None:
    labels = np.asarray([1, 1, 0, 0, 0, 0])
    scores = np.asarray([0.9, 0.8, 0.7, 0.4, 0.2, 0.1])
    prior = 0.25
    calibration = pu_calibration(labels, scores, prior)
    values = (
        pu_precision_score(labels, scores, prior),
        pu_f1_score(labels, scores, prior),
        pu_auc_score(labels, scores, prior),
        calibration["calibration_error"],
    )
    assert all(np.isfinite(value) for value in values)
    assert all(0 <= value <= 1 for value in values)


def test_metric_bundle_contains_pu_semantics() -> None:
    labels = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.2, 0.8, 0.1])
    metrics = compute_pu_metrics(labels, scores, class_prior=0.5)
    assert set(metrics) == {"pu_precision", "pu_recall", "pu_f1", "pu_auc"}
