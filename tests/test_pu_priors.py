"""Tests for official PU class-prior estimators."""

from __future__ import annotations

import numpy as np
import pytest
from tributo_algorithms_tabular_torch.priors import (
    em_prior,
    estimate_class_prior,
    histogram_match_prior,
    label_frequency_prior,
)


def test_label_frequency_and_validation() -> None:
    assert label_frequency_prior(15, 100) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="non-negative"):
        label_frequency_prior(-1, 100)


def test_histogram_and_em_priors_are_bounded() -> None:
    positives = np.asarray([0.8, 0.9, 0.95])
    unlabeled = np.asarray([0.1, 0.2, 0.7, 0.8])
    assert 0 <= histogram_match_prior(positives, unlabeled, n_bins=5) <= 1
    assert 0 <= em_prior(positives, unlabeled, max_iter=10) <= 1


def test_unified_prior_dispatches_and_rejects_unknown_method() -> None:
    assert estimate_class_prior(2, 10) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="Unknown method"):
        estimate_class_prior(2, 10, method="unknown")
