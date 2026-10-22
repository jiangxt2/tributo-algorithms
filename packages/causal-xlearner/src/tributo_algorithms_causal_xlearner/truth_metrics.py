"""Truth-based held-out effect metrics for X-Learner predictions."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import TypedDict

import numpy as np

_PEHE_DEFINITION = "sqrt(mean((cate_pred - (mu1 - mu0)) ** 2))"
_ATE_ERROR_DEFINITION = "abs(mean(cate_pred) - mean(mu1 - mu0))"


class CATEEvaluationResult(TypedDict):
    """Metrics and interpretation for one truth-known evaluation cohort."""

    ATE_error: float
    evaluated_estimand: str
    metric_definitions: dict[str, str]
    metric_units: str
    normalization: str
    sample_count: int
    sqrt_PEHE: float


def _identity_vector(values: object, *, name: str) -> list[str | int]:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a one-dimensional sequence of IDs") from error
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")

    identities: list[str | int] = []
    for value in raw:
        if isinstance(value, bool):
            raise ValueError(f"{name} IDs must be non-empty strings or integers")
        if isinstance(value, Integral):
            identities.append(int(value))
        elif isinstance(value, str) and value:
            identities.append(value)
        else:
            raise ValueError(f"{name} IDs must be non-empty strings or integers")

    if len(set(identities)) != len(identities):
        raise ValueError(f"{name} contains duplicate IDs")
    return identities


def _finite_vector(values: object, *, name: str) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if np.iscomplexobj(raw) or (
        raw.dtype == object and any(np.iscomplexobj(value) for value in raw.flat)
    ):
        raise ValueError(f"{name} must contain real numeric values")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _stable_mean(values: np.ndarray, *, name: str) -> float:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    result = scale * float(np.mean(values / scale))
    if not np.isfinite(result):
        raise ValueError(f"{name} cannot be represented as a finite result")
    return result


def evaluate_cate_predictions(
    prediction_ids: Sequence[str | int] | np.ndarray,
    cate_predictions: Sequence[float] | np.ndarray,
    truth_ids: Sequence[str | int] | np.ndarray,
    mu0: Sequence[float] | np.ndarray,
    mu1: Sequence[float] | np.ndarray,
) -> CATEEvaluationResult:
    """Evaluate CATE predictions against potential-outcome means by sample ID.

    The result uses unnormalized root PEHE in outcome units and absolute ATE
    error over the evaluated cohort. It does not reproduce ACIC competition
    scoring or report SATT error.
    """
    predicted_ids = _identity_vector(prediction_ids, name="prediction_ids")
    expected_ids = _identity_vector(truth_ids, name="truth_ids")
    cate = _finite_vector(cate_predictions, name="cate_predictions")
    control_mean = _finite_vector(mu0, name="mu0")
    treated_mean = _finite_vector(mu1, name="mu1")

    if cate.size != len(predicted_ids):
        raise ValueError("cate_predictions length must match prediction_ids")
    if control_mean.size != len(expected_ids) or treated_mean.size != len(expected_ids):
        raise ValueError("mu0 and mu1 lengths must match truth_ids")

    truth_positions = {sample_id: index for index, sample_id in enumerate(expected_ids)}
    prediction_id_set = set(predicted_ids)
    missing = [
        sample_id for sample_id in predicted_ids if sample_id not in truth_positions
    ]
    extra = [
        sample_id for sample_id in expected_ids if sample_id not in prediction_id_set
    ]
    if missing or extra:
        raise ValueError(
            "prediction and truth IDs must have exact coverage "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    truth_order = np.fromiter(
        (truth_positions[sample_id] for sample_id in predicted_ids),
        dtype=np.intp,
        count=len(predicted_ids),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        true_effect = treated_mean[truth_order] - control_mean[truth_order]
        effect_error = cate - true_effect
    if not np.isfinite(true_effect).all() or not np.isfinite(effect_error).all():
        raise ValueError("derived treatment effects must be finite")

    scale = float(np.max(np.abs(effect_error)))
    sqrt_pehe = (
        0.0
        if scale == 0.0
        else scale * float(np.sqrt(np.mean(np.square(effect_error / scale))))
    )
    ate_error = abs(
        _stable_mean(cate, name="predicted ATE")
        - _stable_mean(true_effect, name="true ATE")
    )
    if not np.isfinite(sqrt_pehe) or not np.isfinite(ate_error):
        raise ValueError("effect metrics cannot be represented as finite results")

    return {
        "ATE_error": ate_error,
        "evaluated_estimand": "population_ATE_over_evaluated_cohort",
        "metric_definitions": {
            "ATE_error": _ATE_ERROR_DEFINITION,
            "sqrt_PEHE": _PEHE_DEFINITION,
        },
        "metric_units": "outcome_units",
        "normalization": "none",
        "sample_count": len(predicted_ids),
        "sqrt_PEHE": sqrt_pehe,
    }


__all__ = ["CATEEvaluationResult", "evaluate_cate_predictions"]
