"""Held-out, binary-outcome RCT evaluation without training or data IO."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import TypedDict

import numpy as np

from .truth_metrics import _finite_vector, _identity_vector, _stable_mean

MAX_RCT_ROWS = 100_000


class RCTMetrics(TypedDict):
    predicted_ate: float
    observed_ate: float
    ipw_ate: float
    auuc_ipw: float
    qini_ipw: float


class GainCurve(TypedDict):
    coverage: list[float]
    cumulative_gain: list[float]
    random_gain: list[float]


class UpliftBin(TypedDict):
    sample_count: int
    treated_rows: int
    control_rows: int
    predicted_cate: float
    observed_uplift: float | None
    confidence_interval: list[float] | None
    status: str


class PolicyEvaluation(TypedDict):
    rule: str
    requested_coverage: float | None
    threshold: float | None
    selected_rows: int
    coverage: float
    value: float
    no_treatment_value: float
    all_treatment_value: float
    random_value: float
    incremental_value: float
    gain_over_random: float


class RCTEvaluationResult(TypedDict):
    schema_version: int
    metric_version: str
    mode: str
    experiment_id: str
    assignment_probability: float
    sample_count: int
    treated_rows: int
    control_rows: int
    metric_units: str
    normalization: str
    metric_definitions: dict[str, str]
    metrics: RCTMetrics
    curve: GainCurve
    bins: list[UpliftBin]
    policy: PolicyEvaluation
    confidence_intervals: dict[str, list[float]]
    uncertainty: dict[str, object]
    limitations: list[str]


def _scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: object, name: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside the supported range")
    return result


def _aligned_inputs(
    prediction_ids: Sequence[str | int] | np.ndarray,
    cate_predictions: Sequence[float] | np.ndarray,
    observation_ids: Sequence[str | int] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    outcome: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = _identity_vector(prediction_ids, name="prediction_ids")
    observed_ids = _identity_vector(observation_ids, name="observation_ids")
    if len(ids) > MAX_RCT_ROWS or len(observed_ids) > MAX_RCT_ROWS:
        raise ValueError(f"RCT evaluation supports at most {MAX_RCT_ROWS} rows")
    if len({type(value) for value in (*ids, *observed_ids)}) != 1:
        raise ValueError("RCT IDs must use one consistent string or integer type")
    if set(ids) != set(observed_ids):
        raise ValueError("prediction and observation IDs must have exact coverage")
    cate = _finite_vector(cate_predictions, name="cate_predictions")
    assigned = _finite_vector(treatment, name="treatment")
    response = _finite_vector(outcome, name="outcome")
    if (
        len(cate) != len(ids)
        or len(assigned) != len(observed_ids)
        or len(response) != len(observed_ids)
    ):
        raise ValueError("RCT vector lengths must match their corresponding IDs")
    if not np.isin(assigned, (0.0, 1.0)).all():
        raise ValueError("treatment must be binary (0 or 1)")
    if not np.isin(response, (0.0, 1.0)).all():
        raise ValueError("outcome must be binary (0 or 1)")
    positions = {identity: index for index, identity in enumerate(observed_ids)}
    # Canonical ID order also makes seeded bootstrap independent of input order.
    order = sorted(range(len(ids)), key=lambda index: ids[index])
    observed_order = [positions[ids[index]] for index in order]
    assigned = assigned[observed_order]
    if min(int(np.sum(assigned == value)) for value in (0, 1)) < 2:
        raise ValueError(
            "RCT evaluation requires at least two rows in each treatment arm"
        )
    return cate[order], assigned, response[observed_order]


def _interval(samples: np.ndarray, confidence_level: float) -> list[float]:
    tail = (1.0 - confidence_level) / 2.0
    with np.errstate(over="ignore", invalid="ignore"):
        bounds = np.quantile(samples, (tail, 1.0 - tail), method="linear")
    if not np.isfinite(bounds).all():
        raise ValueError("confidence interval cannot be represented as finite values")
    return [float(value) for value in bounds]


def _statistics(
    treatment: np.ndarray,
    outcome: np.ndarray,
    treated_value: np.ndarray,
    control_value: np.ndarray,
    selected: np.ndarray,
    coverage: float,
) -> np.ndarray:
    effect = treated_value - control_value
    return np.asarray(
        [
            float(np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0])),
            _stable_mean(effect, name="IPW ATE"),
            _stable_mean(
                np.where(selected, treated_value, control_value), name="policy value"
            ),
            _stable_mean(
                np.where(selected, effect, 0.0), name="incremental policy value"
            ),
            _stable_mean(
                coverage * treated_value + (1.0 - coverage) * control_value,
                name="random policy value",
            ),
            _stable_mean(
                (selected.astype(float) - coverage) * effect, name="gain over random"
            ),
        ]
    )


def _uplift_bins(
    cate: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    ranking: np.ndarray,
    *,
    n_bins: int,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> list[UpliftBin]:
    result: list[UpliftBin] = []
    for indices in np.array_split(ranking, min(n_bins, len(ranking))):
        treated = outcome[indices][treatment[indices] == 1]
        control = outcome[indices][treatment[indices] == 0]
        observed = (
            float(treated.mean() - control.mean())
            if len(treated) and len(control)
            else None
        )
        interval = None
        valid = len(treated) >= 2 and len(control) >= 2
        if valid:
            draws = np.asarray(
                [
                    float(
                        rng.choice(treated, len(treated), replace=True).mean()
                        - rng.choice(control, len(control), replace=True).mean()
                    )
                    for _ in range(n_bootstrap)
                ]
            )
            interval = _interval(draws, confidence_level)
        result.append(
            {
                "sample_count": len(indices),
                "treated_rows": len(treated),
                "control_rows": len(control),
                "predicted_cate": _stable_mean(
                    cate[indices], name="bin predicted CATE"
                ),
                "observed_uplift": observed,
                "confidence_interval": interval,
                "status": "ok" if valid else "insufficient_group_rows",
            }
        )
    return result


def evaluate_rct_predictions(
    prediction_ids: Sequence[str | int] | np.ndarray,
    cate_predictions: Sequence[float] | np.ndarray,
    observation_ids: Sequence[str | int] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    outcome: Sequence[float] | np.ndarray,
    *,
    assignment_probability: float,
    experiment_id: str,
    policy_coverage: float | None = None,
    policy_threshold: float | None = None,
    n_bins: int = 10,
    confidence_level: float = 0.95,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> RCTEvaluationResult:
    """Evaluate frozen predictions and a preselected policy on a binary RCT.

    Supply the known constant random-assignment probability, never an estimated
    propensity. Declaring ``experiment_id`` does not establish randomization.
    Exactly one policy rule is required and must be chosen before evaluation.
    Coverage selects floor(n * coverage) rows; ties use ascending sample ID.
    Threshold selection uses CATE >= threshold. Actions remain fixed during
    customer-level bootstrap within treatment arms; the random baseline uses
    the policy's realized original-cohort coverage. Intervals do not include
    training uncertainty. At most 100,000 rows, 100 bins, and 1,000 resamples
    are accepted. No data are read, written, or collected from Ray here.
    """
    probability = _scalar(assignment_probability, "assignment_probability")
    level = _scalar(confidence_level, "confidence_level")
    if not 0.0 < probability < 1.0 or not 0.0 < level < 1.0:
        raise ValueError(
            "assignment_probability and confidence_level must be between 0 and 1"
        )
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError(
            "experiment_id must identify the declared randomized experiment"
        )
    bins = _integer(n_bins, "n_bins", 1, 100)
    resamples = _integer(n_bootstrap, "n_bootstrap", 2, 1000)
    random_seed = _integer(seed, "seed", 0)
    if (policy_coverage is None) == (policy_threshold is None):
        raise ValueError("provide exactly one of policy_coverage or policy_threshold")
    requested = (
        None if policy_coverage is None else _scalar(policy_coverage, "policy_coverage")
    )
    threshold = (
        None
        if policy_threshold is None
        else _scalar(policy_threshold, "policy_threshold")
    )
    if requested is not None and not 0.0 <= requested <= 1.0:
        raise ValueError("policy_coverage must be between 0 and 1")
    cate, assigned, response = _aligned_inputs(
        prediction_ids, cate_predictions, observation_ids, treatment, outcome
    )
    rows = len(cate)
    ranking = np.argsort(-cate, kind="stable")
    if requested is not None:
        selected = np.zeros(rows, dtype=bool)
        selected[ranking[: math.floor(rows * requested)]] = True
    else:
        selected = cate >= threshold
    selected_rows = int(selected.sum())
    coverage = selected_rows / rows
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        treated_value = assigned * response / probability
        control_value = (1.0 - assigned) * response / (1.0 - probability)
    if not np.isfinite(treated_value).all() or not np.isfinite(control_value).all():
        raise ValueError(
            "IPW contributions must be finite; assignment probability is too extreme"
        )
    effect = treated_value - control_value
    x_axis = np.arange(rows + 1, dtype=float) / rows
    # Normalize before summing to avoid unnecessary cumulative-count overflow.
    gain = np.concatenate(([0.0], np.cumsum(effect[ranking] / rows)))
    if not np.isfinite(gain).all():
        raise ValueError("cumulative IPW gain must be finite")
    random_gain = x_axis * gain[-1]
    auuc = float(np.trapezoid(gain, x_axis))
    qini = float(np.trapezoid(gain - random_gain, x_axis))
    if not math.isfinite(auuc) or not math.isfinite(qini):
        raise ValueError("IPW areas must be finite")
    statistics = _statistics(
        assigned, response, treated_value, control_value, selected, coverage
    )
    rng = np.random.default_rng(random_seed)
    treated_indices = np.flatnonzero(assigned == 1)
    control_indices = np.flatnonzero(assigned == 0)
    draws = np.empty((resamples, len(statistics)), dtype=float)
    for index in range(resamples):
        sample = np.concatenate(
            (
                rng.choice(treated_indices, len(treated_indices), replace=True),
                rng.choice(control_indices, len(control_indices), replace=True),
            )
        )
        draws[index] = _statistics(
            assigned[sample],
            response[sample],
            treated_value[sample],
            control_value[sample],
            selected[sample],
            coverage,
        )
    interval_names = (
        "observed_ate",
        "ipw_ate",
        "policy_value",
        "incremental_value",
        "random_value",
        "gain_over_random",
    )
    return {
        "schema_version": 1,
        "metric_version": "rct-ipw-v1",
        "mode": "rct",
        "experiment_id": experiment_id,
        "assignment_probability": probability,
        "sample_count": rows,
        "treated_rows": len(treated_indices),
        "control_rows": len(control_indices),
        "metric_units": "outcome_probability",
        "normalization": "none",
        "metric_definitions": {
            "predicted_ate": "mean(cate_predictions) over the evaluated cohort",
            "observed_ate": "mean(Y | T=1) - mean(Y | T=0)",
            "ipw_ate": "mean(T*Y/p - (1-T)*Y/(1-p))",
            "cumulative_gain": "sum(top-k (T*Y/p - (1-T)*Y/(1-p))) / n",
            "auuc_ipw": "trapezoid(cumulative_gain, coverage), all n+1 ranks",
            "qini_ipw": "trapezoid(cumulative_gain - random_gain, coverage)",
            "policy_value": "mean(action*T*Y/p + (1-action)*(1-T)*Y/(1-p))",
            "incremental_value": "mean(action*(T*Y/p - (1-T)*Y/(1-p)))",
            "random_value": "mean(coverage*T*Y/p + (1-coverage)*(1-T)*Y/(1-p))",
            "gain_over_random": "mean((action-coverage)*(T*Y/p - (1-T)*Y/(1-p)))",
        },
        "metrics": {
            "predicted_ate": _stable_mean(cate, name="predicted ATE"),
            "observed_ate": float(statistics[0]),
            "ipw_ate": float(statistics[1]),
            "auuc_ipw": auuc,
            "qini_ipw": qini,
        },
        "curve": {
            "coverage": x_axis.tolist(),
            "cumulative_gain": gain.tolist(),
            "random_gain": random_gain.tolist(),
        },
        "bins": _uplift_bins(
            cate,
            assigned,
            response,
            ranking,
            n_bins=bins,
            n_bootstrap=resamples,
            confidence_level=level,
            rng=rng,
        ),
        "policy": {
            "rule": "top_coverage" if requested is not None else "cate_threshold",
            "requested_coverage": requested,
            "threshold": threshold,
            "selected_rows": selected_rows,
            "coverage": coverage,
            "value": float(statistics[2]),
            "no_treatment_value": _stable_mean(
                control_value, name="no-treatment value"
            ),
            "all_treatment_value": _stable_mean(
                treated_value, name="all-treatment value"
            ),
            "random_value": float(statistics[4]),
            "incremental_value": float(statistics[3]),
            "gain_over_random": float(statistics[5]),
        },
        "confidence_intervals": {
            name: _interval(draws[:, index], level)
            for index, name in enumerate(interval_names)
        },
        "uncertainty": {
            "method": "treatment_stratified_customer_percentile_bootstrap",
            "confidence_level": level,
            "resamples": resamples,
            "seed": random_seed,
            "policy_actions": "fixed_on_original_cohort",
            "scope": "fixed_model_and_policy_test_sample_uncertainty",
            "bin_resampling": "within_each_bin_and_treatment_arm",
        },
        "limitations": [
            "Randomization and known assignment probability are caller-declared assumptions.",
            "No individual treatment-effect truth is observed in an RCT.",
            "Intervals cover ATE, policy statistics, and eligible bins; they are not simultaneous or curve-area intervals.",
            "No training uncertainty or policy selection using this test cohort is included.",
        ],
    }


__all__ = ["RCTEvaluationResult", "evaluate_rct_predictions"]
