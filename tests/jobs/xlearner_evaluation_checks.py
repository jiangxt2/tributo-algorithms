"""Real-Ray regression cases executed inside the installed-Wheel Jobs Gate."""

from __future__ import annotations

import random

import numpy as np
import ray
from tributo_algorithms_causal_xlearner import evaluate_rct_predictions
from tributo_algorithms_causal_xlearner.metrics import (
    CATE_COLUMN,
    CONTRIBUTION_COLUMN,
    IDENTITY_COLUMN,
    OUTCOME_COLUMN,
    QUADRANT_COLUMN,
    TREATMENT_COLUMN,
    evaluate_xlearner_dataset,
)


def verify_partitioned_evaluation() -> dict[str, object]:
    """Check the OOF scan on repeated keys, uneven blocks, and input permutations."""
    values = [
        (1, 0.8, 1, 1, "high_response_high_uplift"),
        (2, 0.2, 0, 0, "high_response_high_uplift"),
        (2, 0.2, 0, 0, "high_response_high_uplift"),
        (2, 0.2, 0, 1, "high_response_low_uplift"),
        (2, 0.2, 1, 0, "low_response_low_uplift"),
        (2, 0.2, 1, 1, "low_response_high_uplift"),
        (2, 0.2, 1, 1, "low_response_high_uplift"),
        (4, -0.8, 0, 1, "high_response_low_uplift"),
    ]
    rows = [
        {
            IDENTITY_COLUMN: identity,
            CATE_COLUMN: cate,
            TREATMENT_COLUMN: treatment,
            OUTCOME_COLUMN: float(outcome),
            CONTRIBUTION_COLUMN: float(outcome * (2 * treatment - 1)),
            QUADRANT_COLUMN: quadrant,
        }
        for identity, cate, treatment, outcome, quadrant in values
    ] * 4
    reports = []
    for seed, blocks in ((17, 1), (23, 3), (42, 7)):
        shuffled = rows.copy()
        random.Random(seed).shuffle(shuffled)
        reports.append(
            evaluate_xlearner_dataset(
                ray.data.from_items(shuffled, override_num_blocks=blocks),
                rows=32,
                treated_rows=16,
                control_rows=16,
                fold_count=2,
                n_points=17,
                n_bins=7,
            )
        )
    if reports[1:] != [reports[0], reports[0]]:
        raise AssertionError(
            "OOF report changed with input order or physical partitioning"
        )
    report = reports[0]
    curve = report["upliftCurve"]
    if (
        curve["modelGain"][0] != 0
        or curve["modelGain"][-1] != 4
        or report["ate"] != 0.15
    ):
        raise AssertionError("OOF scan lost rows or changed reference values")
    # A public evaluator also accepts an ordinary held-out RCT cohort without fit.
    rct = evaluate_rct_predictions(
        prediction_ids=list(range(8)),
        cate_predictions=list(range(8, 0, -1)),
        observation_ids=list(range(8)),
        treatment=[1, 0, 1, 0, 1, 0, 1, 0],
        outcome=[1, 0, 1, 1, 0, 0, 0, 1],
        assignment_probability=0.5,
        experiment_id="ray-job-reference-rct",
        policy_coverage=0.5,
        n_bins=2,
        n_bootstrap=200,
        seed=17,
    )
    if not np.isclose(rct["metrics"]["qini_ipw"], 0.25) or not np.isclose(
        rct["policy"]["value"], 0.75
    ):
        raise AssertionError("installed-Wheel RCT evaluation changed reference values")
    return {"partition_layouts": [1, 3, 7], "rows_per_layout": 32, "rct_reference": rct}
