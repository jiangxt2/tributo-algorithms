"""Tests for official distributed causal core algorithms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from tributo.algorithms.api import (
    DistributionStrategy,
    InputBinding,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo_algorithms_causal_core import (
    ATE_DESCRIPTOR,
    LINEAR_DML_DESCRIPTOR,
    LINEAR_IV_DESCRIPTOR,
    DifferenceInMeansATE,
    LinearDMLATE,
    LinearIVATE,
)
from tributo_algorithms_causal_core.contracts import TreatmentCoverageValidator


def _algorithm() -> DifferenceInMeansATE:
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            primary_input_binding=InputBinding(
                name="train",
                resolver_id="tributo.fake_tabular",
                reference="causal",
                feature_names=("treatment", "x0"),
                label_name="outcome",
            ),
            algorithm_config={
                "treatment_col": "treatment",
                "policy_cost": 0.5,
                "cross_fit_folds": 1,
            },
        ),
    )
    return DifferenceInMeansATE(plan)


def test_ate_descriptor_uses_exact_map_reduce() -> None:
    distribution = ATE_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.RAY_MAP_REDUCE
    assert ATE_DESCRIPTOR.registration.contract_bindings is not None
    assert (
        LINEAR_DML_DESCRIPTOR.registration.distribution_spec.strategy
        is DistributionStrategy.RAY_MAP_REDUCE
    )
    assert (
        LINEAR_IV_DESCRIPTOR.registration.distribution_spec.strategy
        is DistributionStrategy.RAY_MAP_REDUCE
    )


def test_difference_in_means_merges_exact_sufficient_statistics() -> None:
    algorithm = _algorithm()
    first = algorithm.map_partition(
        (
            {
                "treatment": [0, 1, 0, 1],
                "outcome": [1.0, 3.0, 2.0, 4.0],
            },
        ),
        AlgorithmExecutionContext(inputs={}),
    )
    second = algorithm.map_partition(
        (
            {
                "treatment": [0, 1, 0, 1],
                "outcome": [2.0, 4.0, 3.0, 5.0],
            },
        ),
        AlgorithmExecutionContext(inputs={}),
    )
    merged = algorithm.merge_states(first, second)
    assert all(isinstance(value, np.ndarray) for value in merged.values())
    model = algorithm.finalize_model(merged)

    assert model.treated_count == 4
    assert model.control_count == 4
    assert model.effect == 2.0
    assert model.treat_all_policy is True
    assert algorithm.coverage_counts(first) == {"treated": 2, "control": 2}


def test_treatment_coverage_contract_partitions_rows() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {
                "input_rows": {
                    "train": 4,
                    "coverage.treated": 2,
                    "coverage.control": 2,
                }
            },
            {
                "input_rows": {
                    "train": 4,
                    "coverage.treated": 2,
                    "coverage.control": 2,
                }
            },
        ],
    }
    assert TreatmentCoverageValidator().validate(value) == value


def test_linear_dml_partials_out_distributed_confounders() -> None:
    base = _algorithm().plan
    algorithm = LinearDMLATE(base)
    treatment = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
    confounder = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.float64)
    outcome = 1.0 + 2.0 * treatment + 1.5 * confounder
    states = []
    for start in (0, 4):
        states.append(
            algorithm.map_partition(
                (
                    {
                        "treatment": treatment[start : start + 4],
                        "x0": confounder[start : start + 4],
                        "outcome": outcome[start : start + 4],
                    },
                ),
                AlgorithmExecutionContext(inputs={}),
            )
        )
    model = algorithm.finalize_model(algorithm.merge_states(*states))
    assert model.method == "distributed_linear_dml"
    assert model.effect == pytest.approx(2.0)


def test_linear_dml_supports_deterministic_cross_fit_state() -> None:
    base = _algorithm().plan
    plan = SimpleNamespace(
        primary_input_binding=base.primary_input_binding,
        algorithm_config={
            "treatment_col": "treatment",
            "cross_fit_folds": 2,
        },
    )
    algorithm = LinearDMLATE(plan)
    treatment = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
    confounder = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.float64)
    outcome = 1.0 + 2.0 * treatment + 1.5 * confounder
    states = [
        algorithm.map_partition(
            (
                {
                    "treatment": treatment[start : start + 4],
                    "x0": confounder[start : start + 4],
                    "outcome": outcome[start : start + 4],
                },
            ),
            AlgorithmExecutionContext(inputs={}),
        )
        for start in (0, 4)
    ]
    merged = algorithm.merge_states(*states)
    assert merged["fold_xtx"].shape == (2, 2, 2)
    model = algorithm.finalize_model(merged)
    assert model.effect == pytest.approx(2.0)
    assert model.diagnostics["cross_fit_folds"] == 2


def test_linear_iv_recovers_synthetic_truth() -> None:
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            primary_input_binding=InputBinding(
                name="train",
                resolver_id="tributo.fake_tabular",
                reference="iv",
                feature_names=("treatment", "instrument", "x0"),
                label_name="outcome",
            ),
            algorithm_config={
                "treatment_col": "treatment",
                "instrument_col": "instrument",
                "policy_cost": 0.5,
            },
        ),
    )
    algorithm = LinearIVATE(plan)
    instrument = np.asarray([0, 1] * 8, dtype=np.float64)
    treatment = instrument.copy()
    confounder = np.asarray([0, 0, 1, 1, 2, 2, 3, 3] * 2, dtype=np.float64)
    outcome = 1.0 + 2.0 * treatment + 1.5 * confounder
    states = []
    for start in (0, 8):
        states.append(
            algorithm.map_partition(
                (
                    {
                        "treatment": treatment[start : start + 8],
                        "instrument": instrument[start : start + 8],
                        "x0": confounder[start : start + 8],
                        "outcome": outcome[start : start + 8],
                    },
                ),
                AlgorithmExecutionContext(inputs={}),
            )
        )
    model = algorithm.finalize_model(algorithm.merge_states(*states))
    assert model.method == "distributed_linear_2sls"
    assert model.effect == pytest.approx(2.0)
    assert model.diagnostics["first_stage_correlation"] == pytest.approx(1.0)
