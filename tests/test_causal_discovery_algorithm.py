"""Tests for official distributed causal discovery."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from tributo.algorithms.api import (
    DistributionStrategy,
    InputBinding,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo_algorithms_causal_discovery import (
    PC_DISCOVERY_DESCRIPTOR,
    DistributedPCStability,
)
from tributo_algorithms_causal_discovery.contracts import (
    DiscoveryCoverageValidator,
)


def _algorithm() -> DistributedPCStability:
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            primary_input_binding=InputBinding(
                name="train",
                resolver_id="tributo.fake_tabular",
                reference="pc",
                feature_names=("x0", "x1", "x2"),
            ),
            algorithm_config={"alpha": 0.05, "vote_threshold": 0.5},
        ),
    )
    return DistributedPCStability(plan)


def test_pc_descriptor_uses_map_reduce_stability_selection() -> None:
    distribution = PC_DISCOVERY_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.RAY_MAP_REDUCE


def test_pc_stability_merges_shard_graph_votes() -> None:
    rng = np.random.default_rng(7)
    x0 = rng.normal(size=256)
    x1 = 1.5 * x0 + rng.normal(scale=0.1, size=256)
    x2 = -0.8 * x1 + rng.normal(scale=0.1, size=256)
    algorithm = _algorithm()
    states = []
    for start in (0, 128):
        states.append(
            algorithm.map_partition(
                (
                    {
                        "x0": x0[start : start + 128],
                        "x1": x1[start : start + 128],
                        "x2": x2[start : start + 128],
                    },
                ),
                AlgorithmExecutionContext(inputs={}),
            )
        )
    model = algorithm.finalize_model(algorithm.merge_states(*states))
    assert model.run_count == 2
    assert model.row_count == 256
    assert any(any(value != 0 for value in row) for row in model.endpoint_matrix)


def test_discovery_coverage_contract_proves_every_shard_row() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {"input_rows": {"train": 128, "coverage.discovery_rows": 128}},
            {"input_rows": {"train": 128, "coverage.discovery_rows": 128}},
        ],
    }
    assert DiscoveryCoverageValidator().validate(value) == value
