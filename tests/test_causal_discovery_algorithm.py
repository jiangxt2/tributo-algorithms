"""Tests for official distributed causal discovery."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from tributo.algorithms.api import (
    DistributionStrategy,
    InputBinding,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.runtime import BundleModelLoader
from tributo_algorithms_causal_discovery import (
    PC_DISCOVERY_DESCRIPTOR,
    DistributedPCStability,
)
from tributo_algorithms_causal_discovery.algorithm import (
    CausalGraphModel,
    _query_onnx,
    export_graph,
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
    assert (
        PC_DISCOVERY_DESCRIPTOR.registration.implementation.flavor_id
        == "onnx-runtime-v1"
    )


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


def test_pc_query_onnx_is_exact_and_marks_invalid_indices() -> None:
    import onnxruntime as ort

    model = CausalGraphModel(
        variables=("x0", "x1"),
        endpoint_matrix=((0, -1), (1, 0)),
        adjacency_vote_fraction=((0.0, 0.75), (0.75, 0.0)),
        run_count=4,
        row_count=64,
        alpha=0.05,
        vote_threshold=0.5,
    )
    session = ort.InferenceSession(
        _query_onnx(model),
        providers=["CPUExecutionProvider"],
    )
    left, right, votes, valid = session.run(
        None,
        {
            "edge_index": np.asarray(
                [[0, 1], [1, 0], [0, 0], [-1, 0], [0, 2]],
                dtype=np.int64,
            )
        },
    )
    np.testing.assert_array_equal(left, [-1, 1, 0, 0, 0])
    np.testing.assert_array_equal(right, [1, -1, 0, 0, 0])
    np.testing.assert_allclose(votes, [0.75, 0.75, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(valid, [True, True, True, False, False])


def test_pc_export_publishes_query_and_report_roles(tmp_path: Path) -> None:
    model = CausalGraphModel(
        variables=("x0", "x1"),
        endpoint_matrix=((0, -1), (1, 0)),
        adjacency_vote_fraction=((0.0, 0.75), (0.75, 0.0)),
        run_count=4,
        row_count=64,
        alpha=0.05,
        vote_threshold=0.5,
    )
    execution = export_graph(
        model=model,
        plan=cast(
            Any,
            SimpleNamespace(
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}},
                resolution=SimpleNamespace(
                    implementation_id=("tributo.official.causal_discovery.pc_stability")
                ),
            ),
        ),
        run_id="pc-export-test",
    )
    bundle_uri = cast(str, execution.outputs["bundle_uri"])
    manifest = BundleReader().read_manifest(bundle_uri)
    assert set(manifest.roles) == {"inference", "report"}
    runtime = BundleModelLoader().open(bundle_uri, role="inference", use_case="batch")
    try:
        outputs = runtime.predict(
            {"edge_index": np.asarray([[0, 1], [-1, 0]], dtype=np.int64)}
        )
    finally:
        runtime.close()
    np.testing.assert_array_equal(outputs["valid"], [True, False])
