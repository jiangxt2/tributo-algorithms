"""Tests for synchronous SGD and the conditional Isolation Forest ensemble."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
from tributo.algorithms.api import ResolvedAlgorithmPlan
from tributo.algorithms.input.tabular import InMemoryTabularInputView
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo_algorithms_classical.isolation_forest import IsolationForestEnsemble
from tributo_algorithms_classical.sgd import (
    DistributedSGDClassifier,
    DistributedSGDRegressor,
)
from tributo_algorithms_classical.unsupervised_exporter import (
    export_isolation_forest_model,
)


def _plan(
    feature_names: tuple[str, ...],
    *,
    label_name: str | None,
    config: dict[str, object],
) -> ResolvedAlgorithmPlan:
    return cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            primary_input_binding=SimpleNamespace(
                feature_names=feature_names,
                label_name=label_name,
                sample_weight_name=None,
            ),
            algorithm_config=config,
            runtime=SimpleNamespace(distribution_digest=""),
            resolution=SimpleNamespace(algorithm="test"),
        ),
    )


def test_synchronous_sgd_classifier_uses_all_shards() -> None:
    plan = _plan(
        ("x0",),
        label_name="label",
        config={
            "alpha": 0.0001,
            "learning_rate": 0.2,
            "max_iter": 20,
            "tolerance": 1e-8,
        },
    )
    algorithm = DistributedSGDClassifier(plan)
    context = AlgorithmExecutionContext(inputs={})
    state = algorithm.initialize_state(plan.algorithm_config, object())
    left = {
        "x0": np.asarray([-3.0, -2.0, -1.0, 0.0]),
        "label": np.asarray([0, 0, 0, 0]),
    }
    right = {"x0": np.asarray([1.0, 2.0, 3.0, 4.0]), "label": np.asarray([1, 1, 1, 1])}
    for round_index in range(20):
        left_update = algorithm.compute_partition_update(
            (left,), state, round_index, context
        )
        right_update = algorithm.compute_partition_update(
            (right,), state, round_index, context
        )
        update = algorithm.merge_updates(left_update, right_update)
        state = algorithm.apply_update(state, update, round_index)
    model = algorithm.finalize_model(state)
    predictions = model.estimator.predict(np.asarray([[-2.0], [2.0]]))
    np.testing.assert_array_equal(predictions, np.asarray([0, 1]))


def test_synchronous_sgd_regressor_converges_on_linear_data() -> None:
    plan = _plan(
        ("x0",),
        label_name="label",
        config={"learning_rate": 0.1, "max_iter": 25},
    )
    algorithm = DistributedSGDRegressor(plan)
    context = AlgorithmExecutionContext(inputs={})
    state = algorithm.initialize_state(plan.algorithm_config, object())
    batch = {
        "x0": np.asarray([-2.0, -1.0, 1.0, 2.0]),
        "label": np.asarray([-4.0, -2.0, 2.0, 4.0]),
    }
    for round_index in range(25):
        update = algorithm.compute_partition_update(
            (batch,), state, round_index, context
        )
        state = algorithm.apply_update(state, update, round_index)
    model = algorithm.finalize_model(state)
    np.testing.assert_allclose(
        model.estimator.predict(np.asarray([[3.0], [-3.0]])), [6.0, -6.0], atol=0.5
    )


def test_sgd_round_aggregates_one_global_gradient_without_stale_steps() -> None:
    plan = _plan(
        ("x0",),
        label_name="label",
        config={"learning_rate": 0.1, "max_iter": 1},
    )
    algorithm = DistributedSGDRegressor(plan)
    state = algorithm.initialize_state(plan.algorithm_config, object())
    update = algorithm.compute_partition_update(
        (
            {"x0": np.asarray([1.0, 2.0]), "label": np.asarray([2.0, 4.0])},
            {"x0": np.asarray([3.0]), "label": np.asarray([6.0])},
        ),
        state,
        0,
        AlgorithmExecutionContext(inputs={}),
    )
    assert "steps" not in update
    np.testing.assert_allclose(update["gradient"], np.asarray([-28.0]))
    updated = algorithm.apply_update(state, update, 0)
    assert algorithm.should_stop(updated, {"loss": 1.0}, 0)


def test_isolation_forest_parallel_units_produce_scores() -> None:
    features = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.2, 0.1], [10.0, 10.0]],
        dtype=np.float64,
    )
    view = InMemoryTabularInputView(
        _columns={"x0": tuple(features[:, 0]), "x1": tuple(features[:, 1])},
        feature_names=("x0", "x1"),
        label_name=None,
    )
    plan = _plan(
        ("x0", "x1"),
        label_name=None,
        config={"n_estimators": 8, "max_samples": 5, "contamination": 0.2},
    )
    algorithm = IsolationForestEnsemble(plan)
    context = AlgorithmExecutionContext(inputs={})
    units = algorithm.plan_units(plan.algorithm_config, object(), seed=7)
    fitted = tuple(algorithm.fit_unit(unit, {"train": view}, context) for unit in units)
    model = algorithm.finalize_ensemble(algorithm.merge_units(fitted))
    scores = model.estimator.score_samples(features)
    assert scores.shape == (5,)
    assert float(scores[-1]) < float(scores[:4].mean())
    np.testing.assert_allclose(model.estimator.offset_, np.quantile(scores, 0.2))


def test_isolation_forest_export_produces_runnable_score_bundle(
    tmp_path: Path,
) -> None:
    from tributo.exporting.bundle_reader import BundleReader

    features = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.2, 0.1], [10.0, 10.0]],
        dtype=np.float64,
    )
    view = InMemoryTabularInputView(
        _columns={"x0": tuple(features[:, 0]), "x1": tuple(features[:, 1])},
        feature_names=("x0", "x1"),
        label_name=None,
    )
    plan = _plan(
        ("x0", "x1"),
        label_name=None,
        config={
            "n_estimators": 8,
            "max_samples": 5,
            "contamination": "auto",
            "output": {"bundle_uri": str(tmp_path / "bundle")},
        },
    )
    algorithm = IsolationForestEnsemble(plan)
    context = AlgorithmExecutionContext(inputs={})
    units = algorithm.plan_units(plan.algorithm_config, object(), seed=7)
    fitted = tuple(algorithm.fit_unit(unit, {"train": view}, context) for unit in units)
    model = algorithm.finalize_ensemble(algorithm.merge_units(fitted))
    execution = export_isolation_forest_model(
        model=model, plan=plan, run_id="isolation-export-test"
    )
    with BundleReader().open_artifact(
        cast(str, execution.outputs["bundle_uri"]), role="inference"
    ) as artifact:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(artifact.entrypoint_path), providers=["CPUExecutionProvider"]
        )
        outputs = session.run(None, {"float_input": features.astype(np.float32)})
        assert {item.name for item in session.get_outputs()} == {"scores", "threshold"}
        assert all(value.shape[0] == len(features) for value in outputs)
