"""Tests for official classical mathematical Hook implementations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tributo.algorithms.input.tabular import InMemoryTabularInputView
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo_algorithms_classical import (
    EXTRA_TREES_JOBLIB_DESCRIPTOR,
    EXTRA_TREES_NATIVE_DESCRIPTOR,
    LINEAR_REGRESSION_DESCRIPTOR,
    LOGISTIC_REGRESSION_DESCRIPTOR,
    RANDOM_FOREST_JOBLIB_DESCRIPTOR,
    RANDOM_FOREST_NATIVE_DESCRIPTOR,
)
from tributo_algorithms_classical.linear_regression import (
    DistributedLinearRegression,
)
from tributo_algorithms_classical.logistic_regression import (
    BinaryL2LogisticRegression,
)
from tributo_algorithms_classical.multinomial_nb import (
    MULTINOMIAL_NB_DESCRIPTOR,
    DistributedMultinomialNB,
)
from tributo_algorithms_classical.random_forest import (
    ExtraTreesEnsemble,
    ExtraTreesJoblibRecipe,
    RandomForestEnsemble,
)


def _view() -> InMemoryTabularInputView:
    return InMemoryTabularInputView(
        _columns={
            "x0": (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0),
            "x1": (-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0),
            "label": (0, 0, 0, 0, 1, 1, 1, 1),
        },
        feature_names=("x0", "x1"),
        label_name="label",
    )


def test_official_descriptors_use_public_v2_contracts() -> None:
    descriptors = (
        RANDOM_FOREST_JOBLIB_DESCRIPTOR,
        RANDOM_FOREST_NATIVE_DESCRIPTOR,
        EXTRA_TREES_JOBLIB_DESCRIPTOR,
        EXTRA_TREES_NATIVE_DESCRIPTOR,
        LOGISTIC_REGRESSION_DESCRIPTOR,
        LINEAR_REGRESSION_DESCRIPTOR,
        MULTINOMIAL_NB_DESCRIPTOR,
    )

    assert {descriptor.name for descriptor in descriptors} == {
        "random_forest",
        "extra_trees",
        "logistic_regression",
        "linear_regression",
        "multinomial_nb",
    }
    for descriptor in descriptors:
        assert descriptor.api_version == 2
        assert descriptor.registration.contract_bindings is not None
        assert descriptor.package_name == "tributo-algorithms-classical"


def test_random_forest_native_is_conditional_until_parity_gate() -> None:
    policy = RANDOM_FOREST_NATIVE_DESCRIPTOR.registration.distribution_spec.policy
    assert policy.exactness.value == "conditional"


def test_multinomial_nb_map_reduce_matches_single_and_two_shards() -> None:
    from tributo.algorithms.api import ResolvedAlgorithmPlan

    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            primary_input_binding=SimpleNamespace(
                feature_names=("f0", "f1"),
                label_name="label",
                sample_weight_name=None,
            ),
            algorithm_config={"alpha": 1.0, "fit_prior": True},
        ),
    )
    algorithm = DistributedMultinomialNB(plan)
    batch = {
        "f0": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "f1": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "label": np.asarray([0, 0, 1, 1]),
    }
    context = AlgorithmExecutionContext(inputs={})
    single = algorithm.finalize_model(algorithm.map_partition((batch,), context))
    left = algorithm.map_partition(
        ({name: values[:2] for name, values in batch.items()},), context
    )
    right = algorithm.map_partition(
        ({name: values[2:] for name, values in batch.items()},), context
    )
    merged = algorithm.finalize_model(algorithm.merge_states(left, right))

    np.testing.assert_allclose(
        single.estimator.feature_count_,
        merged.estimator.feature_count_,
    )
    np.testing.assert_array_equal(single.estimator.classes_, np.asarray([0, 1]))
    assert single.row_count == merged.row_count == 4


def test_native_random_forest_matches_fixed_sklearn_predictions() -> None:
    algorithm = RandomForestEnsemble()
    view = _view()
    config = {"task": "classification", "unit_count": 8}
    units = algorithm.plan_units(config, object(), seed=7)
    context = AlgorithmExecutionContext(inputs={"train": view})
    fitted = tuple(algorithm.fit_unit(unit, {"train": view}, context) for unit in units)
    model = algorithm.finalize_ensemble(algorithm.merge_units(fitted))
    columns = view.columns()
    features = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in view.feature_names]
    )
    labels = np.asarray(columns["label"])
    baseline = RandomForestClassifier(
        n_estimators=8,
        random_state=7,
        max_features="sqrt",
    ).fit(features, labels)

    np.testing.assert_array_equal(
        model.estimator.predict(features),
        baseline.predict(features),
    )
    assert float(np.mean(model.estimator.predict(features) == labels)) == 1.0


def test_extra_trees_reuses_joblib_and_parallel_unit_hooks() -> None:
    assert (
        ExtraTreesJoblibRecipe()
        .build_estimator({"n_estimators": 4, "seed": 7})
        .n_estimators
        == 4
    )
    algorithm = ExtraTreesEnsemble()
    view = _view()
    units = algorithm.plan_units({"unit_count": 4}, object(), seed=7)
    context = AlgorithmExecutionContext(inputs={"train": view})
    fitted = tuple(algorithm.fit_unit(unit, {"train": view}, context) for unit in units)
    model = algorithm.finalize_ensemble(algorithm.merge_units(fitted))
    assert len(model.estimator.estimators_) == 4


def test_linear_regression_reuses_iterative_optimization_hook() -> None:
    algorithm = DistributedLinearRegression()
    view = InMemoryTabularInputView(
        _columns={
            "x0": (-2.0, -1.0, 0.0, 1.0, 2.0),
            "x1": (1.0, 0.5, 0.0, -0.5, -1.0),
            "label": (-3.0, -1.5, 0.0, 1.5, 3.0),
        },
        feature_names=("x0", "x1"),
        label_name="label",
    )
    state = algorithm.initialize_state(
        {"feature_count": 2, "learning_rate": 0.1}, object()
    )
    context = AlgorithmExecutionContext(inputs={"train": view})
    for round_index in range(100):
        update = algorithm.compute_partition_update(
            (view.columns(),), state, round_index, context
        )
        state = algorithm.apply_update(state, update, round_index)
    model = algorithm.finalize_model(state)
    predictions = model.estimator.predict(
        np.column_stack((view.columns()["x0"], view.columns()["x1"]))
    )
    np.testing.assert_allclose(predictions, view.columns()["label"], atol=0.25)


def test_binary_logistic_hook_converges_without_ray_or_runtime_code() -> None:
    algorithm = BinaryL2LogisticRegression()
    view = _view()
    state = algorithm.initialize_state({"feature_count": 2}, object())
    context = AlgorithmExecutionContext(inputs={"train": view})
    columns = view.columns()
    left = {name: values[:4] for name, values in columns.items()}
    right = {name: values[4:] for name, values in columns.items()}
    for round_index in range(60):
        first = algorithm.compute_partition_update((left,), state, round_index, context)
        second = algorithm.compute_partition_update(
            (right,), state, round_index, context
        )
        update = algorithm.merge_updates(first, second)
        state = algorithm.apply_update(state, update, round_index)
    model = algorithm.finalize_model(state)
    features = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in view.feature_names]
    )
    labels = np.asarray(columns["label"])

    probabilities = model.estimator.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    np.testing.assert_array_equal(predictions, labels)
    assert model.estimator.coef_.shape == (1, 2)
    assert int(model.estimator.n_iter_[0]) == 60


def test_binary_logistic_matches_sklearn_l2_objective() -> None:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(17)
    features = rng.normal(size=(120, 3))
    labels = (
        rng.random(120)
        < 1.0 / (1.0 + np.exp(-(features @ np.asarray([0.4, -0.7, 0.2]) + 0.1)))
    ).astype(int)
    view = InMemoryTabularInputView(
        _columns={
            **{
                f"f{index}": tuple(float(value) for value in features[:, index])
                for index in range(3)
            },
            "label": tuple(int(value) for value in labels),
        },
        feature_names=("f0", "f1", "f2"),
        label_name="label",
    )
    algorithm = BinaryL2LogisticRegression()
    state = algorithm.initialize_state(
        {"feature_count": 3, "C": 1.0, "learning_rate": 1.0}, object()
    )
    context = AlgorithmExecutionContext(inputs={"train": view})
    for round_index in range(20):
        update = algorithm.compute_partition_update(
            (view.columns(),), state, round_index, context
        )
        state = algorithm.apply_update(state, update, round_index)
    actual = algorithm.finalize_model(state).estimator
    expected = LogisticRegression(C=1.0, solver="lbfgs", tol=1e-10, max_iter=1000).fit(
        features, labels
    )
    np.testing.assert_allclose(actual.coef_, expected.coef_, atol=1e-5)
    np.testing.assert_allclose(actual.intercept_, expected.intercept_, atol=1e-5)
    np.testing.assert_allclose(
        actual.predict_proba(features), expected.predict_proba(features), atol=1e-6
    )


def test_official_algorithm_source_contains_no_ray_or_private_core_imports() -> None:
    source_root = (
        Path(__file__).parents[1]
        / "packages"
        / "classical"
        / "src"
        / "tributo_algorithms_classical"
    )
    source = "\n".join(path.read_text() for path in source_root.glob("*.py"))

    assert "import ray" not in source
    assert "from ray" not in source
    assert "tributo.algorithms.core" not in source
    assert "tributo.integrations" not in source
