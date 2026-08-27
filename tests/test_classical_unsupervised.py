"""Tests for the official unsupervised classical algorithms."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmInputError,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.input.tabular import InMemoryTabularInputView
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo_algorithms_classical.kmeans import DistributedKMeans
from tributo_algorithms_classical.kmeans_descriptor import (
    KMEANS_DESCRIPTOR,
    MINIBATCH_KMEANS_DESCRIPTOR,
)
from tributo_algorithms_classical.pca import DistributedPCA
from tributo_algorithms_classical.pca_descriptor import PCA_DESCRIPTOR
from tributo_algorithms_classical.unsupervised_exporter import (
    _kmeans_onnx,
    _onnx_metadata,
)
from tributo_algorithms_classical.unsupervised_models import KMeansModel, PCAModel


def _plan(
    feature_names: tuple[str, ...],
    config: dict[str, object] | None = None,
) -> ResolvedAlgorithmPlan:
    return cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            primary_input_binding=SimpleNamespace(
                feature_names=feature_names,
                label_name=None,
                sample_weight_name=None,
            ),
            algorithm_config=config or {},
            runtime=SimpleNamespace(distribution_digest=""),
            resolution=SimpleNamespace(algorithm="test"),
        ),
    )


def _batch() -> dict[str, np.ndarray]:
    return {
        "x0": np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
        "x1": np.asarray([0.0, 1.0, 0.0, 2.0, 1.0, 2.0]),
    }


def test_descriptors_use_dotted_entrypoint_implementations() -> None:
    assert PCA_DESCRIPTOR.name == "pca"
    assert KMEANS_DESCRIPTOR.name == "kmeans"
    assert MINIBATCH_KMEANS_DESCRIPTOR.name == "kmeans_minibatch"
    assert KMEANS_DESCRIPTOR.registration.distribution_spec.policy.exactness.value == (
        "exact"
    )
    assert (
        MINIBATCH_KMEANS_DESCRIPTOR.registration.distribution_spec.policy.exactness.value
        == "approximate"
    )
    assert PCA_DESCRIPTOR.registration.spec.input_contract_ref == (
        "tributo.official.tabular.unlabeled_dense.v1"
    )
    assert KMEANS_DESCRIPTOR.registration.spec.output_contract_ref == (
        "tributo.official.unsupervised.kmeans.clustering.v1"
    )


def test_pca_chan_welford_merge_matches_sklearn_components() -> None:
    values = _batch()
    plan = _plan(("x0", "x1"), {"n_components": 2})
    algorithm = DistributedPCA(plan)
    context = AlgorithmExecutionContext(inputs={})
    first = algorithm.map_partition(
        ({name: column[:3] for name, column in values.items()},), context
    )
    second = algorithm.map_partition(
        ({name: column[3:] for name, column in values.items()},), context
    )
    merged = algorithm.merge_states(first, second)
    model = algorithm.finalize_model(merged)

    features = np.column_stack((values["x0"], values["x1"]))
    expected = SklearnPCA(n_components=2).fit(features)
    assert model.n_samples == len(features)
    np.testing.assert_allclose(model.mean, expected.mean_)
    np.testing.assert_allclose(model.explained_variance, expected.explained_variance_)
    np.testing.assert_allclose(
        np.abs(model.components), np.abs(expected.components_), atol=1e-12
    )


def test_pca_rejects_one_row_input() -> None:
    plan = _plan(("x0", "x1"))
    algorithm = DistributedPCA(plan)
    context = AlgorithmExecutionContext(inputs={})
    state = algorithm.map_partition(
        ({"x0": np.asarray([1.0]), "x1": np.asarray([2.0])},), context
    )
    with np.testing.assert_raises(AlgorithmInputError):
        algorithm.finalize_model(state)


def test_pca_rejects_unbounded_partial_state_dimension() -> None:
    feature_names = tuple(f"x{index}" for index in range(3000))
    with np.testing.assert_raises(AlgorithmConfigurationError):
        DistributedPCA(_plan(feature_names))


def test_kmeans_converges_with_feature_only_input() -> None:
    values = _batch()
    view = InMemoryTabularInputView(
        _columns={name: tuple(column.tolist()) for name, column in values.items()},
        feature_names=("x0", "x1"),
        label_name=None,
    )
    plan = _plan(("x0", "x1"), {"n_clusters": 2, "seed": 7, "tolerance": 1e-8})
    algorithm = DistributedKMeans(plan, variant="kmeans")
    context = AlgorithmExecutionContext(inputs={"train": view})
    state = algorithm.initialize_state(plan.algorithm_config, object())
    for round_index in range(20):
        update = algorithm.compute_partition_update(
            (view.columns(),), state, round_index, context
        )
        state = algorithm.apply_update(state, update, round_index)
        metrics = algorithm.evaluate_round(state, update, round_index)
        if algorithm.should_stop(state, metrics, round_index):
            break
    model = algorithm.finalize_model(state)
    centers = model.centers[np.argsort(model.centers[:, 0])]
    np.testing.assert_allclose(
        centers,
        np.asarray([[-2.0, 1.0 / 3.0], [2.0, 5.0 / 3.0]]),
        atol=1e-12,
    )


def test_minibatch_variant_has_distinct_model_identity() -> None:
    plan = _plan(("x0", "x1"), {"n_clusters": 2, "seed": 2})
    algorithm = DistributedKMeans(plan, variant="minibatch")
    state = algorithm.initialize_state(plan.algorithm_config, object())
    assert (
        algorithm.finalize_model(
            algorithm.apply_update(
                state,
                algorithm.compute_partition_update(
                    (_batch(),), state, 0, AlgorithmExecutionContext(inputs={})
                ),
                0,
            )
        ).variant
        == "minibatch"
    )


def test_kmeans_honors_max_iter_round_limit() -> None:
    plan = _plan(("x0", "x1"), {"n_clusters": 2, "max_iter": 1, "seed": 2})
    algorithm = DistributedKMeans(plan, variant="kmeans")
    state = algorithm.initialize_state(plan.algorithm_config, object())
    update = algorithm.compute_partition_update(
        (_batch(),), state, 0, AlgorithmExecutionContext(inputs={})
    )
    state = algorithm.apply_update(state, update, 0)
    assert algorithm.should_stop(state, algorithm.evaluate_round(state, update, 0), 0)


def test_unsupervised_onnx_exports_execute_with_typed_outputs() -> None:
    import onnxruntime as ort

    plan = _plan(("x0", "x1"))
    pca = PCAModel(
        components=np.asarray([[1.0, 0.0]], dtype=np.float64),
        mean=np.asarray([1.0, 2.0], dtype=np.float64),
        explained_variance=np.asarray([1.0], dtype=np.float64),
        explained_variance_ratio=np.asarray([1.0], dtype=np.float64),
        feature_names=("x0", "x1"),
        n_samples=4,
    )
    pca_session = ort.InferenceSession(
        _onnx_metadata(pca, plan, "tributo-pca"),
        providers=["CPUExecutionProvider"],
    )
    pca_output = pca_session.run(
        None, {"float_input": np.asarray([[2.0, 5.0]], dtype=np.float32)}
    )[0]
    np.testing.assert_allclose(pca_output, np.asarray([[1.0]], dtype=np.float32))

    kmeans = KMeansModel(
        centers=np.asarray([[0.0, 0.0], [3.0, 3.0]], dtype=np.float64),
        feature_names=("x0", "x1"),
        n_iter=1,
        variant="kmeans",
    )
    kmeans_session = ort.InferenceSession(
        _kmeans_onnx(kmeans, plan),
        providers=["CPUExecutionProvider"],
    )
    labels, distances = kmeans_session.run(
        None,
        {"float_input": np.asarray([[0.5, 0.0], [2.5, 3.0]], dtype=np.float32)},
    )
    assert labels.tolist() == [0, 1]
    np.testing.assert_allclose(distances, np.asarray([0.25, 0.25], dtype=np.float32))
