"""Tests for the independent CatBoost package."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
from tributo.algorithms.api import ResolvedAlgorithmPlan
from tributo.algorithms.input.tabular import InMemoryTabularInputView
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo_algorithms_catboost.algorithm import CatBoostEnsemble, export_result
from tributo_algorithms_catboost.contracts import CatBoostConfigValidator
from tributo_algorithms_catboost.descriptor import CATBOOST_DESCRIPTOR
from tributo_algorithms_catboost.exporter import CatBoostNativeExporter
from tributo_algorithms_catboost.flavor import CatBoostNativeFlavor


def _plan() -> ResolvedAlgorithmPlan:
    return cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            primary_input_binding=SimpleNamespace(
                feature_names=("x0", "x1"),
                label_name="label",
                sample_weight_name=None,
            ),
            algorithm_config={
                "task": "classification",
                "n_estimators": 2,
                "model": {"iterations": 10, "depth": 2, "learning_rate": 0.2},
            },
        ),
    )


def test_catboost_descriptor_is_independent_and_conditional() -> None:
    assert CATBOOST_DESCRIPTOR.package_name == "tributo-algorithms-catboost"
    assert CATBOOST_DESCRIPTOR.registration.distribution_spec is not None
    assert (
        CATBOOST_DESCRIPTOR.registration.distribution_spec.policy.exactness.value
        == "conditional"
    )


def test_catboost_config_rejects_silent_model_options() -> None:
    with np.testing.assert_raises(ValueError):
        CatBoostConfigValidator().validate(
            {
                "model": {"unsupported": 1},
                "output": {"bundle_uri": "file:///tmp/model"},
            }
        )


def test_catboost_units_can_be_combined_and_predict() -> None:
    plan = _plan()
    algorithm = CatBoostEnsemble(plan)
    view = InMemoryTabularInputView(
        _columns={
            "x0": (-2.0, -1.0, 1.0, 2.0),
            "x1": (-1.0, -0.5, 0.5, 1.0),
            "label": (0, 0, 1, 1),
        },
        feature_names=("x0", "x1"),
        label_name="label",
    )
    context = AlgorithmExecutionContext(inputs={})
    units = algorithm.plan_units(plan.algorithm_config, object(), seed=3)
    fitted = tuple(algorithm.fit_unit(unit, {"train": view}, context) for unit in units)
    model = algorithm.finalize_ensemble(algorithm.merge_units(fitted))
    predictions = np.asarray(
        model.model.predict(
            np.asarray([[-2.0, -1.0], [2.0, 1.0]]), prediction_type="Class"
        )
    ).reshape(-1)
    assert predictions.tolist() == [0, 1]


def test_catboost_encodes_string_labels_in_native_runtime() -> None:
    plan = _plan()
    algorithm = CatBoostEnsemble(plan)
    view = InMemoryTabularInputView(
        _columns={
            "x0": (-2.0, -1.0, 1.0, 2.0),
            "x1": (-1.0, -0.5, 0.5, 1.0),
            "label": ("negative", "negative", "positive", "positive"),
        },
        feature_names=("x0", "x1"),
        label_name="label",
    )
    model = algorithm.fit_unit(
        algorithm.plan_units(plan.algorithm_config, object(), seed=3)[0],
        {"train": view},
        AlgorithmExecutionContext(inputs={}),
    )
    assert model.classes == ("negative", "positive")


def test_catboost_accepts_explicit_categorical_feature_names() -> None:
    plan = _plan()
    plan.primary_input_binding.feature_names = ("numeric", "category")
    plan.algorithm_config["model"]["cat_features"] = ["category"]
    algorithm = CatBoostEnsemble(plan)
    view = InMemoryTabularInputView(
        _columns={
            "numeric": (0.0, 0.1, 1.0, 1.1, 2.0, 2.1),
            "category": ("a", "a", "b", "b", "a", "b"),
            "label": (0, 0, 1, 1, 0, 1),
        },
        feature_names=("numeric", "category"),
        label_name="label",
    )
    context = AlgorithmExecutionContext(inputs={})
    unit = algorithm.plan_units(plan.algorithm_config, object(), seed=3)[0]
    model = algorithm.fit_unit(unit, {"train": view}, context)
    assert np.asarray(
        model.model.predict(
            np.asarray([[0.0, "a"], [1.0, "b"]], dtype=object),
            prediction_type="Class",
        )
    ).shape == (2,)


def test_catboost_categorical_native_bundle_preserves_object_input(
    tmp_path: Path,
) -> None:
    from tributo.exporting.runtime import BundleModelLoader

    plan = _plan()
    plan.primary_input_binding.feature_names = ("numeric", "category")
    plan.algorithm_config["model"]["cat_features"] = ["category"]
    plan.algorithm_config["output"] = {"bundle_uri": str(tmp_path / "bundle")}
    plan.resolution = SimpleNamespace(algorithm="catboost")
    algorithm = CatBoostEnsemble(plan)
    view = InMemoryTabularInputView(
        _columns={
            "numeric": (0.0, 0.1, 1.0, 1.1, 2.0, 2.1),
            "category": ("a", "a", "b", "b", "a", "b"),
            "label": (0, 0, 1, 1, 0, 1),
        },
        feature_names=("numeric", "category"),
        label_name="label",
    )
    context = AlgorithmExecutionContext(inputs={})
    unit = algorithm.plan_units(plan.algorithm_config, object(), seed=3)[0]
    model = algorithm.fit_unit(unit, {"train": view}, context)
    execution = export_result(model=model, plan=plan, run_id="catboost-category-test")
    runtime = BundleModelLoader().open(
        cast(str, execution.outputs["bundle_uri"]), role="inference", use_case="batch"
    )
    try:
        outputs = runtime.predict(
            {"float_input": np.asarray([[0.0, "a"], [1.0, "b"]], dtype=object)}
        )
    finally:
        runtime.close()
    assert outputs["label"].shape == (2,)


def test_catboost_native_exporter_and_flavor_round_trip(tmp_path: Path) -> None:
    from tributo.exporting.executor import _materialize_artifact
    from tributo.exporting.models import (
        ExportContext,
        ExportSource,
        ExportTarget,
        PlannedTarget,
        ResolvedArtifact,
    )

    plan = _plan()
    algorithm = CatBoostEnsemble(plan)
    view = InMemoryTabularInputView(
        _columns={
            "x0": (-2.0, -1.0, 1.0, 2.0),
            "x1": (-1.0, -0.5, 0.5, 1.0),
            "label": (0, 0, 1, 1),
        },
        feature_names=("x0", "x1"),
        label_name="label",
    )
    context = AlgorithmExecutionContext(inputs={})
    unit = algorithm.plan_units(plan.algorithm_config, object(), seed=3)[0]
    model = algorithm.fit_unit(unit, {"train": view}, context)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    draft = CatBoostNativeExporter().export(
        ExportContext(
            execution_id="test-execution",
            node_id="native-model",
            artifact_dir=artifact_dir,
        ),
        ExportSource(source_kind="catboost_result", model_object=model),
        {},
        PlannedTarget(
            target=ExportTarget(name="native-model", format="catboost"),
            exporter_id="catboost-native-v1",
        ),
    )
    logical = _materialize_artifact(draft, artifact_dir, "native-model")
    runtime = CatBoostNativeFlavor().load(
        ResolvedArtifact(logical, artifact_dir), role="inference"
    )
    outputs = runtime.predict(
        {"float_input": np.asarray([[-2.0, -1.0], [2.0, 1.0]], dtype=np.float32)}
    )
    assert set(outputs) == {"label", "probabilities"}
    assert outputs["label"].tolist() == [0, 1]


def test_catboost_export_result_publishes_native_bundle(tmp_path: Path) -> None:
    from tributo.exporting.bundle_reader import BundleReader

    plan = _plan()
    plan.algorithm_config["output"] = {"bundle_uri": str(tmp_path / "bundle")}
    plan.resolution = SimpleNamespace(algorithm="catboost")
    algorithm = CatBoostEnsemble(plan)
    view = InMemoryTabularInputView(
        _columns={
            "x0": (-2.0, -1.0, 1.0, 2.0),
            "x1": (-1.0, -0.5, 0.5, 1.0),
            "label": (0, 0, 1, 1),
        },
        feature_names=("x0", "x1"),
        label_name="label",
    )
    context = AlgorithmExecutionContext(inputs={})
    unit = algorithm.plan_units(plan.algorithm_config, object(), seed=3)[0]
    model = algorithm.fit_unit(unit, {"train": view}, context)
    execution = export_result(model=model, plan=plan, run_id="catboost-export-test")
    with BundleReader().open_artifact(
        cast(str, execution.outputs["bundle_uri"]), role="inference"
    ) as artifact:
        assert artifact.descriptor.format == "catboost"
        assert artifact.descriptor.flavor_id == "catboost-native-v1"
