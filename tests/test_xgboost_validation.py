"""Validation routing, selection contracts, and selected Bundle artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import onnxruntime
import pandas as pd
import pyarrow as pa
import pytest
import xgboost
from ray.train import Checkpoint
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    InputBinding,
    InputBindingSet,
    QualifiedReference,
    ResolvedAlgorithmPlan,
)
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.models import (
    BundleRef,
    ExportContext,
    ExportSource,
    ExportTarget,
    PlannedTarget,
)
from tributo.exporting.runtime import BundleModelLoader
from tributo.inference.batch_predictor import XGBoostONNXPredictor
from tributo_algorithms_boosting.algorithm import DistributedXGBoost, export_result
from tributo_algorithms_boosting.contracts import (
    XGBoostConfigValidator,
    XGBoostInputValidator,
    XGBoostOutputValidator,
    _validate_training_metadata,
)
from tributo_algorithms_boosting.data_config import CompleteCoverageDataConfig
from tributo_algorithms_boosting.descriptor import XGBOOST_DESCRIPTOR
from tributo_algorithms_boosting.native import (
    OfficialXGBoostNativeFlavor,
    OfficialXGBoostNativeValidator,
    OfficialXGBoostONNXExporter,
    XGBoostValidationOptions,
    _check_training_metadata,
)


def _config() -> dict[str, Any]:
    return {
        "data": {"label_col": "label"},
        "model": {"objective": "binary:logistic", "eval_metric": "logloss"},
        "training": {"num_rounds": 20, "early_stopping_rounds": 2, "maximize": False},
        "ray": {"storage_path": "/tmp/checkpoints"},
        "output": {"bundle_uri": "/tmp/bundle"},
    }


def _evaluation_callback(
    booster: Any, *, feature_names: tuple[str, ...], options: dict[str, Any]
) -> dict[str, float]:
    assert feature_names == ("feature",)
    assert options == {"nested": {"values": [1, 2]}}
    data = xgboost.DMatrix(np.array([[-1.0], [1.0]], dtype=np.float32), label=[0, 1])
    predictions = booster.predict(data)
    score = float(predictions.mean())
    data.set_label(np.array([1, 1]))
    booster.update(data, iteration=booster.num_boosted_rounds())
    assert not np.allclose(booster.predict(data), predictions)
    options["nested"]["values"].append(3)
    return {"mean_prediction": score}


def _evaluation_failure(booster: Any, **kwargs: Any) -> dict[str, float]:
    raise RuntimeError("sensitive callback detail")


def _invalid_evaluation(booster: Any, **kwargs: Any) -> object:
    return kwargs["options"]["result"]


@pytest.fixture
def evaluation_export(tmp_path: Path) -> tuple[Any, Any, Path, Any]:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    data = xgboost.DMatrix(features, label=[0, 0, 1, 1])
    booster = xgboost.train(
        {"objective": "binary:logistic", "min_child_weight": 0, "nthread": 1},
        data,
        num_boost_round=2,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    booster.save_model(checkpoint_dir / "model.ubj")
    (checkpoint_dir / "feature_names.json").write_text(json.dumps(["feature"]))
    checkpoint = Checkpoint.from_directory(checkpoint_dir)
    return booster, checkpoint, tmp_path / "bundle", features


def _evaluation_plan(bundle: Path, evaluation: object) -> Any:
    return SimpleNamespace(
        distribution_spec=None,
        implementation=SimpleNamespace(
            exporter_ref=QualifiedReference.parse(
                "tributo_algorithms_boosting.algorithm:export_result"
            )
        ),
        primary_input_binding=SimpleNamespace(feature_names=("feature",)),
        algorithm_config={
            "output": {"bundle_uri": str(bundle), "checkpoint_evaluation": evaluation}
        },
    )


def _export_with_evaluation(checkpoint: Any, bundle: Path, evaluation: object) -> Any:
    return export_result(
        result=SimpleNamespace(metrics={}),
        checkpoint=checkpoint,
        plan=cast(ResolvedAlgorithmPlan, _evaluation_plan(bundle, evaluation)),
        run_id="checkpoint-evaluation",
    )


@pytest.mark.parametrize(
    "evaluation",
    [
        None,
        {},
        {"callback": "invalid"},
        {"callback": 1},
        {"callback": "module:function", "options": []},
        {"callback": "module:function", "unexpected": True},
    ],
)
def test_checkpoint_evaluation_rejects_invalid_configuration(
    evaluation: object,
) -> None:
    config = _config()
    config["output"]["checkpoint_evaluation"] = evaluation
    with pytest.raises(ValueError, match="checkpoint_evaluation"):
        XGBoostConfigValidator().validate(config)


def test_checkpoint_evaluation_returns_metrics_and_preserves_bundle_models(
    evaluation_export: tuple[Any, Any, Path, Any],
) -> None:
    booster, checkpoint, bundle, features = evaluation_export
    evaluation = {
        "callback": f"{__name__}:_evaluation_callback",
        "options": {"nested": {"values": [1, 2]}},
    }
    config = _config()
    config["output"]["checkpoint_evaluation"] = evaluation
    XGBoostConfigValidator().validate(config)
    from tributo.integrations.algorithm_runtimes.framework_native import (
        _framework_execution_result,
    )

    plan = _evaluation_plan(
        bundle,
        MappingProxyType(
            {
                "callback": evaluation["callback"],
                "options": MappingProxyType(
                    {"nested": MappingProxyType({"values": (1, 2)})}
                ),
            }
        ),
    )
    algorithm = SimpleNamespace(checkpoint_source=lambda result: checkpoint)
    result = _framework_execution_result(
        algorithm=algorithm,
        result=SimpleNamespace(metrics={}),
        plan=plan,
        run_id="checkpoint-evaluation",
    )
    expected = booster.predict(xgboost.DMatrix(features))
    expected_score = booster.predict(xgboost.DMatrix(features[[1, 2]])).mean()
    assert result.metrics["checkpoint_evaluation"]["mean_prediction"] == pytest.approx(
        float(expected_score)
    )
    ref = BundleRef(
        canonical_uri=result.outputs["bundle_uri"],
        bundle_id=result.outputs["bundle_id"],
        manifest_sha256=result.outputs["manifest_sha256"],
    )
    reader = BundleReader()
    for role in ("native", "inference"):
        with reader.open_artifact(ref, role=role) as artifact:
            if role == "native":
                model = OfficialXGBoostNativeFlavor().load(
                    artifact, role="native", architecture_id="xgboost"
                )
                actual = model.predict({"float_input": features})["probabilities"][:, 1]
            else:
                session = onnxruntime.InferenceSession(
                    str(artifact.entrypoint_path), providers=["CPUExecutionProvider"]
                )
                actual = session.run(None, {session.get_inputs()[0].name: features})[1][
                    :, 1
                ]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "callback",
    [
        "missing_module:callback",
        f"{__name__}:_evaluation_failure",
        f"{__name__}:_config.__name__",
    ],
)
def test_checkpoint_evaluation_callback_failure_prevents_publication(
    evaluation_export: tuple[Any, Any, Path, Any], callback: str
) -> None:
    _, checkpoint, bundle, _ = evaluation_export
    with pytest.raises(
        AlgorithmExecutionError, match="checkpoint evaluation failed"
    ) as error:
        _export_with_evaluation(checkpoint, bundle, {"callback": callback})
    assert "sensitive callback detail" not in str(error.value)
    assert not bundle.exists()


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        {"": 1},
        {"score": True},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": "1"},
    ],
)
def test_checkpoint_evaluation_invalid_metrics_prevent_publication(
    evaluation_export: tuple[Any, Any, Path, Any], result: object
) -> None:
    _, checkpoint, bundle, _ = evaluation_export
    with pytest.raises(AlgorithmExecutionError, match="evaluation metrics are invalid"):
        _export_with_evaluation(
            checkpoint,
            bundle,
            {
                "callback": f"{__name__}:_invalid_evaluation",
                "options": {"result": result},
            },
        )
    assert not bundle.exists()


def test_invalid_checkpoint_is_rejected_before_evaluation(
    evaluation_export: tuple[Any, Any, Path, Any],
) -> None:
    _, checkpoint, bundle, _ = evaluation_export
    with checkpoint.as_directory() as directory:
        (Path(directory) / "training_metadata.json").write_text("{}")
    with pytest.raises(AlgorithmExecutionError, match="training metadata is invalid"):
        _export_with_evaluation(
            checkpoint, bundle, {"callback": f"{__name__}:_evaluation_failure"}
        )
    assert not bundle.exists()


def test_checkpoint_feature_mismatch_does_not_invoke_evaluation(
    evaluation_export: tuple[Any, Any, Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, checkpoint, bundle, _ = evaluation_export
    with checkpoint.as_directory() as directory:
        (Path(directory) / "feature_names.json").write_text(json.dumps(["different"]))
    calls: list[object] = []

    def record_call(booster: object, **kwargs: Any) -> dict[str, float]:
        calls.append(booster)
        return {"score": 1.0}

    monkeypatch.setattr(sys.modules[__name__], "_evaluation_callback", record_call)
    with pytest.raises(AlgorithmExecutionError, match="checkpoint features"):
        _export_with_evaluation(
            checkpoint, bundle, {"callback": f"{__name__}:_evaluation_callback"}
        )
    assert not calls
    assert not bundle.exists()


@pytest.mark.parametrize(
    "change,error",
    [
        ({"num_rounds": True}, "positive"),
        ({"early_stopping_rounds": True}, "positive integer"),
        ({"early_stopping_rounds": 0}, "positive integer"),
        ({"early_stopping_rounds": "2"}, "positive integer"),
        ({"maximize": "min"}, "explicit"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_stopping_configuration_rejects_invalid_types(
    change: dict[str, Any], error: str
) -> None:
    config = _config()
    config["training"].update(change)
    with pytest.raises(ValueError, match=error):
        XGBoostConfigValidator().validate(config)


@pytest.mark.parametrize("metric", [None, "", [], ["logloss", ""]])
def test_stopping_requires_explicit_metrics(metric: object) -> None:
    config = _config()
    config["model"]["eval_metric"] = metric
    with pytest.raises(ValueError, match="model.eval_metric"):
        XGBoostConfigValidator().validate(config)


def test_multiple_metrics_require_a_configured_selection() -> None:
    config = _config()
    config["model"]["eval_metric"] = ["logloss", "auc"]
    with pytest.raises(ValueError, match="selection_metric"):
        XGBoostConfigValidator().validate(config)
    config["training"]["selection_metric"] = "error"
    with pytest.raises(ValueError, match="selection_metric"):
        XGBoostConfigValidator().validate(config)
    config["training"]["selection_metric"] = "logloss"
    XGBoostConfigValidator().validate(config)
    config["model"]["eval_metric"] = ["error@0.5", "auc"]
    config["training"]["selection_metric"] = "error@0.5"
    with pytest.raises(ValueError, match="parameterized selection_metric"):
        XGBoostConfigValidator().validate(config)
    config["training"]["selection_metric"] = "auc"
    XGBoostConfigValidator().validate(config)


def test_stopping_rejects_linear_booster_and_inactive_direction() -> None:
    config = _config()
    config["model"]["booster"] = "gblinear"
    with pytest.raises(ValueError, match="tree Booster"):
        XGBoostConfigValidator().validate(config)
    config = _config()
    config["training"]["early_stopping_rounds"] = None
    with pytest.raises(ValueError, match="requires early stopping"):
        XGBoostConfigValidator().validate(config)


def _bindings(validation: bool = True) -> InputBindingSet:
    return InputBindingSet(
        bindings=tuple(
            InputBinding(
                name=role,
                resolver_id="tests.memory",
                reference=role,
                feature_names=("b", "a"),
                label_name="label",
            )
            for role in (("train", "validation") if validation else ("train",))
        )
    )


@pytest.mark.parametrize("validation", [False, True])
def test_input_contract_preserves_train_and_allows_optional_validation(
    validation: bool,
) -> None:
    payload = _bindings(validation).descriptor_payload()
    assert XGBoostInputValidator().validate(payload) == payload
    policy = XGBOOST_DESCRIPTOR.registration.distribution_spec.policy
    assert [(r.role, r.mode, r.required) for r in policy.input_routes] == [
        ("train", "partitioned", True),
        ("validation", "partitioned", False),
    ]


@pytest.mark.parametrize(
    "field,value",
    [("feature_names", ["a", "b"]), ("label_name", "other"), ("name", "test")],
)
def test_input_contract_rejects_incompatible_validation(
    field: str, value: object
) -> None:
    payload = _bindings().descriptor_payload()
    payload["bindings"][1][field] = value
    with pytest.raises(ValueError):
        XGBoostInputValidator().validate(payload)


class _Dataset:
    def __init__(self, feature_type: pa.DataType = pa.float32()) -> None:
        self.schema_value = pa.schema(
            [("a", feature_type), ("b", feature_type), ("label", pa.int64())]
        )
        self.columns: list[str] = []

    def schema(self) -> pa.Schema:
        return self.schema_value

    def select_columns(self, columns: list[str]) -> _Dataset:
        self.columns = columns
        return self


def _algorithm(config: dict[str, Any], validation: bool = True) -> DistributedXGBoost:
    bindings = _bindings(validation)
    return DistributedXGBoost(
        cast(
            ResolvedAlgorithmPlan,
            SimpleNamespace(
                runtime=SimpleNamespace(resume_from=None),
                input_bindings=bindings,
                primary_input_binding=bindings.primary,
                algorithm_config=config,
            ),
        )
    )


def test_binding_projects_roles_independently_and_accepts_numeric_dtypes() -> None:
    train, validation = _Dataset(), _Dataset(pa.float64())
    datasets = {"train": train, "validation": validation}
    assert _algorithm(_config()).bind_datasets(datasets) == datasets
    assert train.columns == validation.columns == ["b", "a", "label"]


def test_binding_rejects_missing_validation_and_incompatible_schema() -> None:
    train = _Dataset()
    with pytest.raises(AlgorithmConfigurationError, match="requires validation"):
        _algorithm(_config(), validation=False).bind_datasets({"train": train})
    assert not train.columns
    with pytest.raises(AlgorithmConfigurationError, match="validation.*numeric"):
        _algorithm(_config()).bind_datasets(
            {"train": train, "validation": _Dataset(pa.string())}
        )


def test_lossless_data_config_routes_both_roles_and_copies_execution_options() -> None:
    from ray.data import ExecutionOptions

    copies: list[Any] = []

    class Dataset:
        name = "fixture"

        def __init__(self) -> None:
            self.context = SimpleNamespace(execution_options=None)

        def copy(self, _dataset: object) -> Dataset:
            value = Dataset()
            copies.append(value)
            return value

        def streaming_split(self, workers: int, **kwargs: Any) -> list[object]:
            assert workers == 2 and kwargs["equal"] is False
            return [object(), object()]

    options = ExecutionOptions(preserve_order=True)
    datasets = {"train": Dataset(), "validation": Dataset()}
    config = CompleteCoverageDataConfig(
        datasets_to_split=list(datasets), execution_options=options
    )
    assigned = config.configure(datasets, 2, None, None)
    assert [set(worker) for worker in assigned] == [set(datasets), set(datasets)]
    assert all(
        dataset.context.execution_options is None for dataset in datasets.values()
    )
    assert all(dataset.context.execution_options is not options for dataset in copies)
    assert all(dataset.context.execution_options.preserve_order for dataset in copies)


@pytest.mark.parametrize(
    "change",
    [
        {"model_rounds": 0},
        {"training_rounds": True},
        {"unknown": 1},
        {"best_score": float("nan")},
    ],
)
def test_training_metadata_is_strict_and_finite(change: dict[str, Any]) -> None:
    metadata = {
        "schema_version": 1,
        "training_rounds": 3,
        "model_rounds": 1,
        "best_iteration": 0,
        "best_score": 0.4,
        "selection_metric": "validation-logloss",
        "maximize": False,
    }
    metadata.update(change)
    with pytest.raises(ValueError):
        _validate_training_metadata(metadata)


def test_output_contract_rejects_partial_best_model_metrics() -> None:
    with pytest.raises(ValueError, match="metadata fields"):
        XGBoostOutputValidator().validate(
            {
                "status": "succeeded",
                "outputs": {"bundle_uri": "/tmp/bundle"},
                "metrics": {"training_rounds": 3, "best_score": 0.2},
            }
        )


@pytest.mark.parametrize("include_metadata", [True, False])
def test_selected_model_and_metadata_survive_both_bundle_formats(
    tmp_path: Path,
    include_metadata: bool,
) -> None:
    features = np.linspace(-2, 2, 32, dtype=np.float32).reshape(-1, 1)
    labels = (features[:, 0] > 0).astype(np.int64)
    history: dict[str, Any] = {}
    selected = xgboost.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "min_child_weight": 0,
            "nthread": 1,
        },
        xgboost.DMatrix(features, label=labels),
        num_boost_round=20,
        evals=[(xgboost.DMatrix(features, label=1 - labels), "validation")],
        evals_result=history,
        callbacks=[
            xgboost.callback.EarlyStopping(
                rounds=2, data_name="validation", maximize=False, save_best=True
            )
        ],
        verbose_eval=False,
    )
    metadata = {
        "schema_version": 1,
        "training_rounds": len(history["validation"]["logloss"]),
        "model_rounds": selected.num_boosted_rounds(),
        "best_iteration": selected.best_iteration,
        "best_score": selected.best_score,
        "selection_metric": "validation-logloss",
        "maximize": False,
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    selected.save_model(checkpoint / "model.ubj")
    (checkpoint / "feature_names.json").write_text(json.dumps(["feature"]))
    if include_metadata:
        (checkpoint / "training_metadata.json").write_text(json.dumps(metadata))
    result = export_result(
        result=SimpleNamespace(
            metrics={
                "validation-logloss": history["validation"]["logloss"][-1],
                "config": {"evidence_actor": object()},
            }
        ),
        checkpoint=Checkpoint.from_directory(checkpoint),
        plan=cast(
            ResolvedAlgorithmPlan,
            SimpleNamespace(
                algorithm_config={
                    "output": {
                        "bundle_uri": str(tmp_path / "bundle"),
                        **(
                            {
                                "checkpoint_evaluation": {
                                    "callback": f"{__name__}:_evaluation_callback",
                                    "options": {"nested": {"values": [1, 2]}},
                                }
                            }
                            if include_metadata
                            else {}
                        ),
                    }
                },
                primary_input_binding=SimpleNamespace(feature_names=("feature",)),
            ),
        ),
        run_id="selected-model",
    )
    assert "config" not in result.metrics
    if include_metadata:
        assert result.metrics["model_rounds"] == result.metrics["best_iteration"] + 1
        assert result.metrics["model_rounds"] < result.metrics["training_rounds"]
        evaluation_data = xgboost.DMatrix(np.array([[-1.0], [1.0]], dtype=np.float32))
        assert result.metrics["checkpoint_evaluation"][
            "mean_prediction"
        ] == pytest.approx(float(selected.predict(evaluation_data).mean()))
    else:
        assert "best_iteration" not in result.metrics
        assert "checkpoint_evaluation" not in result.metrics
        rejected = tmp_path / "missing-metadata-bundle"
        with pytest.raises(AlgorithmExecutionError, match="metadata is missing"):
            export_result(
                result=SimpleNamespace(metrics={}),
                checkpoint=Checkpoint.from_directory(checkpoint),
                plan=cast(
                    ResolvedAlgorithmPlan,
                    SimpleNamespace(
                        algorithm_config={
                            "output": {"bundle_uri": str(rejected)},
                            "training": {"early_stopping_rounds": 2},
                        }
                    ),
                ),
                run_id="missing-metadata",
            )
        assert not rejected.exists()
    ref = BundleRef(
        canonical_uri=result.outputs["bundle_uri"],
        bundle_id=result.outputs["bundle_id"],
        manifest_sha256=result.outputs["manifest_sha256"],
    )
    reader = BundleReader()
    expected = selected.predict(xgboost.DMatrix(features))
    for role in ("native", "inference"):
        with reader.open_artifact(ref, role=role) as artifact:
            metadata_path = artifact.path_for("training_metadata.json")
            if include_metadata:
                assert json.loads(metadata_path.read_text()) == metadata
            else:
                assert not metadata_path.is_file()
            if role == "native":
                drifted_source = ExportSource(
                    source_kind="xgboost_result",
                    model_object=selected,
                    metadata={
                        "training_metadata": {
                            **metadata,
                            "training_rounds": metadata["training_rounds"] + 1,
                        }
                    },
                )
                failed = OfficialXGBoostNativeValidator().validate(
                    drifted_source, artifact, {}, XGBoostValidationOptions()
                )
                assert failed.status == "failed"
                assert failed.failure is not None
                assert "metadata" in failed.failure.message
                model = OfficialXGBoostNativeFlavor().load(
                    artifact, role="native", architecture_id="xgboost"
                )
                actual = model.predict({"float_input": features})["probabilities"][:, 1]
            else:
                session = onnxruntime.InferenceSession(
                    str(artifact.entrypoint_path), providers=["CPUExecutionProvider"]
                )
                actual = session.run(None, {session.get_inputs()[0].name: features})[1][
                    :, 1
                ]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="model rounds drifted"):
        _check_training_metadata(
            selected, {"schema_version": 1, "training_rounds": 5, "model_rounds": 5}
        )


@pytest.mark.parametrize("objective", ["multi:softprob", "reg:squarederror"])
def test_explicit_onnx_task_preserves_multiclass_and_regression(
    tmp_path: Path, objective: str
) -> None:
    features = np.linspace(-2, 2, 32, dtype=np.float32).reshape(-1, 1)
    classification = objective.startswith("multi:")
    labels = (
        np.digitize(features[:, 0], [-0.5, 0.5])
        if classification
        else features[:, 0] ** 2
    )
    booster = xgboost.train(
        {
            "objective": objective,
            "nthread": 1,
            **({"num_class": 3} if classification else {}),
        },
        xgboost.DMatrix(features, label=labels),
        num_boost_round=2,
    )
    digest = hashlib.sha256(booster.save_raw(raw_format="ubj")).hexdigest()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    booster.save_model(checkpoint / "model.ubj")
    (checkpoint / "feature_names.json").write_text(json.dumps(["f0"]))
    result = export_result(
        result=SimpleNamespace(metrics={}),
        checkpoint=Checkpoint.from_directory(checkpoint),
        plan=cast(
            ResolvedAlgorithmPlan,
            SimpleNamespace(
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}},
                primary_input_binding=SimpleNamespace(feature_names=("f0",)),
            ),
        ),
        run_id="task-parity",
    )
    ref = BundleRef(
        canonical_uri=result.outputs["bundle_uri"],
        bundle_id=result.outputs["bundle_id"],
        manifest_sha256=result.outputs["manifest_sha256"],
    )
    with BundleReader().open_artifact(ref, role="inference") as artifact:
        session = onnxruntime.InferenceSession(
            str(artifact.entrypoint_path), providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {session.get_inputs()[0].name: features})[
            int(classification)
        ]
    expected = booster.predict(xgboost.DMatrix(features))
    np.testing.assert_allclose(
        actual.reshape(expected.shape), expected, rtol=1e-6, atol=1e-7
    )
    if not classification:
        with BundleModelLoader().open(ref.canonical_uri) as runtime:
            predictions = runtime.predict({"float_input": features})
            assert set(predictions) == {"prediction"}
            np.testing.assert_allclose(
                predictions["prediction"].reshape(expected.shape),
                expected,
                rtol=1e-6,
                atol=1e-7,
            )
    assert hashlib.sha256(booster.save_raw(raw_format="ubj")).hexdigest() == digest


@pytest.mark.parametrize(
    ("base_score", "min_child_weight"),
    [
        pytest.param(0.0, 0, id="zero"),
        pytest.param(1.0, 0, id="one"),
        pytest.param(1e-40, 0, id="near-zero"),
        pytest.param(0.99999994, 0, id="near-one"),
        pytest.param(0.2, 0, id="interior"),
        pytest.param(0.0, 100, id="zero-leaf"),
        pytest.param(1.0, 100, id="one-leaf"),
    ],
)
def test_onnx_binary_predictions_match_native_base_score(
    tmp_path: Path, base_score: float, min_child_weight: int
) -> None:
    features = np.linspace(-3, 3, 64, dtype=np.float32).reshape(-1, 1)
    matrix = xgboost.DMatrix(features, label=(features[:, 0] > 0).astype(np.int64))
    booster = xgboost.train(
        {
            "objective": "binary:logistic",
            "base_score": base_score,
            "min_child_weight": min_child_weight,
            "nthread": 1,
        },
        matrix,
        num_boost_round=4,
    )
    original = bytes(booster.save_raw(raw_format="ubj"))
    OfficialXGBoostONNXExporter().export(
        ExportContext(execution_id="test", node_id="onnx", artifact_dir=tmp_path),
        ExportSource(
            source_kind="xgboost_result",
            model_object=booster,
            feature_schema={"feature_names": ["feature"]},
        ),
        {},
        PlannedTarget(
            target=ExportTarget(name="onnx", format="onnx"),
            exporter_id="official-xgboost-onnx-v1",
        ),
    )
    session = onnxruntime.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    labels, probabilities = session.run(None, {"float_input": features})
    expected = booster.predict(matrix)
    np.testing.assert_allclose(probabilities[:, 1], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(labels, (expected > 0.5).astype(np.int64))
    assert XGBoostONNXPredictor._load_feature_names(tmp_path / "model.onnx") == [
        "feature"
    ]
    assert bytes(booster.save_raw(raw_format="ubj")) == original


@pytest.mark.parametrize(
    ("sidecar", "binding", "model_names"),
    [
        pytest.param(["b", "a"], ("a", "b"), None, id="sidecar-order"),
        pytest.param(["a"], ("a",), None, id="model-count"),
        pytest.param(["a", "b"], ("a", "b"), ["b", "a"], id="model-order"),
        pytest.param("invalid", ("a", "b"), None, id="malformed-sidecar"),
    ],
)
def test_export_rejects_checkpoint_feature_mismatch(
    tmp_path: Path,
    sidecar: object,
    binding: tuple[str, ...],
    model_names: list[str] | None,
) -> None:
    features = np.array([[0, 1], [1, 0]], dtype=np.float32)
    booster = xgboost.train(
        {"objective": "reg:squarederror", "nthread": 1},
        xgboost.DMatrix(features, label=[0, 1], feature_names=model_names),
        num_boost_round=1,
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    booster.save_model(checkpoint / "model.ubj")
    (checkpoint / "feature_names.json").write_text(json.dumps(sidecar))
    bundle = tmp_path / "bundle"
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            algorithm_config={"output": {"bundle_uri": str(bundle)}},
            primary_input_binding=SimpleNamespace(feature_names=binding),
        ),
    )
    with pytest.raises(AlgorithmExecutionError, match="checkpoint features"):
        export_result(
            result=SimpleNamespace(metrics={}),
            checkpoint=Checkpoint.from_directory(checkpoint),
            plan=plan,
            run_id="feature-mismatch",
        )
    assert not bundle.exists()


@pytest.mark.parametrize("base_score", [0.0, 1.0, 0.5])
def test_onnx_bundle_discovers_training_feature_order(
    tmp_path: Path, base_score: float
) -> None:
    first = np.linspace(-3, 3, 64, dtype=np.float32)
    second = np.zeros_like(first)
    features = np.column_stack((first, second))
    booster = xgboost.train(
        {
            "objective": "binary:logistic",
            "base_score": base_score,
            "min_child_weight": 0,
            "nthread": 1,
        },
        xgboost.DMatrix(features, label=(first > 0).astype(np.int64)),
        num_boost_round=4,
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    booster.save_model(checkpoint / "model.ubj")
    (checkpoint / "feature_names.json").write_text(json.dumps(["first", "second"]))
    bundle_uri = str(tmp_path / "bundle")
    result = export_result(
        result=SimpleNamespace(metrics={}),
        checkpoint=Checkpoint.from_directory(checkpoint),
        plan=cast(
            ResolvedAlgorithmPlan,
            SimpleNamespace(
                algorithm_config={"output": {"bundle_uri": bundle_uri}},
                primary_input_binding=SimpleNamespace(
                    feature_names=("first", "second")
                ),
            ),
        ),
        run_id="feature-order",
    )
    bundle_uri = result.outputs["bundle_uri"]

    assert XGBoostONNXPredictor.get_feature_names(bundle_uri=bundle_uri) == [
        "first",
        "second",
    ]
    automatic = XGBoostONNXPredictor(bundle_uri=bundle_uri)
    explicit = XGBoostONNXPredictor(
        bundle_uri=bundle_uri, predictor_config={"feature_names": ["first", "second"]}
    )
    table = pd.DataFrame({"first": first, "second": second})

    def batch(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {name: frame[name].to_numpy() for name in frame.columns}

    try:
        expected = explicit(batch(table))["prediction"][:, 1]
        for frame in (table, table[["second", "first"]]):
            actual = automatic(batch(frame))["prediction"][:, 1]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    finally:
        automatic.close()
        explicit.close()
    native = booster.predict(xgboost.DMatrix(features))
    swapped = booster.predict(xgboost.DMatrix(features[:, ::-1]))
    assert not np.allclose(native, swapped)
    np.testing.assert_allclose(expected, native, rtol=1e-6, atol=1e-7)


def test_onnx_export_does_not_advertise_generated_feature_names(tmp_path: Path) -> None:
    features = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    booster = xgboost.train(
        {"objective": "binary:logistic", "nthread": 1},
        xgboost.DMatrix(features, label=[0, 1]),
        num_boost_round=1,
    )
    OfficialXGBoostONNXExporter().export(
        ExportContext(execution_id="test", node_id="onnx", artifact_dir=tmp_path),
        ExportSource(source_kind="xgboost_result", model_object=booster),
        {},
        PlannedTarget(
            target=ExportTarget(name="onnx", format="onnx"),
            exporter_id="official-xgboost-onnx-v1",
        ),
    )
    assert XGBoostONNXPredictor._load_feature_names(tmp_path / "model.onnx") == []
