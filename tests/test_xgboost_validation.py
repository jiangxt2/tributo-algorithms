"""Validation routing, selection contracts, and selected Bundle artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import onnxruntime
import pyarrow as pa
import pytest
import xgboost
from ray.train import Checkpoint
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    InputBinding,
    InputBindingSet,
    ResolvedAlgorithmPlan,
)
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.models import BundleRef, ExportSource
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


@pytest.mark.parametrize("metric", [None, "", ["logloss", "auc"]])
def test_stopping_requires_one_explicit_metric(metric: object) -> None:
    config = _config()
    config["model"]["eval_metric"] = metric
    with pytest.raises(ValueError, match="single model.eval_metric"):
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
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}}
            ),
        ),
        run_id="selected-model",
    )
    assert "config" not in result.metrics
    if include_metadata:
        assert result.metrics["model_rounds"] == result.metrics["best_iteration"] + 1
        assert result.metrics["model_rounds"] < result.metrics["training_rounds"]
    else:
        assert "best_iteration" not in result.metrics
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
                algorithm_config={"output": {"bundle_uri": str(tmp_path / "bundle")}}
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
    assert hashlib.sha256(booster.save_raw(raw_format="ubj")).hexdigest() == digest
