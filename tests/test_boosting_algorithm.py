"""Tests for official out-of-tree distributed XGBoost."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xgboost
from tributo.algorithms.api import DistributionStrategy
from tributo.exceptions import ModelLoadError
from tributo.exporting.models import (
    ArtifactFile,
    LogicalArtifact,
    ProducerInfo,
    ResolvedArtifact,
)
from tributo_algorithms_boosting import XGBOOST_DESCRIPTOR, DistributedXGBoost
from tributo_algorithms_boosting.contracts import XGBoostConfigValidator
from tributo_algorithms_boosting.native import (
    OfficialXGBoostNativeFlavor,
    OfficialXGBoostNativeValidator,
    OfficialXGBoostONNXExporter,
    OfficialXGBoostUBJExporter,
)


def _native_xgboost_artifact(
    tmp_path: Path,
    booster: xgboost.Booster,
    feature_names: list[str] | None,
) -> ResolvedArtifact:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "model.ubj"
    booster.save_model(model_path)
    paths = [("model.ubj", "model")]
    if feature_names is not None:
        sidecar_path = tmp_path / "feature_names.json"
        sidecar_path.write_text(json.dumps(feature_names), encoding="utf-8")
        paths.append(("feature_names.json", "config"))
    files = tuple(
        ArtifactFile(
            relative_path=name,
            sha256=hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
            size_bytes=(tmp_path / name).stat().st_size,
            role=role,
        )
        for name, role in paths
    )
    descriptor = LogicalArtifact(
        name="native-model",
        format="ubj",
        flavor_id="official-xgboost-native-v1",
        entrypoint="model.ubj",
        files=files,
        tree_digest=LogicalArtifact.compute_tree_digest(files),
        producer=ProducerInfo(exporter_id="official-xgboost-ubj-v1"),
    )
    return ResolvedArtifact(descriptor, tmp_path)


def _train_xgboost_booster(
    feature_names: list[str] | None,
) -> tuple[xgboost.Booster, np.ndarray]:
    features = np.asarray(
        [
            [-2.0, -1.0],
            [-1.0, -0.5],
            [-0.5, -1.0],
            [-0.2, 0.0],
            [0.2, 0.0],
            [0.5, 1.0],
            [1.0, 0.5],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    booster = xgboost.train(
        {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": 2,
            "nthread": 1,
            "seed": 0,
        },
        xgboost.DMatrix(features, label=labels, feature_names=feature_names),
        num_boost_round=2,
    )
    return booster, features


def test_official_xgboost_owns_the_portable_algorithm_spec() -> None:
    assert XGBOOST_DESCRIPTOR.name == "xgboost"
    assert XGBOOST_DESCRIPTOR.package_name == "tributo-algorithms"
    distribution = XGBOOST_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.FRAMEWORK_NATIVE
    assert issubclass(DistributedXGBoost, object)


def test_xgboost_wheel_owns_multi_format_delivery_plugins() -> None:
    assert OfficialXGBoostONNXExporter.exporter_id == "official-xgboost-onnx-v1"
    assert OfficialXGBoostUBJExporter.exporter_id == "official-xgboost-ubj-v1"
    assert (
        OfficialXGBoostNativeValidator.validator_id
        == "official-xgboost-native-runtime-v1"
    )
    assert OfficialXGBoostNativeFlavor.flavor_id == "official-xgboost-native-v1"


@pytest.mark.parametrize("booster_names", [None, ["x0", "x1"]])
def test_native_ubj_flavor_predicts_with_feature_names(
    tmp_path: Path, booster_names: list[str] | None
) -> None:
    names = ["x0", "x1"]
    booster, features = _train_xgboost_booster(booster_names)
    artifact = _native_xgboost_artifact(tmp_path, booster, names)

    model = OfficialXGBoostNativeFlavor().load(
        artifact, role="native", architecture_id="xgboost"
    )
    actual = model.predict({"float_input": features})

    expected_booster = xgboost.Booster()
    expected_booster.load_model(str(artifact.entrypoint_path))
    expected_booster.feature_names = names
    expected = expected_booster.predict(xgboost.DMatrix(features, feature_names=names))
    np.testing.assert_allclose(actual["probabilities"][:, 1], expected)
    expected_labels = np.column_stack((1.0 - expected, expected)).argmax(axis=1)
    np.testing.assert_array_equal(actual["label"], expected_labels)


def test_native_ubj_flavor_preserves_feature_metadata_validation(
    tmp_path: Path,
) -> None:
    names = ["x0", "x1"]
    booster, features = _train_xgboost_booster(names)
    drifted_artifact = _native_xgboost_artifact(tmp_path, booster, ["x0", "other"])

    with pytest.raises(ModelLoadError, match="feature-name sidecar drifted"):
        OfficialXGBoostNativeFlavor().load(
            drifted_artifact, role="native", architecture_id="xgboost"
        )

    valid_artifact = _native_xgboost_artifact(tmp_path / "valid", booster, names)
    model = OfficialXGBoostNativeFlavor().load(
        valid_artifact, role="native", architecture_id="xgboost"
    )
    with pytest.raises(ValueError):
        model.predict({"float_input": np.zeros((1, 3), dtype=np.float32)})


def test_native_ubj_flavor_predicts_without_feature_metadata(tmp_path: Path) -> None:
    booster, features = _train_xgboost_booster(None)
    artifact = _native_xgboost_artifact(tmp_path, booster, None)

    model = OfficialXGBoostNativeFlavor().load(
        artifact, role="native", architecture_id="xgboost"
    )
    actual = model.predict({"float_input": features})
    expected = booster.predict(xgboost.DMatrix(features))

    np.testing.assert_allclose(actual["probabilities"][:, 1], expected)


def test_xgboost_config_contract_requires_data_runtime_and_output() -> None:
    value = {
        "data": {"label_col": "label", "feature_columns": ["x0", "x1"]},
        "model": {"objective": "binary:logistic"},
        "training": {"num_rounds": 2},
        "ray": {"storage_path": "/tmp/xgboost"},
        "output": {"bundle_uri": "/tmp/xgboost-bundle"},
    }
    assert XGBoostConfigValidator().validate(value) == value


def test_xgboost_wheel_owns_native_tree_attribution_math() -> None:
    code = """
from types import SimpleNamespace
import numpy as np
import xgboost
from tributo_algorithms_boosting.native import _XGBoostModel
features = np.asarray([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
labels = np.asarray([0, 0, 1], dtype=np.float32)
booster = xgboost.train(
    {"objective": "binary:logistic", "max_depth": 2},
    xgboost.DMatrix(features, label=labels, feature_names=["x0", "x1"]),
    num_boost_round=2,
)
model = _XGBoostModel(booster, xgboost)
request = SimpleNamespace(output_target="raw", output_selection="all")
assert model.native_attribution_support(request).supported is True
prepared = model.prepare_native_attribution(
    request, feature_names=("x0", "x1"), reference_data=None
)
explanation = prepared.explain(features)
assert explanation.values.shape == (3, 2, 1)
np.testing.assert_allclose(
    explanation.values.sum(axis=1) + explanation.base_values,
    explanation.model_outputs,
    rtol=1e-5,
    atol=1e-6,
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
