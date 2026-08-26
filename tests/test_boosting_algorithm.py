"""Tests for official out-of-tree distributed XGBoost."""

from __future__ import annotations

import subprocess
import sys

from tributo.algorithms.api import DistributionStrategy
from tributo_algorithms_boosting import XGBOOST_DESCRIPTOR, DistributedXGBoost
from tributo_algorithms_boosting.contracts import XGBoostConfigValidator
from tributo_algorithms_boosting.native import (
    OfficialXGBoostNativeFlavor,
    OfficialXGBoostNativeValidator,
    OfficialXGBoostONNXExporter,
    OfficialXGBoostUBJExporter,
)


def test_official_xgboost_owns_the_portable_algorithm_spec() -> None:
    assert XGBOOST_DESCRIPTOR.name == "xgboost"
    assert XGBOOST_DESCRIPTOR.package_name == "tributo-algorithms-boosting"
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
