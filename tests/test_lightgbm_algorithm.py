"""Tests for the official LightGBM descriptor and native model boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import lightgbm
import numpy as np
import pytest
from ray.train import Checkpoint
from tributo.algorithms.api import ResolvedAlgorithmPlan
from tributo.exporting.bundle_reader import BundleReader
from tributo_algorithms_boosting.lightgbm import export_result
from tributo_algorithms_boosting.lightgbm_contracts import LightGBMConfigValidator
from tributo_algorithms_boosting.lightgbm_descriptor import LIGHTGBM_DESCRIPTOR


def test_lightgbm_descriptor_uses_ray_train_framework_native() -> None:
    registration = LIGHTGBM_DESCRIPTOR.registration
    assert registration.spec.name == "lightgbm"
    assert registration.implementation.framework == "lightgbm"
    assert registration.distribution_spec is not None
    assert registration.distribution_spec.strategy.value == "framework_native"
    assert registration.implementation.executable_factory_ref is not None
    assert str(registration.implementation.executable_factory_ref).endswith(
        "lightgbm:create_algorithm"
    )


def test_lightgbm_config_requires_bounded_training_namespaces() -> None:
    with pytest.raises(ValueError, match="data.label_col"):
        LightGBMConfigValidator().validate(
            {"output": {"bundle_uri": "file:///tmp/model"}}
        )
    with pytest.raises(ValueError, match="ray.storage_path"):
        LightGBMConfigValidator().validate(
            {
                "data": {"label_col": "label"},
                "model": {},
                "training": {},
                "output": {"bundle_uri": "file:///tmp/model"},
                "ray": {},
            }
        )
    with pytest.raises(ValueError, match="unknown LightGBM model keys"):
        LightGBMConfigValidator().validate(
            {
                "data": {"label_col": "label"},
                "model": {"unsupported": 1},
                "output": {"bundle_uri": "file:///tmp/model"},
                "ray": {"storage_path": "/tmp/ray"},
            }
        )


@pytest.mark.parametrize(
    ("task", "labels", "model_config", "expected_outputs"),
    (
        (
            "classification",
            np.asarray([0, 0, 1, 1]),
            {"objective": "binary", "num_leaves": 3},
            {"label", "probabilities"},
        ),
        (
            "classification",
            np.asarray([0, 1, 2, 0, 1, 2]),
            {"objective": "multiclass", "num_class": 3, "num_leaves": 3},
            {"label", "probabilities"},
        ),
        (
            "regression",
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            {"objective": "regression", "num_leaves": 3},
            {"variable"},
        ),
    ),
)
def test_lightgbm_export_result_creates_runnable_bundle(
    tmp_path: Path,
    task: str,
    labels: np.ndarray,
    model_config: dict[str, Any],
    expected_outputs: set[str],
) -> None:
    features = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32
    )
    if len(labels) == 6:
        features = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ],
            dtype=np.float32,
        )
    dataset = lightgbm.Dataset(features, label=labels, free_raw_data=False)
    booster = lightgbm.train(
        {
            **model_config,
            "verbosity": -1,
            "min_data_in_leaf": 1,
            "num_threads": 1,
        },
        dataset,
        num_boost_round=3,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    booster.save_model(str(checkpoint_dir / "model.txt"))
    (checkpoint_dir / "feature_names.json").write_text(
        json.dumps(["x0", "x1"]), encoding="utf-8"
    )
    (checkpoint_dir / "model_config.json").write_text(
        json.dumps({"task": task, "class_count": 3 if len(labels) == 6 else 2}),
        encoding="utf-8",
    )
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            algorithm_config={
                "data": {"label_col": "label"},
                "output": {"bundle_uri": str(tmp_path / "bundle")},
            },
            primary_input_binding=SimpleNamespace(
                feature_names=("x0", "x1"), label_name="label"
            ),
            resolution=SimpleNamespace(algorithm="lightgbm"),
        ),
    )
    execution = export_result(
        result=object(),
        checkpoint=Checkpoint.from_directory(str(checkpoint_dir)),
        plan=plan,
        run_id="lightgbm-export-test",
    )
    with BundleReader().open_artifact(
        cast(str, execution.outputs["bundle_uri"]), role="inference"
    ) as artifact:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(artifact.entrypoint_path), providers=["CPUExecutionProvider"]
        )
        outputs = session.run(None, {"float_input": features})
        assert {item.name for item in session.get_outputs()} == expected_outputs
        assert len(outputs) == len(expected_outputs)
