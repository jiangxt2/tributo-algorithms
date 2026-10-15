"""Behavior tests for the shared bounded XGBoost Worker loop."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import ray
import xgboost
from ray import train
from tributo.algorithms.api import AlgorithmExecutionError
from tributo_algorithms_boosting import algorithm, stages
from tributo_algorithms_boosting.checkpointing import REPORT_KEY


class _Shard:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict[str, Any]] = []
        self.batch_sizes: list[int] = []

    def iter_batches(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        yield self.frame.iloc[:0]
        for start in range(0, len(self.frame), kwargs["batch_size"]):
            batch = self.frame.iloc[start : start + kwargs["batch_size"]]
            self.batch_sizes.append(len(batch))
            yield batch


def _worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frame: pd.DataFrame | None,
    *,
    rank: int = 0,
    validation: _Shard | None = None,
) -> tuple[_Shard | None, list[dict[str, Any]], list[dict[str, Any]]]:
    shard = _Shard(frame) if frame is not None else None
    evidence: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    monkeypatch.setattr(algorithm, "_XGBOOST_BATCH_ROWS", 2)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        train,
        "get_dataset_shard",
        lambda name: shard if name == "train" else validation,
    )
    monkeypatch.setattr(
        train,
        "get_context",
        lambda: SimpleNamespace(get_world_rank=lambda: rank, get_world_size=lambda: 2),
    )
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(
            get_worker_id=lambda: f"worker-{rank}",
            get_node_id=lambda: f"node-{rank}",
            get_assigned_resources=lambda: {"CPU": 1.0},
        ),
    )
    monkeypatch.setattr(ray, "get", lambda value: value)

    def report(metrics: dict[str, Any], *, checkpoint: Any = None) -> None:
        item = {"metrics": metrics, "checkpoint": None}
        if checkpoint is not None:
            with checkpoint.as_directory() as directory:
                root = Path(directory)
                item["checkpoint"] = {
                    "model": (root / "model.ubj").read_bytes(),
                    "features": json.loads(
                        (root / "feature_names.json").read_text(encoding="utf-8")
                    ),
                    "training": json.loads(
                        (root / "training_metadata.json").read_text(encoding="utf-8")
                    ),
                }
        reports.append(item)

    monkeypatch.setattr(train, "report", report)
    return shard, evidence, reports


def _config(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "feature_names": ["b", "a"],
        "label_name": "label",
        "params": {"objective": "binary:logistic", "seed": 7, "nthread": 1},
        "num_boost_round": 3,
        "evidence_actor": SimpleNamespace(
            record=SimpleNamespace(remote=evidence.append)
        ),
        "binding_digest": "binding",
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6],
            "b": [6, 5, 4, 3, 2, 1, 0],
            "label": [0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_worker_batches_preserve_rows_order_checkpoint_and_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frame = _frame()
    shard, evidence, reports = _worker(monkeypatch, tmp_path, frame)
    config = _config(evidence)
    config["params"]["min_child_weight"] = 0
    algorithm._train_loop(config)

    assert shard is not None
    assert shard.calls
    assert all(
        call == {"batch_format": "pandas", "batch_size": 2, "prefetch_batches": 1}
        for call in shard.calls
    )
    assert max(shard.batch_sizes) == 2
    assert 1 in shard.batch_sizes
    assert len(evidence) == len(reports) == 1
    worker = evidence[0]
    assert worker["input_rows"] == {"train": 7}
    assert worker["rows_processed"] == 7
    assert worker["batch_count"] == worker["collective_steps"] == 3
    assert worker["rank"] == 0 and worker["world_size"] == 2
    assert worker["shard_id"] == hashlib.sha256(b"binding:0/2").hexdigest()
    checkpoint = reports[0]["checkpoint"]
    assert checkpoint["features"] == ["b", "a"]
    assert (
        hashlib.sha256(checkpoint["model"]).hexdigest() == worker["model_state_digest"]
    )
    assert reports[0]["metrics"]["training_rounds"] == 3
    restored = xgboost.Booster()
    restored.load_model(bytearray(checkpoint["model"]))
    matrix = xgboost.DMatrix(frame[["b", "a"]].to_numpy(), label=frame["label"])
    baseline = xgboost.train(config["params"], matrix, num_boost_round=3)
    predictions = restored.predict(matrix)
    assert np.ptp(predictions) > 0
    np.testing.assert_allclose(predictions, baseline.predict(matrix))
    swapped = xgboost.DMatrix(frame[["a", "b"]].to_numpy())
    assert not np.allclose(predictions, restored.predict(swapped))
    assert list(tmp_path.iterdir()) == []


def test_nonzero_rank_reports_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, evidence, reports = _worker(monkeypatch, tmp_path, _frame(), rank=1)
    algorithm._train_loop(_config(evidence))
    assert evidence[0]["rank"] == 1
    assert evidence[0]["input_rows"] == {"train": 7}
    assert reports[0]["checkpoint"] is None
    assert (
        reports[0]["metrics"]["model_state_digest"] == evidence[0]["model_state_digest"]
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5, "3"])
def test_worker_rejects_invalid_rounds_before_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rounds: object
) -> None:
    shard, evidence, reports = _worker(monkeypatch, tmp_path, _frame())
    config = {**_config(evidence), "num_boost_round": rounds}
    with pytest.raises(AlgorithmExecutionError, match="num_boost_round"):
        algorithm._train_loop(config)
    assert shard is not None and not shard.calls
    assert not evidence and not reports


@pytest.mark.parametrize("missing", [True, False])
def test_worker_rejects_missing_or_empty_shard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: bool
) -> None:
    frame = None if missing else _frame().iloc[:0]
    _, evidence, reports = _worker(monkeypatch, tmp_path, frame)
    with pytest.raises(AlgorithmExecutionError, match="shard is (missing|empty)"):
        algorithm._train_loop(_config(evidence))
    assert not evidence and not reports
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("column", ["a", "label"])
def test_worker_rejects_missing_columns_without_reporting_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, column: str
) -> None:
    _, evidence, reports = _worker(monkeypatch, tmp_path, _frame().drop(columns=column))
    with pytest.raises(KeyError):
        algorithm._train_loop(_config(evidence))
    assert not evidence and not reports
    assert list(tmp_path.iterdir()) == []


def test_stage_runner_uses_the_same_worker_loop() -> None:
    assert stages._train_loop is algorithm._train_loop


@pytest.mark.parametrize("rounds,patience", [(20, 2), (3, 50)])
def test_validation_selects_a_physical_best_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rounds: int, patience: int
) -> None:
    validation_frame = _frame().iloc[:5].copy()
    validation_frame["label"] = 1 - validation_frame["label"]
    validation = _Shard(validation_frame)
    shard, evidence, reports = _worker(
        monkeypatch, tmp_path, _frame(), validation=validation
    )
    config = _config(evidence)
    config.update(
        input_roles=["train", "validation"],
        num_boost_round=rounds,
        early_stopping_rounds=patience,
        maximize=False,
    )
    config["params"].update(eval_metric="logloss", min_child_weight=0, max_depth=1)
    algorithm._train_loop(config)

    assert shard is not None and shard.calls and validation.calls
    assert max(validation.batch_sizes) == 2 and 1 in validation.batch_sizes
    assert len(evidence) == len(reports) == 1
    assert evidence[0]["input_rows"] == {"train": 7, "validation": 5}
    metadata = reports[0]["checkpoint"]["training"]
    assert metadata["model_rounds"] == metadata["best_iteration"] + 1
    assert metadata["model_rounds"] < metadata["training_rounds"] <= rounds
    assert evidence[0]["collective_steps"] == metadata["training_rounds"]
    assert metadata["selection_metric"] == "validation-logloss"
    assert metadata["maximize"] is False
    assert (
        reports[0]["metrics"]["best_score"]
        < reports[0]["metrics"]["validation-logloss"]
    )
    if patience > rounds:
        assert metadata["training_rounds"] == rounds

    restored = xgboost.Booster()
    restored.load_model(bytearray(reports[0]["checkpoint"]["model"]))
    assert restored.num_boosted_rounds() == metadata["model_rounds"]
    matrix = xgboost.DMatrix(_frame()[["b", "a"]].to_numpy(), label=_frame()["label"])
    last = xgboost.train(
        config["params"], matrix, num_boost_round=metadata["training_rounds"]
    )
    np.testing.assert_allclose(
        restored.predict(matrix),
        last.predict(matrix, iteration_range=(0, metadata["model_rounds"])),
    )
    assert not np.allclose(restored.predict(matrix), last.predict(matrix))
    assert list(tmp_path.iterdir()) == []


def test_validation_is_evaluated_without_entering_training_gradients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    validation_frame = _frame().copy()
    validation_frame["label"] = 1 - validation_frame["label"]
    _, evidence, reports = _worker(
        monkeypatch, tmp_path, _frame(), validation=_Shard(validation_frame)
    )
    config = _config(evidence)
    config["input_roles"] = ["train", "validation"]
    config["params"]["min_child_weight"] = 0
    algorithm._train_loop(config)
    restored = xgboost.Booster()
    restored.load_model(bytearray(reports[0]["checkpoint"]["model"]))
    matrix = xgboost.DMatrix(_frame()[["b", "a"]].to_numpy(), label=_frame()["label"])
    baseline = xgboost.train(config["params"], matrix, num_boost_round=3)
    np.testing.assert_allclose(restored.predict(matrix), baseline.predict(matrix))
    assert (
        reports[0]["metrics"]["validation-logloss"]
        > reports[0]["metrics"]["train-logloss"]
    )
    assert "best_iteration" not in reports[0]["metrics"]


def test_worker_uses_explicit_metric_direction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    validation_frame = _frame().copy()
    validation_frame["label"] = 1 - validation_frame["label"]
    _, evidence, reports = _worker(
        monkeypatch, tmp_path, _frame(), validation=_Shard(validation_frame)
    )
    config = _config(evidence)
    config.update(
        input_roles=["train", "validation"],
        num_boost_round=4,
        early_stopping_rounds=1,
        maximize=True,
    )
    config["params"].update(eval_metric="logloss", min_child_weight=0)
    algorithm._train_loop(config)
    metadata = reports[0]["checkpoint"]["training"]
    assert metadata["maximize"] is True
    assert metadata["best_iteration"] == 3
    assert metadata["model_rounds"] == metadata["training_rounds"] == 4
    assert metadata["best_score"] == reports[0]["metrics"]["validation-logloss"]


def test_worker_rejects_early_stopping_without_validation_before_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shard, evidence, reports = _worker(monkeypatch, tmp_path, _frame())
    config = _config(evidence)
    config.update(early_stopping_rounds=2, maximize=False)
    config["params"]["eval_metric"] = "logloss"
    with pytest.raises(AlgorithmExecutionError, match="requires validation"):
        algorithm._train_loop(config)
    assert shard is not None and not shard.calls
    assert not evidence and not reports


@pytest.mark.parametrize("validation", [None, _frame().iloc[:0]])
def test_worker_rejects_missing_or_empty_validation_without_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, validation: pd.DataFrame | None
) -> None:
    _, evidence, reports = _worker(
        monkeypatch,
        tmp_path,
        _frame(),
        validation=_Shard(validation) if validation is not None else None,
    )
    config = _config(evidence)
    config["input_roles"] = ["train", "validation"]
    with pytest.raises(
        AlgorithmExecutionError, match="validation.*shard is (missing|empty)"
    ):
        algorithm._train_loop(config)
    assert not evidence and not reports
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("early", [False, True])
def test_worker_resumes_saved_progress_and_native_patience(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, early: bool
) -> None:
    import shutil

    from ray.train import Checkpoint

    frame = _frame()
    validation_frame = frame.copy()
    validation_frame["label"] = 1 - validation_frame["label"]
    _, evidence, _ = _worker(
        monkeypatch, tmp_path, frame, validation=_Shard(validation_frame)
    )
    config = _config(evidence)
    config.update(
        input_roles=["train", "validation"],
        num_boost_round=20,
        checkpoint_interval_rounds=3 if early else 10,
        invocation="same-fit",
        plan_id="frozen-plan",
    )
    config["params"].update(min_child_weight=0, max_depth=1, eval_metric="logloss")
    if early:
        config.update(early_stopping_rounds=4, maximize=False)
    monkeypatch.setattr(train, "get_checkpoint", lambda: None)
    saved = tmp_path / "committed"

    class Failure(RuntimeError):
        pass

    def fail_after_commit(metrics: Any, *, checkpoint: Any = None) -> None:
        assert metrics[REPORT_KEY]["phase"] == "training"
        with checkpoint.as_directory() as directory:
            shutil.copytree(directory, saved)
        raise Failure("committed Worker failure")

    monkeypatch.setattr(train, "report", fail_after_commit)
    with pytest.raises(Failure):
        algorithm._train_loop(config)
    assert not evidence
    reports: list[dict[str, Any]] = []

    def capture(metrics: Any, *, checkpoint: Any = None) -> None:
        if metrics[REPORT_KEY]["phase"] == "final":
            with checkpoint.as_directory() as directory:
                raw = (Path(directory) / "model.ubj").read_bytes()
                full = (Path(directory) / "training.ubj").read_bytes()
            reports.append({"metrics": metrics, "raw": raw, "full": full})

    monkeypatch.setattr(
        train, "get_checkpoint", lambda: Checkpoint.from_directory(saved)
    )
    monkeypatch.setattr(train, "report", capture)
    algorithm._train_loop(config)
    assert len(reports) == 1 and not evidence
    metrics = reports[0]["metrics"]
    start = 3 if early else 10
    assert metrics["resumed_from_round"] == start
    expected_rounds = 5 if early else 20
    assert metrics["training_rounds"] == expected_rounds
    assert metrics["attempt_training_rounds"] == expected_rounds - start
    worker = metrics[REPORT_KEY]["worker"]
    assert worker["input_rows"] == {"train": 7, "validation": 7}
    assert (
        worker["batch_count"] == worker["collective_steps"] == expected_rounds - start
    )
    full = xgboost.Booster(model_file=bytearray(reports[0]["full"]))
    assert full.num_boosted_rounds() == expected_rounds
    matrix = xgboost.DMatrix(frame[["b", "a"]].to_numpy(), label=frame["label"])
    validation = xgboost.DMatrix(
        validation_frame[["b", "a"]].to_numpy(), label=validation_frame["label"]
    )
    native = xgboost.train(
        config["params"],
        matrix,
        num_boost_round=20,
        evals=[(matrix, "train"), (validation, "validation")],
        callbacks=[
            xgboost.callback.EarlyStopping(
                rounds=4, data_name="validation", maximize=False, save_best=True
            )
        ]
        if early
        else [],
        verbose_eval=False,
    )
    restored = xgboost.Booster(model_file=bytearray(reports[0]["raw"]))
    assert restored.num_boosted_rounds() == native.num_boosted_rounds()
    np.testing.assert_allclose(restored.predict(matrix), native.predict(matrix))
    if early:
        assert metrics["best_iteration"] == native.best_iteration == 0
        assert metrics["best_score"] == native.best_score


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("phase", ["training", "final"])
def test_worker_restores_final_checkpoint_without_training_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, early: bool, phase: str
) -> None:
    import shutil

    from ray.train import Checkpoint

    frame = _frame()
    validation = frame.copy()
    validation["label"] = 1 - validation["label"]
    _, evidence, _ = _worker(
        monkeypatch, tmp_path, frame, validation=_Shard(validation)
    )
    config = _config(evidence)
    config.update(
        input_roles=["train", "validation"],
        checkpoint_interval_rounds=3,
        invocation="same-fit",
        plan_id="frozen-plan",
        num_boost_round=20 if early else 3,
    )
    config["params"].update(min_child_weight=0, max_depth=1, eval_metric="logloss")
    if early:
        config.update(early_stopping_rounds=2, maximize=False)
    monkeypatch.setattr(train, "get_checkpoint", lambda: None)
    saved = tmp_path / "final"
    initial: list[dict[str, Any]] = []

    def capture_final(metrics: Any, *, checkpoint: Any = None) -> None:
        if metrics[REPORT_KEY]["phase"] == phase:
            with checkpoint.as_directory() as directory:
                shutil.copytree(directory, saved)
        if metrics[REPORT_KEY]["phase"] == "final":
            initial.append(metrics)

    monkeypatch.setattr(train, "report", capture_final)
    algorithm._train_loop(config)
    actual: list[dict[str, Any]] = []
    monkeypatch.setattr(
        train, "get_checkpoint", lambda: Checkpoint.from_directory(saved)
    )
    monkeypatch.setattr(
        train, "report", lambda metrics, **kwargs: actual.append(metrics)
    )
    native_train = xgboost.train
    calls: list[int] = []

    def train_zero(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["num_boost_round"])
        return native_train(*args, **kwargs)

    monkeypatch.setattr(xgboost, "train", train_zero)
    algorithm._train_loop(config)
    assert calls == [0] and len(actual) == 1 and not evidence
    assert actual[0]["attempt_training_rounds"] == 0
    assert actual[0]["resumed_from_round"] == initial[0]["training_rounds"]
    for key in (
        "model_state_digest",
        "training_rounds",
        "model_rounds",
        "validation-logloss",
    ):
        assert actual[0][key] == initial[0][key]
