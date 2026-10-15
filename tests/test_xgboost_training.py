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
) -> tuple[_Shard | None, list[dict[str, Any]], list[dict[str, Any]]]:
    shard = _Shard(frame) if frame is not None else None
    evidence: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    monkeypatch.setattr(algorithm, "_XGBOOST_BATCH_ROWS", 2)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(train, "get_dataset_shard", lambda _name: shard)
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
