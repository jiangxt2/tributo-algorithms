"""Validate the installed XGBoost Wheel through Core and stage execution."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pandas as pd
import ray
import xgboost
from tributo.algorithms import build_algorithm_dispatcher
from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRequest,
    ExecutionProfile,
    ExecutionRequest,
    InputBinding,
    InputBindingSet,
    WorkerResources,
)
from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
from tributo.data import IngestionRequest, ParquetSourceConfig
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.models import BundleRef
from tributo.integrations.algorithm_inputs import (
    INGESTION_RESOLVER_ID,
    IngestionInputInvocation,
)
from tributo_algorithms_boosting import XGBoostStageRunner
from tributo_algorithms_boosting.checkpointing import REPORT_KEY
from tributo_algorithms_boosting.data_config import CompleteCoverageDataConfig
from tributo_algorithms_boosting.native import OfficialXGBoostNativeFlavor


def _read_roles(iterators: dict[str, Any]) -> dict[str, list[int]]:
    return {
        role: [int(row["id"]) for row in iterator.iter_rows()]
        for role, iterator in iterators.items()
    }


def _identity_contract() -> list[dict[str, list[int]]]:
    datasets = {
        "train": ray.data.range(5, override_num_blocks=5),
        "validation": ray.data.range(7, override_num_blocks=7),
    }
    assigned = CompleteCoverageDataConfig(datasets_to_split=list(datasets)).configure(
        datasets, 2, None, None
    )
    reader = ray.remote(num_cpus=0)(_read_roles)
    workers = ray.get([reader.remote(inputs) for inputs in assigned])
    for role, rows in (("train", 5), ("validation", 7)):
        identities = [identity for worker in workers for identity in worker[role]]
        assert sorted(identities) == list(range(rows))
        assert len(identities) == len(set(identities))
    return workers


def _request(
    root: Path,
    train_path: Path,
    validation_path: Path,
    params: dict[str, Any],
    *,
    early: bool,
) -> tuple[ExecutionRequest, dict[str, IngestionInputInvocation]]:
    bindings = tuple(
        InputBinding(
            name=role,
            resolver_id=INGESTION_RESOLVER_ID,
            reference=f"{role}-input",
            feature_names=("f0", "f1"),
            label_name="label",
        )
        for role in ("train", "validation")
    )
    inputs = {
        f"{role}-input": IngestionInputInvocation(
            request=IngestionRequest(
                source=ParquetSourceConfig(path=str(path)), engine="ray"
            )
        )
        for role, path in (("train", train_path), ("validation", validation_path))
    }
    return (
        ExecutionRequest(
            algorithm_request=AlgorithmRequest(
                algorithm="xgboost",
                operation=AlgorithmOperation.FIT,
                input_binding=InputBindingSet(bindings=bindings),
                algorithm_config={
                    "data": {"label_col": "label"},
                    "model": {**params, "eval_metric": "logloss"},
                    "training": {
                        "num_rounds": 20 if early else 3,
                        **(
                            {"early_stopping_rounds": 2, "maximize": False}
                            if early
                            else {}
                        ),
                    },
                    "ray": {"storage_path": str(root / "checkpoints")},
                    "output": {"bundle_uri": str(root / "bundle")},
                },
            ),
            profile=ExecutionProfile.CLUSTER,
            worker_count=2,
        ),
        inputs,
    )


def _validation_case(
    root: Path,
    train_path: Path,
    validation_path: Path,
    params: dict[str, Any],
    frame: pd.DataFrame,
    *,
    early: bool,
    validation_rows: int,
) -> dict[str, Any]:
    request, inputs = _request(root, train_path, validation_path, params, early=early)
    result = build_algorithm_dispatcher().execute(
        request,
        InputExecutionContext(inputs),
        resolution_context=InputResolutionContext(values=inputs),
    )
    receipt = result.execution_receipt
    assert receipt is not None and result.execution.status == "succeeded"
    assert len({worker.node_id for worker in receipt.workers}) == 2
    assert sum(worker.input_rows["train"] for worker in receipt.workers) == len(frame)
    assert (
        sum(worker.input_rows["validation"] for worker in receipt.workers)
        == validation_rows
    )
    digests = {worker.model_state_digest for worker in receipt.workers}
    assert len(digests) == 1
    metrics = dict(result.execution.metrics)
    assert all(
        worker.collective_steps == metrics["training_rounds"]
        for worker in receipt.workers
    )
    if early:
        assert metrics["best_iteration"] + 1 == metrics["model_rounds"]
        assert metrics["model_rounds"] < metrics["training_rounds"] < 20
        assert metrics["best_score"] < metrics["validation-logloss"]
    else:
        assert metrics["model_rounds"] == metrics["training_rounds"] == 3
        assert "best_iteration" not in metrics
    outputs = result.execution.outputs
    ref = BundleRef(
        canonical_uri=outputs["bundle_uri"],
        bundle_id=outputs["bundle_id"],
        manifest_sha256=outputs["manifest_sha256"],
    )
    reader = BundleReader()
    sample = frame[["f0", "f1"]].iloc[::8192].to_numpy()
    with reader.open_artifact(ref, role="native") as artifact:
        booster = xgboost.Booster()
        booster.load_model(artifact.path_for("model.ubj"))
        assert booster.num_boosted_rounds() == metrics["model_rounds"]
        assert (
            hashlib.sha256(bytes(booster.save_raw(raw_format="ubj"))).hexdigest()
            in digests
        )
        metadata = json.loads(artifact.path_for("training_metadata.json").read_text())
        for key, value in metadata.items():
            if key != "schema_version":
                assert metrics[key] == value
        model = OfficialXGBoostNativeFlavor().load(
            artifact, role="native", architecture_id="xgboost"
        )
        expected = model.predict({"float_input": sample})["probabilities"][:, 1]
    with reader.open_artifact(ref, role="inference") as artifact:
        assert (
            json.loads(artifact.path_for("training_metadata.json").read_text())
            == metadata
        )
        session = onnxruntime.InferenceSession(
            str(artifact.path_for("model.onnx")), providers=["CPUExecutionProvider"]
        )
        predictions = session.run(None, {session.get_inputs()[0].name: sample})
        np.testing.assert_allclose(predictions[1][:, 1], expected, atol=1e-6)
    return {"metrics": metrics, "receipt": receipt.to_dict(), "bundle": dict(outputs)}


@ray.remote(num_cpus=0)
class _FaultLatch:
    """Test-only synchronization with the controller's persisted-report event."""

    def __init__(self, phase: str | None, at_round: int | None, failures: int) -> None:
        self.phase, self.at_round, self.failures = phase, at_round, failures
        self.events: dict[tuple[str, str, int], asyncio.Event] = {}
        self.decisions: dict[tuple[str, str, int], bool] = {}
        self.commits: list[dict[str, Any]] = []
        self.starts: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    def started(self, value: dict[str, Any]) -> None:
        self.starts.append(value)

    def updated(self, worker: str, round_number: int) -> None:
        self.updates.append({"worker": worker, "first_new_round": round_number})

    def committed(self, metrics: list[dict[str, Any]], checkpoint_path: str) -> None:
        first = metrics[0]
        phase = first[REPORT_KEY]["phase"]
        completed = first["training_rounds"]
        fail = (
            self.failures > 0
            and phase == self.phase
            and (self.at_round is None or completed == self.at_round)
        )
        if fail:
            self.failures -= 1
        workers = [item[REPORT_KEY]["gate_worker_id"] for item in metrics]
        self.commits.append(
            {
                "phase": phase,
                "completed_rounds": completed,
                "workers": workers,
                "checkpoint_path": checkpoint_path,
                "injected_failure": fail,
                "resumed_from_round": first.get("resumed_from_round"),
                "attempt_training_rounds": first.get("attempt_training_rounds"),
            }
        )
        for rank, worker in enumerate(workers):
            key = (worker, phase, completed)
            self.decisions[key] = fail and rank == 0
            self.events.setdefault(key, asyncio.Event()).set()

    async def decision(self, worker: str, phase: str, completed: int) -> bool:
        key = (worker, phase, completed)
        await asyncio.wait_for(
            self.events.setdefault(key, asyncio.Event()).wait(), timeout=90
        )
        return self.decisions[key]

    def snapshot(self) -> dict[str, Any]:
        return {"starts": self.starts, "updates": self.updates, "commits": self.commits}


def _fault_worker_loop(config: dict[str, Any]) -> None:
    """Exercise installed production code; injection exists only in this job."""
    from ray import train
    from ray.train import Checkpoint
    from tributo_algorithms_boosting.algorithm import _train_loop

    latch = config["gate_latch"]
    worker = str(ray.get_runtime_context().get_worker_id())
    rank = train.get_context().get_world_rank()
    get_checkpoint, report, update = (
        train.get_checkpoint,
        train.report,
        xgboost.Booster.update,
    )
    checkpoint = get_checkpoint()
    completed = 0
    if checkpoint is not None:
        with checkpoint.as_directory() as directory:
            completed = json.loads(
                (Path(directory) / "recovery_state.json").read_text()
            )["completed_rounds"]
    ray.get(
        latch.started.remote(
            {
                "worker": worker,
                "rank": rank,
                "checkpoint_rounds": completed,
                "invocation": config["invocation"],
            }
        )
    )
    first_update = True

    def observed_update(model: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal first_update
        if first_update:
            ray.get(latch.updated.remote(worker, completed + 1))
            first_update = False
        return update(model, *args, **kwargs)

    def committed_report(metrics: dict[str, Any], **kwargs: Any) -> None:
        metrics[REPORT_KEY]["gate_worker_id"] = worker
        report(metrics, **kwargs)
        if ray.get(
            latch.decision.remote(
                worker, metrics[REPORT_KEY]["phase"], metrics["training_rounds"]
            )
        ):
            print(
                f"GATE: worker {worker} exits after persisted {metrics['training_rounds']} / {metrics[REPORT_KEY]['phase']}",
                flush=True,
            )
            os._exit(86)

    try:
        train.report = committed_report
        xgboost.Booster.update = observed_update
        if config.get("gate_foreign_checkpoint"):
            train.get_checkpoint = lambda: Checkpoint.from_directory(
                config["gate_foreign_checkpoint"]
            )
        _train_loop(config)
    finally:
        train.report, train.get_checkpoint, xgboost.Booster.update = (
            report,
            get_checkpoint,
            update,
        )


def _recovery_case(
    root: Path,
    train_path: Path,
    validation_path: Path,
    params: dict[str, Any],
    frame: pd.DataFrame,
    *,
    early: bool = False,
    rounds: int = 20,
    patience: int = 4,
    interval: int = 10,
    phase: str | None = None,
    at_round: int | None = None,
    failures: int = 1,
    max_failures: int = 1,
    storage: Path | None = None,
    foreign: str | None = None,
    expect_failure: bool = False,
) -> dict[str, Any]:
    import ray.train.xgboost as ray_xgboost
    from ray.train import UserCallback

    request, inputs = _request(root, train_path, validation_path, params, early=early)
    config = dict(request.algorithm_request.algorithm_config)
    config["training"] = {
        "num_rounds": rounds,
        "checkpoint_interval_rounds": interval,
        **({"early_stopping_rounds": patience, "maximize": False} if early else {}),
    }
    config["ray"] = {
        "storage_path": str(storage or root / "checkpoints"),
        "max_failures": max_failures,
    }
    request = replace(
        request,
        algorithm_request=replace(request.algorithm_request, algorithm_config=config),
    )
    latch = _FaultLatch.remote(phase, at_round, failures)
    trainer_type = ray_xgboost.XGBoostTrainer

    class PersistedReport(UserCallback):
        def after_report(
            self, run_context: Any, metrics: list[dict[str, Any]], checkpoint: Any
        ) -> None:
            assert checkpoint is not None
            ray.get(latch.committed.remote(metrics, checkpoint.path))

    def trainer(**kwargs: Any) -> Any:
        kwargs["train_loop_per_worker"] = _fault_worker_loop
        kwargs["train_loop_config"].update(
            gate_latch=latch, gate_foreign_checkpoint=foreign
        )
        kwargs["run_config"].callbacks.append(PersistedReport())
        return trainer_type(**kwargs)

    error = None
    result = None
    try:
        ray_xgboost.XGBoostTrainer = trainer
        try:
            result = build_algorithm_dispatcher().execute(
                request,
                InputExecutionContext(inputs),
                resolution_context=InputResolutionContext(values=inputs),
            )
        except Exception:
            if not expect_failure:
                raise
            error = traceback.format_exc()
        events = ray.get(latch.snapshot.remote())
    finally:
        ray_xgboost.XGBoostTrainer = trainer_type
        ray.kill(latch, no_restart=True)
    if expect_failure:
        assert error is not None or (
            result is not None and result.execution.status == "failed"
        )
        assert not (root / "bundle").exists()
        return {
            "events": events,
            "error": error or result.execution.error_message,
            "no_bundle": True,
        }
    assert result is not None and result.execution.status == "succeeded"
    receipt = result.execution_receipt
    assert (
        receipt is not None and len({worker.node_id for worker in receipt.workers}) == 2
    )
    assert sum(worker.input_rows["train"] for worker in receipt.workers) == len(frame)
    metrics = dict(result.execution.metrics)
    assert sum(worker.input_rows["validation"] for worker in receipt.workers) == (
        len(frame) if validation_path == train_path else len(frame) + 2
    )
    assert all(
        worker.batch_count
        == worker.collective_steps
        == metrics["attempt_training_rounds"]
        for worker in receipt.workers
    )
    final = events["commits"][-1]
    assert final["phase"] == "final"
    assert {worker.worker_id for worker in receipt.workers} == set(final["workers"])
    outputs = result.execution.outputs
    ref = BundleRef(
        canonical_uri=outputs["bundle_uri"],
        bundle_id=outputs["bundle_id"],
        manifest_sha256=outputs["manifest_sha256"],
    )
    with BundleReader().open_artifact(ref, role="native") as artifact:
        booster = xgboost.Booster(model_file=artifact.path_for("model.ubj"))
        assert booster.num_boosted_rounds() == metrics["model_rounds"]
        sample = frame[["f0", "f1"]].iloc[::8192].to_numpy()
        predictions = booster.predict(xgboost.DMatrix(sample))
    assert REPORT_KEY not in metrics
    return {
        "metrics": metrics,
        "receipt": receipt.to_dict(),
        "events": events,
        "predictions": predictions.tolist(),
    }


def _recovery_contract(
    root: Path,
    train: Path,
    validation: Path,
    params: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    storage = root / "shared-checkpoints"
    cases["fixed-baseline"] = _recovery_case(
        root / "fixed-baseline", train, validation, params, frame, storage=storage
    )
    cases["fixed-recovery"] = _recovery_case(
        root / "fixed-recovery",
        train,
        validation,
        params,
        frame,
        storage=storage,
        phase="training",
        at_round=10,
    )
    fixed = cases["fixed-recovery"]
    assert (
        fixed["metrics"]["resumed_from_round"]
        == fixed["metrics"]["attempt_training_rounds"]
        == 10
    )
    assert fixed["metrics"]["training_rounds"] == fixed["metrics"]["model_rounds"] == 20
    recovered_workers = {
        item["worker"]
        for item in fixed["events"]["starts"]
        if item["checkpoint_rounds"] == 10
    }
    assert len(recovered_workers) == 2
    assert {
        item["first_new_round"]
        for item in fixed["events"]["updates"]
        if item["worker"] in recovered_workers
    } == {11}
    np.testing.assert_allclose(
        fixed["predictions"], cases["fixed-baseline"]["predictions"], atol=1e-6
    )
    cases["new-invocation"] = _recovery_case(
        root / "new-invocation", train, validation, params, frame, storage=storage
    )
    assert all(
        item["checkpoint_rounds"] == 0
        for item in cases["new-invocation"]["events"]["starts"]
    )
    assert {
        item["invocation"] for item in cases["new-invocation"]["events"]["starts"]
    }.isdisjoint({item["invocation"] for item in fixed["events"]["starts"]})
    for name, validation_path, patience, rounds, interval, at_round in (
        ("patience", validation, 4, 20, 3, 3),
        ("cap", train, 20, 6, 2, 2),
    ):
        baseline = _recovery_case(
            root / f"{name}-baseline",
            train,
            validation_path,
            params,
            frame,
            early=True,
            patience=patience,
            rounds=rounds,
            interval=interval,
        )
        recovered = _recovery_case(
            root / f"{name}-recovery",
            train,
            validation_path,
            params,
            frame,
            early=True,
            patience=patience,
            rounds=rounds,
            interval=interval,
            phase="training",
            at_round=at_round,
        )
        for field in ("training_rounds", "model_rounds", "best_iteration"):
            assert recovered["metrics"][field] == baseline["metrics"][field]
        assert (
            abs(recovered["metrics"]["best_score"] - baseline["metrics"]["best_score"])
            < 1e-6
        )
        np.testing.assert_allclose(
            recovered["predictions"], baseline["predictions"], atol=1e-6
        )
        if name == "patience":
            assert recovered["metrics"]["training_rounds"] == 5
            assert recovered["metrics"]["best_iteration"] == 0
            assert recovered["metrics"]["attempt_training_rounds"] == 2
        else:
            assert recovered["metrics"]["training_rounds"] == rounds
            assert recovered["metrics"]["best_iteration"] >= at_round
        cases[f"{name}-baseline"], cases[f"{name}-recovery"] = baseline, recovered
    for early in (False, True):
        terminal = _recovery_case(
            root / f"terminal-{early}",
            train,
            validation,
            params,
            frame,
            early=early,
            interval=3,
            phase="final",
        )
        assert terminal["metrics"]["attempt_training_rounds"] == 0
        assert (
            terminal["metrics"]["resumed_from_round"]
            == terminal["metrics"]["training_rounds"]
        )
        final_workers = set(terminal["events"]["commits"][-1]["workers"])
        assert not any(
            item["worker"] in final_workers for item in terminal["events"]["updates"]
        )
        assert (
            sum(item["phase"] == "final" for item in terminal["events"]["commits"]) == 2
        )
        cases[f"terminal-{early}"] = terminal
    cases["budget-exhausted"] = _recovery_case(
        root / "budget-exhausted",
        train,
        validation,
        params,
        frame,
        phase="training",
        failures=2,
        expect_failure=True,
    )
    assert (
        sum(
            item["injected_failure"]
            for item in cases["budget-exhausted"]["events"]["commits"]
        )
        == 2
    )
    cases["foreign-checkpoint"] = _recovery_case(
        root / "foreign-checkpoint",
        train,
        validation,
        params,
        frame,
        max_failures=0,
        foreign=fixed["events"]["commits"][-1]["checkpoint_path"],
        expect_failure=True,
    )
    assert "checkpoint is invalid" in str(cases["foreign-checkpoint"]["error"])
    return cases


def main(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    rows = 131_075  # Uneven split; both shards exceed the 65,536-row batch bound.
    values = np.linspace(-2, 2, rows, dtype=np.float32)
    frame = pd.DataFrame({"f0": values, "f1": values**2, "label": values > 0})
    frame["label"] = frame["label"].astype(np.int64)
    data_path = root / "train"
    data_path.mkdir()
    for index, part in enumerate(np.array_split(np.arange(rows), 4)):
        frame.iloc[part].to_parquet(data_path / f"part-{index}.parquet", index=False)
    validation_rows = rows + 2
    validation_values = np.linspace(-2, 2, validation_rows, dtype=np.float32)
    validation_frame = pd.DataFrame(
        {
            "f0": validation_values,
            "f1": validation_values**2,
            "label": (validation_values <= 0).astype(np.int64),
        }
    )
    validation_path = root / "validation"
    validation_path.mkdir()
    for index, part in enumerate(np.array_split(np.arange(validation_rows), 4)):
        validation_frame.iloc[part].to_parquet(
            validation_path / f"part-{index}.parquet", index=False
        )
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "seed": 7,
        "nthread": 1,
        "max_depth": 2,
    }
    invocation = IngestionInputInvocation(
        request=IngestionRequest(
            source=ParquetSourceConfig(path=str(data_path)), engine="ray"
        )
    )
    inputs = {"train-input": invocation}
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm="xgboost",
            operation=AlgorithmOperation.FIT,
            input_binding=InputBinding(
                name="train",
                resolver_id=INGESTION_RESOLVER_ID,
                reference="train-input",
                feature_names=("f0", "f1"),
                label_name="label",
            ),
            algorithm_config={
                "data": {"label_col": "label"},
                "model": params,
                "training": {"num_rounds": 3},
                "ray": {"storage_path": str(root / "checkpoints")},
                "output": {"bundle_uri": str(root / "bundle")},
            },
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    ray.init(address="auto")
    try:
        identities = _identity_contract()
        result = build_algorithm_dispatcher().execute(
            request,
            InputExecutionContext(inputs),
            resolution_context=InputResolutionContext(values=inputs),
        )
        receipt = result.execution_receipt
        assert receipt is not None and result.execution.status == "succeeded"
        assert len(receipt.workers) == 2
        assert len({worker.node_id for worker in receipt.workers}) == 2
        assert sum(worker.input_rows["train"] for worker in receipt.workers) == rows
        assert all(
            worker.batch_count == worker.collective_steps == 3
            for worker in receipt.workers
        )
        digests = {worker.model_state_digest for worker in receipt.workers}
        assert len(digests) == 1
        outputs = result.execution.outputs
        ref = BundleRef(
            canonical_uri=outputs["bundle_uri"],
            bundle_id=outputs["bundle_id"],
            manifest_sha256=outputs["manifest_sha256"],
        )
        reader = BundleReader()
        manifest = reader.read_manifest(ref)
        assert set(manifest.roles) == {"inference", "native"}
        with reader.open_artifact(ref, role="native") as artifact:
            booster = xgboost.Booster()
            booster.load_model(artifact.path_for("model.ubj"))
            assert booster.num_boosted_rounds() == 3
            assert (
                hashlib.sha256(bytes(booster.save_raw(raw_format="ubj"))).hexdigest()
                in digests
            )
        sample = frame[["f0", "f1"]].iloc[:8].to_numpy()
        expected = booster.predict(xgboost.DMatrix(sample))
        with reader.open_artifact(ref, role="inference") as artifact:
            session = onnxruntime.InferenceSession(
                str(artifact.path_for("model.onnx")),
                providers=["CPUExecutionProvider"],
            )
            predictions = session.run(None, {session.get_inputs()[0].name: sample})
            np.testing.assert_allclose(predictions[1][:, 1], expected, atol=1e-6)

        stage = XGBoostStageRunner(
            worker_count=2,
            resources_per_worker=WorkerResources(num_cpus=1),
            storage_path=str(root / "stage-checkpoints"),
            input_binding_digest="gate-input",
        ).fit(
            "bounded-worker",
            ray.data.from_pandas(frame).repartition(4),
            feature_names=("f0", "f1"),
            label_name="label",
            params=params,
            num_boost_round=3,
        )
        workers = stage.evidence["workers"]
        assert stage.row_count == rows
        assert sum(worker["rows_processed"] for worker in workers) == rows
        assert len({worker["node_id"] for worker in workers}) == 2
        assert all(
            worker["batch_count"] == worker["collective_steps"] == 3
            for worker in workers
        )
        recovered = xgboost.Booster()
        recovered.load_model(bytearray(stage.booster_raw))
        assert recovered.num_boosted_rounds() == 3
        validation_cases = {
            name: _validation_case(
                root / name,
                data_path,
                validation_path,
                params,
                frame,
                early=early,
                validation_rows=validation_rows,
            )
            for name, early in (("validation-fixed", False), ("validation-early", True))
        }
        empty_path = root / "empty-validation"
        empty_path.mkdir()
        validation_frame.iloc[:0].to_parquet(empty_path / "empty.parquet", index=False)
        failed_root = root / "empty-validation-run"
        empty_request, empty_inputs = _request(
            failed_root, data_path, empty_path, params, early=True
        )
        try:
            failed = build_algorithm_dispatcher().execute(
                empty_request,
                InputExecutionContext(empty_inputs),
                resolution_context=InputResolutionContext(values=empty_inputs),
            )
        except Exception:
            error = traceback.format_exc()
            assert "empty" in error.lower(), error
        else:
            assert failed.execution.status == "failed"
            assert "empty" in str(failed.execution.error_message).lower()
        assert not (failed_root / "bundle").exists()
        recovery_cases = _recovery_contract(
            root / "recovery", data_path, validation_path, params, frame
        )
        summary = {
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("tributo", "tributo-algorithms", "ray", "xgboost")
            },
            "rows": rows,
            "rounds": 3,
            "workers": 2,
            "core_receipt": receipt.to_dict(),
            "stage_evidence": stage.evidence,
            "bundle": outputs,
            "identity_contract": identities,
            "validation_cases": validation_cases,
            "empty_validation_rejected": True,
            "recovery_cases": recovery_cases,
        }
        (root / "result.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print("RESULT: " + json.dumps(summary), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="New shared directory visible at the same path on every Ray node",
    )
    main(parser.parse_args().root)
