"""Validate the installed XGBoost Wheel through Core and stage execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import traceback
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
