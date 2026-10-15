"""Validate the installed XGBoost Wheel through Core and stage execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

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
