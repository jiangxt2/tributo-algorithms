"""Exercise the installed XGBoost Wheel through one Ray Jobs node."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pandas as pd
import ray
import xgboost
from tributo.algorithms import (
    AlgorithmOperation,
    AlgorithmRequest,
    ExecutionProfile,
    ExecutionRequest,
    InputBinding,
    build_algorithm_dispatcher,
)
from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
from tributo.data import IngestionRequest, ParquetSourceConfig, ReadOptions
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.models import BundleRef
from tributo.integrations.algorithm_inputs import (
    INGESTION_RESOLVER_ID,
    IngestionInputInvocation,
)

SMOKE_ROW_COUNT = 131_075


def _runtime_versions(
    *,
    expected_python_version: str,
    expected_tributo_version: str,
    expected_ray_version: str,
) -> dict[str, str]:
    actual_python = ".".join(str(part) for part in sys.version_info[:2])
    versions = {
        "python": actual_python,
        "tributo": importlib.metadata.version("tributo"),
        "tributo-algorithms": importlib.metadata.version("tributo-algorithms"),
        "ray": ray.__version__,
        "xgboost": xgboost.__version__,
    }
    expected = {
        "python": expected_python_version,
        "tributo": expected_tributo_version,
        "ray": expected_ray_version,
    }
    for name, value in expected.items():
        if versions[name] != value:
            raise AssertionError(
                f"runtime {name} version drifted: expected {value}, got {versions[name]}"
            )
    return versions


def _write_training_data(root: Path) -> tuple[Path, pd.DataFrame]:
    values = np.linspace(-2.0, 2.0, SMOKE_ROW_COUNT, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "f0": values,
            "f1": values**2,
            "label": (values > 0).astype(np.int64),
        }
    )
    order = np.random.default_rng(7).permutation(len(frame))
    frame = frame.iloc[order].reset_index(drop=True)

    data_path = root / "train"
    data_path.mkdir()
    for index, rows in enumerate(np.array_split(np.arange(len(frame)), 4)):
        frame.iloc[rows].to_parquet(data_path / f"part-{index}.parquet", index=False)
    return data_path, frame


def _run_training(root: Path, train_path: Path, row_count: int) -> dict[str, Any]:
    invocation = IngestionInputInvocation(
        request=IngestionRequest(
            source=ParquetSourceConfig(path=str(train_path)),
            engine="ray",
            read_options=ReadOptions(target_parallelism=4),
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
                "model": {
                    "objective": "binary:logistic",
                    "tree_method": "hist",
                    "seed": 7,
                    "nthread": 1,
                    "max_depth": 2,
                    "eval_metric": "logloss",
                },
                "training": {"num_rounds": 3},
                "ray": {"storage_path": str(root / "checkpoints")},
                "output": {"bundle_uri": str(root / "bundle")},
            },
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    result = build_algorithm_dispatcher().execute(
        request,
        InputExecutionContext(inputs),
        resolution_context=InputResolutionContext(values=inputs),
    )
    receipt = result.execution_receipt
    assert receipt is not None and result.execution.status == "succeeded"
    assert len(receipt.workers) == 2
    assert len({worker.node_id for worker in receipt.workers}) == 1
    assert sum(worker.input_rows["train"] for worker in receipt.workers) == row_count
    assert len({worker.model_state_digest for worker in receipt.workers}) == 1

    outputs = dict(result.execution.outputs)
    bundle = BundleRef(
        canonical_uri=outputs["bundle_uri"],
        bundle_id=outputs["bundle_id"],
        manifest_sha256=outputs["manifest_sha256"],
    )
    reader = BundleReader()
    manifest = reader.read_manifest(bundle)
    assert set(manifest.roles) == {"inference", "native"}
    return {
        "receipt": receipt,
        "bundle": bundle,
        "outputs": outputs,
        "manifest_roles": sorted(manifest.roles),
    }


def _verify_bundle(bundle: BundleRef, frame: pd.DataFrame) -> None:
    sample = frame[["f0", "f1"]].head(8).to_numpy(dtype=np.float32)
    reader = BundleReader()
    with reader.open_artifact(bundle, role="native") as artifact:
        booster = xgboost.Booster()
        booster.load_model(artifact.path_for("model.ubj"))
        expected = booster.predict(xgboost.DMatrix(sample))

    with reader.open_artifact(bundle, role="inference") as artifact:
        session = onnxruntime.InferenceSession(
            str(artifact.path_for("model.onnx")),
            providers=["CPUExecutionProvider"],
        )
        predictions = session.run(
            None,
            {session.get_inputs()[0].name: sample},
        )

    np.testing.assert_allclose(predictions[1][:, 1], expected, atol=1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--expected-tributo-version", required=True)
    parser.add_argument("--expected-ray-version", required=True)
    parser.add_argument("--expected-ray-image", required=True)
    args = parser.parse_args()

    versions = _runtime_versions(
        expected_python_version=args.expected_python_version,
        expected_tributo_version=args.expected_tributo_version,
        expected_ray_version=args.expected_ray_version,
    )
    print(
        "RUNTIME VERSIONS: "
        + json.dumps(
            {"versions": versions, "ray_image": args.expected_ray_image},
            sort_keys=True,
        ),
        flush=True,
    )
    args.root.mkdir(parents=True, exist_ok=False)

    ray.init(address="auto")
    try:
        train_path, frame = _write_training_data(args.root)
        result = _run_training(args.root, train_path, len(frame))
        _verify_bundle(result["bundle"], frame)

        summary = {
            "versions": versions,
            "ray_image": args.expected_ray_image,
            "rows": len(frame),
            "workers": [
                {
                    "worker_id": worker.worker_id,
                    "node_id": worker.node_id,
                    "input_rows": dict(worker.input_rows),
                }
                for worker in result["receipt"].workers
            ],
            "bundle": result["outputs"],
            "bundle_roles": result["manifest_roles"],
            "native_onnx_prediction_parity": True,
        }
        (args.root / "result.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print("RESULT: " + json.dumps(summary), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
