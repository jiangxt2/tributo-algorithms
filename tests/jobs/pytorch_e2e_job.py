"""Fixed-Wheel CPU probes submitted through Ray Jobs, with bounded fixtures."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

ALGORITHMS = (
    "dnn",
    "pu",
    "temporal_conv_classifier",
    "lstm_classifier",
    "gru_classifier",
    "tabular_autoencoder",
    "token_transformer_classifier",
    "graphsage_node_classifier",
    "rgcn_node_classifier",
    "two_tower_recommender",
    "jagged_embedding_recommender",
    "teacher_student_distillation",
    "pretrain_finetune_classifier",
)
NEGATIVE_CASES = (
    "missing_column",
    "nan",
    "inf",
    "invalid_label",
    "empty",
    "insufficient",
    "one_worker",
    "resume",
)
PERFORMANCE_CASES = tuple(
    f"perf_{rows}_{workers}_{engine}"
    for rows in (1_000_000, 10_000_000)
    for workers in (2, 4)
    for engine in ("tributo", "ray")
)
DEFAULT_CASES = (
    "core_unit",
    "package",
    "unit",
    *ALGORITHMS,
    "multi_epoch",
    *NEGATIVE_CASES,
    "integrity",
    "delivery",
    "tune",
    "failure",
    "stop",
    "split_probe",
    "reference_contract",
    "token_transformer_large_batch",
    "transformer_unit",
    "transformer_ddp_alignment",
    "core_runtime_profile",
    *PERFORMANCE_CASES,
)
ROOT = Path("/e2e")
EVIDENCE = ROOT / "evidence"
JOB = "/opt/tributo-e2e/pytorch_e2e_job.py"
RUNTIME_REQUIREMENTS = Path("/opt/tributo-e2e/requirements.txt")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )


def validate_receipt(receipt: dict[str, Any], rows: int, workers: int) -> None:
    """Reject false distributed success or incomplete role coverage."""
    assert receipt["input_complete"] is True
    assert receipt["distributed"] is True
    observed = receipt["workers"]
    assert len(observed) == workers
    assert len({worker["worker_id"] for worker in observed}) == workers
    assert len({worker["rank"] for worker in observed}) == workers
    assert len({worker["node_id"] for worker in observed}) >= 2
    assert sum(worker["input_rows"]["train"] for worker in observed) == rows
    digests = {worker["model_state_digest"] for worker in observed}
    assert len(digests) == 1 and None not in digests


def runtime_versions() -> dict[str, str]:
    import torch

    expected = {
        "tributo": os.environ["TRIBUTO_EXPECTED_CORE"],
        "ray": os.environ["TRIBUTO_EXPECTED_RAY"],
        "torch": os.environ["TRIBUTO_EXPECTED_TORCH"],
    }
    actual = {
        name: importlib.metadata.version(name)
        for name in (
            *expected,
            "tributo-algorithms",
            "onnxruntime",
            "torch-geometric",
            "mlflow",
        )
    }
    assert sys.version_info[:2] == (3, 12)
    assert all(actual[name] == version for name, version in expected.items()), actual
    assert not torch.cuda.is_available()
    assert torch.version.cuda is None
    operand = torch.ones(8, 2, requires_grad=True)
    (operand @ torch.ones(2, 4)).sum().backward()
    assert operand.device.type == "cpu" and bool((operand.grad == 4).all())
    for module in ("tributo", "tributo_algorithms_tabular_torch"):
        loaded = __import__(module)
        location = str(loaded.__file__)
        assert "/site-packages/" in location, location
    return actual


def ready() -> bool:
    import requests
    from ray.job_submission import JobSubmissionClient
    from ray.util.state import list_nodes

    JobSubmissionClient("http://127.0.0.1:8265").get_version()
    nodes = list_nodes(
        address="http://127.0.0.1:8265", filters=[("state", "=", "ALIVE")]
    )
    assert len(nodes) == 3
    requests.get("http://mlflow:5000/health", timeout=3).raise_for_status()
    requests.get("http://minio:9000/minio/health/live", timeout=3).raise_for_status()
    return True


def package_probe() -> dict[str, Any]:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0)
    def inspect() -> dict[str, Any]:
        from packaging.tags import Tag, parse_tag, sys_tags

        supported = set(sys_tags())
        unsupported = []
        for distribution in importlib.metadata.distributions():
            metadata = distribution.read_text("WHEEL") or ""
            tags: set[Tag] = set()
            for line in metadata.splitlines():
                if line.startswith("Tag:"):
                    tags.update(parse_tag(line.split(":", 1)[1].strip()))
            if tags and supported.isdisjoint(tags):
                unsupported.append(distribution.metadata["Name"])
        check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "versions": runtime_versions(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "pip_check": {
                "returncode": check.returncode,
                "stdout": check.stdout,
                "stderr": check.stderr,
            },
            "unsupported_wheels": unsupported,
            "torch_wheel_metadata": importlib.metadata.distribution("torch").read_text(
                "WHEEL"
            ),
        }

    nodes = [node for node in ray.nodes() if node["Alive"]]
    observations = ray.get(
        [
            inspect.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node["NodeID"], soft=False
                )
            ).remote()
            for node in nodes
        ]
    )
    entries = {
        ep.name: ep
        for ep in importlib.metadata.entry_points(group="tributo.algorithms")
        if ep.dist and ep.dist.name == "tributo-algorithms"
    }
    assert len(entries) == 37
    assert set(ALGORITHMS) <= set(entries)
    report = {
        "nodes": observations,
        "entry_points": sorted(entries),
        "torch_algorithms": list(ALGORITHMS),
    }
    save(EVIDENCE / "package-preflight.json", report)
    validate_package_nodes(observations)
    return report


def validate_package_nodes(observations: list[dict[str, Any]]) -> None:
    assert len(observations) == 3
    assert len({node["node_id"] for node in observations}) == 3
    for node in observations:
        assert node["pip_check"]["returncode"] == 0, node
        assert not node["unsupported_wheels"], node


def dense_frame(rows: int, seed: int, *, features: int = 2, pu: bool = False) -> Any:
    import numpy as np
    import pandas as pd

    values = (
        np.random.default_rng(seed).normal(size=(rows, features)).astype(np.float32)
    )
    truth = (values[:, 0] + 0.5 * values[:, 1] > 0).astype(np.int64)
    labels = truth.copy()
    if pu:
        # Only half of true positives are observed; zero includes hidden positives.
        labels[np.arange(rows) % 2 == 0] = 0
    return pd.DataFrame(
        {
            **{f"x{i}": values[:, i] for i in range(features)},
            "label": labels,
            "truth": truth,
            "row_id": np.arange(rows, dtype=np.int64),
        }
    )


def write_frame(path: Path, frame: Any, parts: int = 4) -> None:
    import numpy as np

    path.mkdir(parents=True, exist_ok=False)
    for index, positions in enumerate(np.array_split(np.arange(len(frame)), parts)):
        frame.iloc[positions].to_parquet(
            path / f"part-{index:03d}.parquet", index=False
        )


def fixture(
    algorithm: str,
    root: Path,
    *,
    rows: int = 8195,
    epochs: int = 1,
    batch_size: int = 128,
    workers: int = 2,
    invalid: str | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import numpy as np
    import pandas as pd
    from tributo.algorithms import (
        AlgorithmOperation,
        AlgorithmRequest,
        ExecutionProfile,
        ExecutionRequest,
        InputBinding,
        InputBindingSet,
    )
    from tributo.data import IngestionRequest, ParquetSourceConfig, ReadOptions
    from tributo.integrations.algorithm_inputs import (
        INGESTION_RESOLVER_ID,
        IngestionInputInvocation,
    )

    root.mkdir(parents=True, exist_ok=False)
    frames: dict[str, Any] = {}
    names: tuple[str, ...] = ("x0", "x1")
    model: dict[str, Any] = {"input_features": 2, "hidden_units": [16, 8]}
    if algorithm in {
        "temporal_conv_classifier",
        "lstm_classifier",
        "gru_classifier",
        "token_transformer_classifier",
    }:
        names = tuple(f"x{i}" for i in range(4))
        model = {"input_features": 4, "hidden_size": 8, "hidden_channels": 4}
    frame = dense_frame(rows, 20261023, features=len(names), pu=algorithm == "pu")
    if algorithm == "tabular_autoencoder":
        model = {"input_features": 2, "latent_features": 1, "hidden_features": 4}
    if algorithm == "token_transformer_classifier":
        model = {
            "vocab_size": 32,
            "sequence_length": 4,
            "hidden_size": 8,
            "heads": 2,
            "layers": 1,
            "dropout": 0.0,
        }
        for i, name in enumerate(names):
            frame[name] = np.where(frame["truth"] == 1, 2 + i, 10 + i).astype(np.int64)
    if algorithm in {"two_tower_recommender", "jagged_embedding_recommender"}:
        model = {"user_count": 16, "item_count": 32, "embedding_dim": 4}
        frame["user_id"] = np.arange(rows, dtype=np.int64) % 16
        frame["item_id"] = np.arange(rows, dtype=np.int64) % 32
        names = ("user_id", "item_id")
        if algorithm == "jagged_embedding_recommender":
            frame["item_history"] = [
                [int(i % 32), int((i + 1) % 32)] for i in range(rows)
            ]
            names = ("user_id", "item_history", "item_id")
    if algorithm == "teacher_student_distillation":
        model = {"input_features": 2, "teacher_hidden": 8, "student_hidden": 4}
    if algorithm == "pretrain_finetune_classifier":
        model = {"input_features": 2, "hidden_features": 4}
    graph = algorithm in {"graphsage_node_classifier", "rgcn_node_classifier"}
    if graph:
        model = {"hidden_features": 4, "num_classes": 2, "num_relations": 2}
        frames["nodes"] = pd.DataFrame(
            {
                "node_id": np.arange(32),
                "x0": np.arange(32, dtype=np.float32) % 2,
                "x1": np.ones(32, dtype=np.float32),
            }
        )
        frames["edges"] = pd.DataFrame(
            {
                "source": np.arange(64) % 32,
                "destination": (np.arange(64) + 1) % 32,
                "relation": np.arange(64) % 2,
            }
        )
        frame = pd.DataFrame({"node_id": np.arange(31), "label": np.arange(31) % 2})
        names = ("node_id",)
    if invalid in {"nan", "inf"}:
        frame.loc[0, names[0]] = float(invalid)
    if invalid == "invalid_label":
        frame.loc[0, "label"] = 2
    if invalid == "missing_column":
        frame = frame.drop(columns=[names[0]])
    if invalid == "empty":
        frame = frame.iloc[:0]
    if invalid == "insufficient":
        frame = frame.iloc[:1]
    frames["train"] = frame
    if algorithm in {"dnn", "pu"} and invalid is None:
        evaluation_rows = 256 if algorithm == "pu" else 257
        frames["val"] = dense_frame(evaluation_rows, 20261024, pu=algorithm == "pu")
        frames["test"] = dense_frame(evaluation_rows, 20261025, pu=algorithm == "pu")
    bindings = []
    inputs = {}
    for role, data in frames.items():
        path = root / "data" / role
        write_frame(path, data)
        feature_names = names
        label = (
            None
            if role in {"nodes", "edges"} or algorithm == "tabular_autoencoder"
            else "label"
        )
        if role == "nodes":
            feature_names = ("node_id", "x0", "x1")
        if role == "edges":
            feature_names = (
                ("source", "destination", "relation")
                if algorithm == "rgcn_node_classifier"
                else ("source", "destination")
            )
        inputs[role] = IngestionInputInvocation(
            request=IngestionRequest(
                source=ParquetSourceConfig(path=str(path)),
                engine="ray",
                read_options=ReadOptions(target_parallelism=4),
            )
        )
        bindings.append(
            InputBinding(
                name=role,
                resolver_id=INGESTION_RESOLVER_ID,
                reference=role,
                feature_names=feature_names,
                label_name=label,
            )
        )
    config: dict[str, Any] = {
        "model": model,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": 42,
            "learning_rate": 0.01,
        },
        "ray": {"storage_path": str(root / "checkpoints"), "max_failures": 0},
        "output": {"bundle_uri": str(root / "bundle")},
    }
    if not graph and algorithm not in {
        "teacher_student_distillation",
        "pretrain_finetune_classifier",
        "jagged_embedding_recommender",
    }:
        config["optimizer"] = {"learning_rate": 0.01, "max_gradient_norm": 1.0}
    if algorithm == "pu":
        config["loss"] = {"type": "nnpu", "class_prior": 0.5}
    if algorithm == "pretrain_finetune_classifier":
        config["training"].update(pretrain_epochs=1, finetune_epochs=1)
    if algorithm == "jagged_embedding_recommender":
        config["data"] = {
            "user_col": "user_id",
            "history_col": "item_history",
            "candidate_col": "item_id",
            "label_col": "label",
            "inference_history_width": 2,
        }
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm=algorithm,
            operation=AlgorithmOperation.FIT,
            input_binding=InputBindingSet(tuple(bindings)),
            algorithm_config=config,
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=workers,
        resume_from=str(root / "incompatible-checkpoint")
        if invalid == "resume"
        else None,
    )
    manifest = {
        "roles": {role: len(data) for role, data in frames.items()},
        "files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "data").rglob("*.parquet"))
        },
    }
    save(EVIDENCE / f"{root.name}-data.json", manifest)
    return request, inputs, frames


def execute(request: Any, inputs: dict[str, Any]) -> Any:
    from tributo.algorithms import build_algorithm_dispatcher
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext

    return build_algorithm_dispatcher().execute(
        request,
        InputExecutionContext(inputs),
        resolution_context=InputResolutionContext(values=inputs),
    )


def feeds_for(session: Any, size: int) -> dict[str, Any]:
    import numpy as np

    result: dict[str, Any] = {}
    for item in session.get_inputs():
        shape = (
            size,
            *[
                dimension if isinstance(dimension, int) else 2
                for dimension in item.shape[1:]
            ],
        )
        dtype = np.int64 if "int64" in item.type else np.float32
        result[item.name] = np.ones(shape, dtype=dtype)
    return result


def verify_bundle(
    outputs: dict[str, Any], root: Path, algorithm: str
) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    import torch
    from tributo.exporting.bundle_reader import BundleReader
    from tributo.exporting.models import BundleRef
    from tributo.exporting.runtime import BundleModelLoader

    bundle = BundleRef(
        canonical_uri=outputs["bundle_uri"],
        bundle_id=outputs["bundle_id"],
        manifest_sha256=outputs["manifest_sha256"],
    )
    reader = BundleReader()
    manifest = reader.read_manifest(bundle)
    report: dict[str, Any] = {
        "bundle": outputs,
        "roles": dict(manifest.roles),
        "dynamic_batches": {},
    }
    with reader.open_artifact(bundle, role="inference") as artifact:
        onnx_path = artifact.entrypoint_path
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        with BundleModelLoader().open(
            bundle.canonical_uri,
            expected_manifest_sha256=bundle.manifest_sha256,
            use_case="batch",
        ) as runtime:
            for size in (1, 17, 257):
                feeds = feeds_for(session, size)
                expected = session.run(None, feeds)
                actual = runtime.predict(feeds)
                for name, value in zip(
                    [item.name for item in session.get_outputs()], expected, strict=True
                ):
                    assert len(value) == size and np.isfinite(value).all()
                    np.testing.assert_allclose(
                        actual[name], value, rtol=1e-4, atol=1e-5
                    )
                report["dynamic_batches"][str(size)] = {
                    name: list(value.shape) for name, value in actual.items()
                }
        if algorithm in {"dnn", "pu"}:
            from sklearn.metrics import average_precision_score, roc_auc_score
            from tributo_algorithms_tabular_torch.recipe import _model

            checkpoints = list((root / "checkpoints").rglob("model.pt"))
            assert len(checkpoints) == 1, checkpoints
            model = _model({"input_features": 2, "hidden_units": [16, 8]})
            model.load_state_dict(
                torch.load(checkpoints[0], map_location="cpu", weights_only=True)
            )
            model.eval()
            errors = {}
            for size in (1, 17, 257):
                samples = dense_frame(size, 20261026)[["x0", "x1"]].to_numpy(
                    dtype=np.float32
                )
                with torch.no_grad():
                    expected = model(torch.from_numpy(samples)).numpy()
                predicted = session.run(None, {session.get_inputs()[0].name: samples})[
                    0
                ]
                np.testing.assert_allclose(predicted, expected, rtol=1e-4, atol=1e-5)
                errors[str(size)] = float(np.max(np.abs(predicted - expected)))
            holdout = dense_frame(4099, 20261027)
            samples = holdout[["x0", "x1"]].to_numpy(dtype=np.float32)
            logits = session.run(None, {session.get_inputs()[0].name: samples})[
                0
            ].reshape(-1)
            report["quality"] = {
                "roc_auc": float(roc_auc_score(holdout["truth"], logits)),
                "pr_auc": float(average_precision_score(holdout["truth"], logits)),
                "positive_baseline": float(holdout["truth"].mean()),
                "holdout_rows": len(holdout),
            }
            report["checkpoint_onnx_max_abs_error"] = errors
            if algorithm == "dnn" and root.name == "dnn":
                assert report["quality"]["roc_auc"] >= 0.95, report["quality"]
    if algorithm == "token_transformer_classifier":
        from tributo_algorithms_transformers_nlp.recipe import _model

        checkpoints = list((root / "checkpoints").rglob("model.pt"))
        assert len(checkpoints) == 1, checkpoints
        model = cast(
            torch.nn.Module,
            _model(vocab_size=32, sequence_length=4, hidden_size=8, heads=2),
        )
        model.load_state_dict(
            torch.load(checkpoints[0], map_location="cpu", weights_only=True)
        )
        model.eval()
        errors = {}
        for size in (1, 17, 257):
            feeds = feeds_for(session, size)
            with torch.no_grad():
                expected = model(torch.as_tensor(feeds["input_ids"])).numpy()
            predicted = session.run(None, feeds)[0]
            np.testing.assert_allclose(predicted, expected, rtol=1e-4, atol=1e-5)
            errors[str(size)] = float(np.max(np.abs(predicted - expected)))
        report["checkpoint_onnx_max_abs_error"] = errors
    return report


def training_parameters(case: str) -> dict[str, Any]:
    algorithm = {
        "multi_epoch": "dnn",
        "token_transformer_large_batch": "token_transformer_classifier",
    }.get(case, case)
    return {
        "algorithm": algorithm,
        "rows": 8192
        if algorithm == "pu"
        else 8193
        if case == "token_transformer_classifier"
        else 8195,
        "epochs": 3 if case == "multi_epoch" else 1,
        "batch_size": 8192 if case == "token_transformer_large_batch" else 128,
    }


def semantic_source(source: str) -> str:
    """Fingerprint executable syntax without invalidating comments or formatting."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def training_reuse_fingerprint() -> str:
    functions = (
        dense_frame,
        write_frame,
        fixture,
        execute,
        training_parameters,
        validate_receipt,
        validate_training_report,
        run_algorithm,
        run_case,
    )
    contract = {
        "schema_version": 1,
        "training_code": [semantic_source(inspect.getsource(fn)) for fn in functions],
        "requirements_sha256": hashlib.sha256(
            RUNTIME_REQUIREMENTS.read_bytes()
        ).hexdigest(),
        "python": list(sys.version_info[:3]),
        "distributions": sorted(
            (str(dist.metadata.get("Name", "")), dist.version)
            for dist in importlib.metadata.distributions()
        ),
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def validate_training_report(case: str, report: dict[str, Any]) -> None:
    """Apply the same coverage and zero-row checks to fresh and reused training."""
    parameters = training_parameters(case)
    rows = (
        31
        if parameters["algorithm"]
        in {"graphsage_node_classifier", "rgcn_node_classifier"}
        else parameters["rows"]
    )
    assert report["algorithm"] == case
    receipt = report["receipt"]
    validate_receipt(receipt, rows, 2)
    if case == "token_transformer_classifier":
        width = parameters["batch_size"]
        non_empty = [
            (w["input_rows"]["train"] + width - 1) // width for w in receipt["workers"]
        ]
        assert len(set(non_empty)) == 2
        zeros = [
            w["batch_count"] - n
            for w, n in zip(receipt["workers"], non_empty, strict=True)
        ]
        assert max(zeros) > 0 and min(zeros) >= 0
        report["empty_batch_alignment"] = {
            "non_empty_batches_by_rank": non_empty,
            "zero_steps_by_rank": zeros,
        }


def require_training_reuse(case: str, report: dict[str, Any]) -> None:
    if report.get("reuse_fingerprint") != training_reuse_fingerprint():
        raise ValueError(
            "Refusing stale or unbound training evidence; no automatic retraining"
        )
    validate_training_report(case, report)


def run_algorithm(case: str) -> dict[str, Any]:
    root = ROOT / case
    parameters = training_parameters(case)
    algorithm = parameters.pop("algorithm")
    proof = EVIDENCE / f"{case}-training.json"
    if proof.exists():
        report: dict[str, Any] = json.loads(proof.read_text())
        require_training_reuse(case, report)
        report["delivery"] = verify_bundle(report["outputs"], root, algorithm)
        report["reused_training"] = True
        return report
    fingerprint = training_reuse_fingerprint()
    request, inputs, _ = fixture(
        algorithm,
        root,
        **parameters,
    )
    start = time.perf_counter()
    result = execute(request, inputs)
    elapsed = time.perf_counter() - start
    assert result.execution.status == "succeeded"
    receipt = result.execution_receipt.to_dict()
    report = {
        "algorithm": case,
        "elapsed_s": elapsed,
        "receipt": receipt,
        "metrics": dict(result.execution.metrics),
        "outputs": dict(result.execution.outputs),
        "reuse_fingerprint": fingerprint,
    }
    validate_training_report(case, report)
    save(EVIDENCE / f"{case}-training.json", report)
    # Only small successful formal Runs are retained for downstream continuation.
    # Large benchmark data never enters this preservation path.
    shutil.copytree(root, EVIDENCE / "reusable" / case)
    report["delivery"] = verify_bundle(dict(result.execution.outputs), root, algorithm)
    return report


def transformer_unit_probe() -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "-o",
            "cache_dir=/e2e/transformer-pytest-cache",
            "--basetemp=/e2e/transformer-test-tmp",
            "--junitxml=/e2e/evidence/transformer-unit.xml",
            "tests/test_transformer_algorithm.py",
        ],
        cwd="/opt/tributo-e2e",
        env={**os.environ, "PYTHONPATH": "/opt/tributo-e2e"},
        capture_output=True,
        text=True,
        check=False,
    )
    (EVIDENCE / "transformer-unit.log").write_text(completed.stdout + completed.stderr)
    print(completed.stdout + completed.stderr, flush=True)
    assert completed.returncode == 0
    suites = list(
        ET.parse(EVIDENCE / "transformer-unit.xml").getroot().iter("testsuite")
    )
    assert sum(int(s.get("skipped", "0")) for s in suites) == 0
    return {"tests": sum(int(s.get("tests", "0")) for s in suites), "skipped": 0}


def core_runtime_profile_probe() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "tests/integrations/jobs/runtime_image_gate_job.py"],
        cwd="/opt/tributo-core-tests",
        env={
            **os.environ,
            "PYTHONPATH": "/opt/tributo-core-tests",
            "TRIBUTO_RUNTIME_PLATFORM": "linux/arm64",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    (EVIDENCE / "core-runtime-profile.log").write_text(output)
    print(output, flush=True)
    assert completed.returncode == 0
    return {"status": "passed", "driver_worker_imports_and_ray_data": True}


def transformer_ddp_worker(config: dict[str, Any]) -> None:
    """Prove zero-row DDP hooks and accumulation against an unsharded baseline."""
    import tempfile
    from contextlib import nullcontext

    import ray
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import prepare_model
    from tributo_algorithms_transformers_nlp.recipe import _model

    torch.set_num_threads(1)
    torch.manual_seed(1234)
    rank = train.get_context().get_world_rank()
    accumulation = config["accumulation"]

    def model_factory() -> Any:
        return _model(vocab_size=32, sequence_length=4, hidden_size=8, heads=2)

    def batch(owner: int, index: int) -> tuple[Any, Any]:
        sizes = (2, 1, 3) if owner == 0 else (1, 0, 0)
        tokens = torch.tensor([1 + index + 7 * owner, 2, 3, 0])
        values = tokens.repeat(sizes[index], 1)
        labels = torch.full((sizes[index], 1), float(owner))
        return values, labels

    raw = model_factory()
    initial = {name: value.detach().clone() for name, value in raw.state_dict().items()}
    model = prepare_model(raw)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    local_rows = 0
    empty_steps = 0
    optimizer_steps = 0
    for index in range(3):
        values, labels = batch(rank, index)
        local_rows += len(values)
        empty_steps += int(len(values) == 0)
        boundary = (index + 1) % accumulation == 0 or index == 2
        with nullcontext() if boundary else model.no_sync():
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(values), labels, reduction="sum"
            )
            loss.backward()
        if boundary:
            denominator = torch.tensor(local_rows, dtype=torch.float64)
            dist.all_reduce(denominator)
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                parameter.grad.mul_(2.0 / denominator.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            local_rows = 0

    reference = model_factory()
    reference.load_state_dict(initial)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    for start in range(0, 3, accumulation):
        batches = [
            batch(owner, i)
            for i in range(start, min(3, start + accumulation))
            for owner in (0, 1)
        ]
        values = torch.cat([value for value, _ in batches])
        labels = torch.cat([label for _, label in batches])
        torch.nn.functional.binary_cross_entropy_with_logits(
            reference(values), labels, reduction="mean"
        ).backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
    for name, value in raw.state_dict().items():
        torch.testing.assert_close(
            value, reference.state_dict()[name], rtol=1e-5, atol=1e-6
        )
    digest = hashlib.sha256()
    for name, value in raw.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    own = {
        "rank": rank,
        "node_id": ray.get_runtime_context().get_node_id(),
        "empty_steps": empty_steps,
        "optimizer_steps": optimizer_steps,
        "model_digest": digest.hexdigest(),
    }
    workers: list[Any] = [None, None]
    dist.all_gather_object(workers, own)
    assert len({w["node_id"] for w in workers}) == 2
    assert [w["empty_steps"] for w in workers] == [0, 2]
    assert len({w["model_digest"] for w in workers}) == 1
    with tempfile.TemporaryDirectory(dir="/e2e") as directory:
        torch.save(raw.state_dict(), Path(directory) / "model.pt")
        train.report(
            {"workers": workers, "accumulation": accumulation},
            checkpoint=Checkpoint.from_directory(directory) if rank == 0 else None,
        )


def transformer_ddp_probe() -> dict[str, Any]:
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    reports = {}
    for accumulation in (1, 2):
        result = TorchTrainer(
            transformer_ddp_worker,
            train_loop_config={"accumulation": accumulation},
            torch_config=TorchConfig(backend="gloo", timeout_s=30),
            scaling_config=ScalingConfig(
                num_workers=2,
                use_gpu=False,
                resources_per_worker={"CPU": 1},
                placement_strategy="SPREAD",
            ),
            run_config=RunConfig(storage_path=f"/e2e/transformer-ddp-{accumulation}"),
        ).fit()
        assert result.error is None, result.error
        assert result.metrics is not None
        reports[str(accumulation)] = result.metrics
    return reports


def negative_probe(case: str) -> dict[str, Any]:
    request, inputs, _ = fixture(
        "dnn",
        ROOT / case,
        rows=35,
        epochs=1,
        workers=1 if case == "one_worker" else 2,
        invalid=case,
    )
    try:
        execute(request, inputs)
    except Exception as exc:
        error = str(exc)
        # Only contract/data failures qualify; dependency or connection errors do not.
        patterns = {
            "missing_column": ("column", "schema"),
            "nan": ("finite", "NaN"),
            "inf": ("finite",),
            "invalid_label": ("0 or 1",),
            "empty": ("empty", "rows"),
            "insufficient": ("rows", "shard", "batch", "cannot cover"),
            "one_worker": ("coverage", "distributed"),
            "resume": ("resume", "recovery"),
        }
        assert any(pattern.lower() in error.lower() for pattern in patterns[case]), (
            error
        )
        return {
            "expected_rejection": True,
            "error_type": type(exc).__name__,
            "error": error,
            "bundles_left_by_rejected_run": [
                str(path)
                for path in (ROOT / case).rglob("manifest.json")
                if "/bundle/" in str(path)
            ],
        }
    raise AssertionError(f"{case}: invalid request unexpectedly succeeded")


def unit_probe() -> dict[str, Any]:
    paths = [
        "test_pytorch_e2e_harness.py",
        "test_package_conformance.py",
        "test_tabular_torch_algorithms.py",
        "test_pu_metrics.py",
        "test_pu_priors.py",
        "test_timeseries_algorithm.py",
        "test_timeseries_rnn.py",
        "test_representation_algorithm.py",
        "test_transformer_algorithm.py",
        "test_graph_pyg_algorithm.py",
        "test_recsys_torch_algorithm.py",
        "test_jagged_recsys_algorithm.py",
        "test_multistage_torch_algorithm.py",
        "test_pretrain_finetune_algorithm.py",
        "test_torch_contract_identity.py",
    ]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-o",
            "cache_dir=/e2e/pytest-cache",
            "-o",
            "filterwarnings=error\nignore::DeprecationWarning",
            "-q",
            "--junitxml=/e2e/evidence/unit.xml",
            *[f"/opt/tributo-e2e/tests/{path}" for path in paths],
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (EVIDENCE / "unit.log").write_text(completed.stdout)
    print(completed.stdout, flush=True)
    assert completed.returncode == 0
    import xml.etree.ElementTree as ET

    suites = list(ET.parse(EVIDENCE / "unit.xml").getroot().iter("testsuite"))
    assert sum(int(suite.get("skipped", "0")) for suite in suites) == 0
    return {
        "test_count": sum(int(suite.get("tests", "0")) for suite in suites),
        "skipped": 0,
    }


def core_unit_probe() -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-o",
            "cache_dir=/e2e/core-pytest-cache",
            "-q",
            "--junitxml=/e2e/evidence/core-unit.xml",
            "tests/algorithms/test_worker_environment.py",
            "tests/algorithms/test_torch_recipe_worker.py",
            "tests/test_version_compatibility.py",
        ],
        cwd="/opt/tributo-core-tests",
        env={**os.environ, "PYTHONPATH": "/opt/tributo-core-tests"},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (EVIDENCE / "core-unit.log").write_text(completed.stdout)
    print(completed.stdout, flush=True)
    assert completed.returncode == 0
    suites = list(ET.parse(EVIDENCE / "core-unit.xml").getroot().iter("testsuite"))
    assert sum(int(item.get("skipped", "0")) for item in suites) == 0
    return {"tests": sum(int(item.get("tests", "0")) for item in suites), "skipped": 0}


def delivery_probe() -> dict[str, Any]:
    import boto3
    import grpc
    import numpy as np
    import requests
    from ray import serve
    from tributo.data import IngestionRequest, ParquetSourceConfig
    from tributo.exporting.events import OperationEvent
    from tributo.exporting.hooks import BundleArtifactAccessor
    from tributo.exporting.models import BundleRef, HookStatus
    from tributo.exporting.runtime import BundleModelLoader
    from tributo.inference.api import run_inference
    from tributo.inference.contracts import (
        BundleModelReference,
        InferenceRequest,
        InputBindingSpec,
        OutputBindingSpec,
        ParquetResultSinkRequest,
        RayExecutionPolicy,
        TensorInputBinding,
        TensorOutputBinding,
    )
    from tributo.integrations.hooks.mlflow_hook import (
        MLflowHookOptions,
        MLflowPostPublishHook,
    )
    from tributo.serving.grpc_runner import start_grpc_serving
    from tributo.serving.proto import inference_pb2, inference_pb2_grpc
    from tributo.serving.serve_runner import start_serving

    training = json.loads((EVIDENCE / "dnn-training.json").read_text())
    # Reuse the formal model and input; no second training for inference testing.
    outputs = training["outputs"]
    bundle = BundleRef(
        canonical_uri=outputs["bundle_uri"],
        bundle_id=outputs["bundle_id"],
        manifest_sha256=outputs["manifest_sha256"],
    )
    ref = BundleModelReference.from_bundle_ref(bundle)
    request = InferenceRequest(
        model=ref,
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/e2e/dnn/data/test"), engine="ray"
        ),
        input_binding=InputBindingSpec(
            tensors=(
                TensorInputBinding(
                    tensor_name="features", columns=("x0", "x1"), dtype="float32"
                ),
            ),
            passthrough_columns=("row_id",),
        ),
        output_binding=OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="output",
                    column="score",
                    semantic="score",
                    squeeze_singleton=True,
                ),
            )
        ),
        result_sink=ParquetResultSinkRequest(uri="/e2e/delivery/batch"),
        execution=RayExecutionPolicy(batch_size=17, concurrency=2),
    )
    start = time.perf_counter()
    inference = run_inference(request)
    import pandas as pd

    output_frame = pd.read_parquet("/e2e/delivery/batch").sort_values("row_id")
    input_frame = pd.read_parquet("/e2e/dnn/data/test").sort_values("row_id")
    assert output_frame["row_id"].tolist() == input_frame["row_id"].tolist()
    values = input_frame[["x0", "x1"]].to_numpy(dtype=np.float32)
    with BundleModelLoader().open(bundle.canonical_uri) as runtime:
        expected = runtime.predict({"features": values})["output"].reshape(-1)
    np.testing.assert_allclose(output_frame["score"], expected, rtol=1e-4, atol=1e-5)
    report: dict[str, Any] = {
        "batch": inference.model_dump(mode="json"),
        "batch_elapsed_s": time.perf_counter() - start,
        "rows": len(output_frame),
        "source_training_elapsed_s": training["elapsed_s"],
    }

    # Publish the same trusted checkpoint to MinIO through Core's existing source/export service.
    import torch
    from tributo.exporting.models import (
        BundleOutputConfig,
        CheckpointField,
        ExportCheckpointV1,
        ExportSource,
        ExportTarget,
    )
    from tributo.exporting.service import BundleExportService
    from tributo_algorithms_tabular_torch.recipe import _model

    client = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
    client.create_bucket(Bucket="pytorch-e2e")
    model = _model({"input_features": 2, "hidden_units": [16, 8]})
    model.load_state_dict(
        torch.load(
            next((ROOT / "dnn/checkpoints").rglob("model.pt")), weights_only=True
        )
    )
    contract = ExportCheckpointV1(
        trainer_type="ray_train_torch",
        architecture_id="tributo.official.tabular_torch.dnn",
        input_schema=(
            CheckpointField(name="features", dtype="float32", shape=("batch", 2)),
        ),
        output_schema=(
            CheckpointField(name="output", dtype="float32", shape=("batch", 1)),
        ),
        task_type="binary_classification",
        framework="pytorch",
        framework_version=importlib.metadata.version("torch"),
    )
    source = ExportSource(
        source_kind="torch_module",
        model_object=model.eval(),
        sample_inputs={"features": torch.ones(2, 2)},
        feature_schema={"input_names": ["features"]},
        checkpoint_contract=contract,
    )
    published = BundleExportService().export_bundle(
        source,
        BundleOutputConfig(
            bundle_uri="s3://pytorch-e2e/model",
            targets=[
                ExportTarget(
                    name="onnx-model",
                    format="onnx",
                    exporter_id="torch-onnx-v1",
                    options={"dynamo": False},
                )
            ],
            roles={"inference": "onnx-model"},
        ),
        tributo_version=importlib.metadata.version("tributo"),
    )
    with BundleModelLoader().open(published.canonical_uri) as runtime:
        np.testing.assert_allclose(
            runtime.predict({"features": values})["output"].reshape(-1),
            expected,
            rtol=1e-4,
            atol=1e-5,
        )
    report["s3_bundle"] = published.model_dump(mode="json")
    from datetime import datetime, timezone

    event = OperationEvent.bundle_published(
        bundle_id=bundle.bundle_id,
        canonical_uri=bundle.canonical_uri,
        manifest_sha256=bundle.manifest_sha256,
        occurred_at=datetime.now(timezone.utc),
    )
    hook = MLflowPostPublishHook()
    options = MLflowHookOptions(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"], experiment_name="pytorch-e2e"
    )
    accessor = BundleArtifactAccessor(event)
    outcomes = [hook.deliver(event, accessor, options) for _ in range(2)]
    assert all(outcome.status == HookStatus.SUCCEEDED for outcome in outcomes), outcomes
    from mlflow import MlflowClient

    ml = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    experiment = ml.get_experiment_by_name("pytorch-e2e")
    runs = ml.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    report["mlflow"] = {
        "run_id": runs[0].info.run_id,
        "replay_deduplicated": True,
        "tags": runs[0].data.tags,
    }

    # Enable both transports in one Serve instance, then deploy Bundle-backed applications.
    serve.start(
        http_options={"host": "0.0.0.0", "port": 8000},
        grpc_options={
            "port": 8001,
            "grpc_servicer_functions": [
                "tributo.serving.proto.inference_pb2_grpc.add_InferenceServiceServicer_to_server"
            ],
        },
    )
    try:
        start_serving(
            bundle_uri=bundle.canonical_uri,
            app_name="e2e-http",
            route_prefix="/predict",
        )
        start_grpc_serving(
            bundle_uri=bundle.canonical_uri, app_name="e2e-grpc", grpc_port=8001
        )
        sample = values[:17]
        payload = {
            "inputs": [
                {
                    "name": "features",
                    "datatype": "float32",
                    "shape": [17, 2],
                    "data": sample.reshape(-1).tolist(),
                }
            ]
        }
        times = []
        for index in range(23):
            start = time.perf_counter()
            response = requests.post(
                "http://127.0.0.1:8000/predict", json=payload, timeout=30
            )
            response.raise_for_status()
            if index >= 3:
                times.append(time.perf_counter() - start)
            np.testing.assert_allclose(
                np.asarray(response.json()["predictions"]).reshape(-1),
                expected[:17],
                rtol=1e-4,
                atol=1e-5,
            )
        report["http"] = {
            "response": response.json(),
            "p50_ms": float(np.percentile(times, 50) * 1000),
            "p95_ms": float(np.percentile(times, 95) * 1000),
            "measured_requests": len(times),
        }
        with grpc.insecure_channel("127.0.0.1:8001") as channel:
            grpc.channel_ready_future(channel).result(timeout=30)
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            # Protobuf creates these message classes dynamically at import time.
            proto = cast(Any, inference_pb2)
            tensor = proto.InputTensor(
                schema_version=1,
                name="features",
                datatype="float32",
                shape=[17, 2],
                data=sample.reshape(-1).tolist(),
            )
            prediction = stub.Predict(
                proto.PredictRequest(inputs=[tensor]),
                metadata=(("application", "e2e-grpc"),),
                timeout=30,
            )
            assert len(prediction.predictions) == 17
            np.testing.assert_allclose(
                prediction.predictions, expected[:17], rtol=1e-4, atol=1e-5
            )
            report["grpc"] = {
                "rows": len(prediction.predictions),
                "bundle_id": prediction.bundle_id,
                "request_id": prediction.request_id,
            }
    finally:
        serve.shutdown()
    return report


def tune_probe() -> dict[str, Any]:
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
    from tributo.training.portable_tune import PortableTuneRunner
    from tributo.training.tune_config import TuneSearchConfig
    from tributo.training.tune_space import SearchParamSpec, SearchSpaceSpec
    from tributo_algorithms_tabular_torch import DNN_DESCRIPTOR

    # Tune creates separate driver actors, which use their own DataContext.
    # Four sufficiently sized files keep both ranks non-empty under defaults.
    request, inputs, _ = fixture(
        "dnn", ROOT / "tune", rows=1_000_003, epochs=1, batch_size=4096
    )
    runner = PortableTuneRunner(
        DNN_DESCRIPTOR,
        request,
        TuneSearchConfig(
            metric="train_loss",
            mode="min",
            num_samples=1,
            max_concurrent_trials=1,
            time_budget_s=180,
        ),
        SearchSpaceSpec(
            parameters=(
                SearchParamSpec(
                    path="optimizer.learning_rate",
                    kind="grid_search",
                    values=(0.005, 0.01),
                ),
            )
        ),
        InputExecutionContext(inputs),
        InputResolutionContext(values=inputs),
    )
    results = runner.run(output_path="/e2e/tune", experiment_name="cpu-trials")
    trials = list(cast(Iterable[Any], results))
    assert len(trials) == 2
    assert all(trial.error is None and trial.metrics is not None for trial in trials)
    assert not list((ROOT / "tune/bundle").rglob("manifest.json"))
    best = results.get_best_result(metric="train_loss", mode="min")
    assert best.config is not None
    from dataclasses import replace

    config = dict(request.algorithm_request.algorithm_config)
    config["optimizer"] = {"learning_rate": best.config["optimizer.learning_rate"]}
    result = execute(
        replace(
            request,
            algorithm_request=replace(
                request.algorithm_request, algorithm_config=config
            ),
        ),
        inputs,
    )
    assert result.execution.status == "succeeded"
    return {
        "trials": [
            {"config": item.config, "train_loss": item.metrics["train_loss"]}
            for item in trials
        ],
        "trial_bundles": 0,
        "formal_bundle": dict(result.execution.outputs),
    }


def integrity_probe() -> dict[str, Any]:
    from tributo.exporting.runtime import BundleModelLoader

    outputs = json.loads((EVIDENCE / "dnn-training.json").read_text())["outputs"]
    files = list((ROOT / "dnn/bundle").rglob("model.onnx"))
    assert len(files) == 1
    path = files[0]
    original = path.read_bytes()
    rejected = None
    try:
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        try:
            with BundleModelLoader().open(outputs["bundle_uri"]):
                raise AssertionError("corrupted model unexpectedly loaded")
        except Exception as exc:
            assert any(
                signal in str(exc).lower()
                for signal in ("digest", "checksum", "sha-256 mismatch")
            ), str(exc)
            rejected = {"type": type(exc).__name__, "error": str(exc)}
    finally:
        path.write_bytes(original)
    with BundleModelLoader().open(
        outputs["bundle_uri"], expected_manifest_sha256=outputs["manifest_sha256"]
    ):
        pass
    return {
        "corrupted_artifact_rejected": rejected,
        "restored_sha256": hashlib.sha256(original).hexdigest(),
    }


def kill_process(pid: int) -> None:
    import signal

    os.kill(pid, signal.SIGKILL)


def failure_probe() -> dict[str, Any]:
    from dataclasses import replace

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from ray.util.state import list_actors

    request, inputs, _ = fixture("dnn", ROOT / "failure", rows=100_003, epochs=1)
    config = dict(request.algorithm_request.algorithm_config)
    config["ray"] = {"storage_path": "/e2e/failure/checkpoints", "max_failures": 1}
    request = replace(
        request,
        algorithm_request=replace(request.algorithm_request, algorithm_config=config),
    )
    injected = []
    finished = threading.Event()
    job_id = ray.get_runtime_context().get_job_id()

    def inject() -> None:
        deadline = time.monotonic() + 120
        while not finished.wait(0.25) and time.monotonic() < deadline:
            actors = list_actors(
                address="http://ray-head:8265",
                detail=True,
                filters=[("state", "=", "ALIVE")],
            )
            candidates = [
                actor
                for actor in actors
                if "RayTrainWorker" in actor["class_name"]
                and actor["pid"]
                and actor["job_id"] == job_id
            ]
            if candidates:
                actor = candidates[0]
                killer = ray.remote(num_cpus=0)(kill_process)
                ray.get(
                    killer.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            actor["node_id"], soft=False
                        )
                    ).remote(actor["pid"])
                )
                injected.append(
                    {
                        key: actor[key]
                        for key in (
                            "actor_id",
                            "class_name",
                            "pid",
                            "node_id",
                            "job_id",
                        )
                    }
                )
                return

    thread = threading.Thread(target=inject, daemon=True)
    thread.start()
    try:
        result = execute(request, inputs)
        assert result.execution.status == "succeeded"
        assert injected, "no worker failure was actually injected"
        validate_receipt(result.execution_receipt.to_dict(), 100_003, 2)
        return {
            "injected": injected,
            "recovery_semantics": "Stage replay",
            "receipt": result.execution_receipt.to_dict(),
            "metrics": dict(result.execution.metrics),
        }
    finally:
        finished.set()
        thread.join(timeout=5)


def reference_worker(config: dict[str, Any]) -> None:
    import tempfile

    import torch
    from ray import train
    from ray.train import Checkpoint
    from ray.train.torch import prepare_model
    from tributo_algorithms_tabular_torch.recipe import _model

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = prepare_model(_model({"input_features": 2, "hidden_units": [16, 8]}))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    rows = 0
    begin = time.perf_counter()
    for batch in train.get_dataset_shard("train").iter_torch_batches(
        batch_size=config["batch_size"], drop_last=False
    ):
        features = torch.stack([batch["x0"].float(), batch["x1"].float()], dim=1)
        labels = batch["label"].float().reshape(-1, 1)
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(features), labels
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rows += len(labels)
    metrics = {
        "rows_per_worker": rows,
        "worker_training_s": time.perf_counter() - begin,
    }
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = None
        if train.get_context().get_world_rank() == 0:
            torch.save(
                getattr(model, "module", model).state_dict(),
                Path(directory) / "model.pt",
            )
            checkpoint = Checkpoint.from_directory(directory)
        train.report(metrics, checkpoint=checkpoint)


def split_probe() -> dict[str, Any]:
    import ray
    import ray.data

    path = ROOT / "split-probe-data"
    write_frame(path, dense_frame(8195, 20261023))
    context = ray.data.DataContext.get_current()
    original = context.target_max_block_size, context.target_min_block_size

    @ray.remote(num_cpus=0)
    def count(shard: Any) -> int:
        return sum(
            len(batch["row_id"]) for batch in shard.iter_batches(batch_format="numpy")
        )

    reports = {}
    try:
        for mode in ("default", "small_blocks", "repartition"):
            context.target_max_block_size, context.target_min_block_size = original
            if mode == "small_blocks":
                context.target_max_block_size = 16 * 1024
                context.target_min_block_size = 1
            dataset = ray.data.read_parquet(str(path), override_num_blocks=4)
            if mode == "repartition":
                dataset = dataset.repartition(8)
            materialized = dataset.materialize()
            counts = ray.get(
                [
                    count.remote(shard)
                    for shard in materialized.streaming_split(2, equal=False)
                ]
            )
            assert sum(counts) == 8195
            if mode == "repartition":
                assert min(counts) > 0
            reports[mode] = {
                "blocks": materialized.num_blocks(),
                "rows_by_rank": counts,
            }
    finally:
        context.target_max_block_size, context.target_min_block_size = original
    return reports


def reference_contract() -> dict[str, Any]:
    import ray.data
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    path = ROOT / "reference-contract-data"
    write_frame(path, dense_frame(1024, 20261023))
    result = TorchTrainer(
        reference_worker,
        train_loop_config={"batch_size": 128},
        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=False,
            resources_per_worker={"CPU": 1},
            placement_strategy="SPREAD",
        ),
        run_config=RunConfig(storage_path="/e2e/reference-contract-checkpoints"),
        datasets={"train": ray.data.read_parquet(str(path), override_num_blocks=4)},
    ).fit()
    assert result.error is None, result.error
    assert result.metrics is not None
    assert result.metrics["rows_per_worker"] == 512
    return {"metrics": result.metrics, "checkpoint": str(result.checkpoint)}


def performance_probe(case: str) -> dict[str, Any]:
    from dataclasses import replace

    import ray.data
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    _, row_text, worker_text, engine = case.split("_")
    rows, workers = int(row_text), int(worker_text)
    # Generate bounded chunks; never collect the training Dataset in the driver.
    root = ROOT / case
    request, inputs, _ = fixture("dnn", root, rows=35, epochs=1, workers=workers)
    path = root / "performance-data"
    path.mkdir()
    for index, start in enumerate(range(0, rows, 100_000)):
        frame = dense_frame(min(100_000, rows - start), 20261023 + index)
        frame.to_parquet(path / f"part-{index:04d}.parquet", index=False)
    from tributo.data import IngestionRequest, ParquetSourceConfig, ReadOptions
    from tributo.integrations.algorithm_inputs import IngestionInputInvocation

    inputs = {
        "train": IngestionInputInvocation(
            request=IngestionRequest(
                source=ParquetSourceConfig(path=str(path)),
                engine="ray",
                read_options=ReadOptions(target_parallelism=8),
            )
        )
    }
    config = dict(request.algorithm_request.algorithm_config)
    config["training"] = {"epochs": 1, "batch_size": 8192 // workers, "seed": 42}
    request = replace(
        request,
        algorithm_request=replace(
            request.algorithm_request,
            input_binding=request.algorithm_request.input_bindings.get("train"),
            algorithm_config=config,
        ),
    )
    begin = time.perf_counter()
    if engine == "tributo":
        result = execute(request, inputs)
        elapsed = time.perf_counter() - begin
        validate_receipt(result.execution_receipt.to_dict(), rows, workers)
        details = {
            "receipt": result.execution_receipt.to_dict(),
            "metrics": dict(result.execution.metrics),
            "bundle": dict(result.execution.outputs),
            "includes_bundle_export": True,
        }
    else:
        dataset = ray.data.read_parquet(str(path), override_num_blocks=8)
        result = TorchTrainer(
            reference_worker,
            train_loop_config={"batch_size": 8192 // workers},
            scaling_config=ScalingConfig(
                num_workers=workers,
                use_gpu=False,
                resources_per_worker={"CPU": 1},
                placement_strategy="SPREAD",
            ),
            run_config=RunConfig(storage_path=str(root / "reference-checkpoints")),
            datasets={"train": dataset},
        ).fit()
        elapsed = time.perf_counter() - begin
        assert result.error is None, result.error
        assert result.metrics is not None
        assert int(result.metrics["rows_per_worker"]) * workers == rows
        details = {"metrics": result.metrics, "includes_bundle_export": False}
    return {
        "rows": rows,
        "workers": workers,
        "engine": engine,
        "global_batch_size": 8192,
        "elapsed_s": elapsed,
        "rows_per_s": rows / elapsed,
        "comparison_boundary": "Tributo total includes planning/validation/export; Ray reference is fit-only, so ratios are lifecycle cost, not isolated framework overhead",
        **details,
    }


def run_case(case: str) -> int:
    import ray
    import ray.data

    EVIDENCE.mkdir(exist_ok=True)
    begin = time.perf_counter()
    record: dict[str, Any] = {"case": case, "status": "failed"}
    try:
        ray.init(address="auto")
        runtime_versions()
        # Keep the default large-input benchmark unchanged. Small fixtures need
        # explicit physical blocks for split_exact; the dedicated probe retains
        # and reports Ray's default one-block behavior as independent evidence.
        if case not in {
            *PERFORMANCE_CASES,
            "split_probe",
            "reference_contract",
            "package",
            "unit",
        }:
            context = ray.data.DataContext.get_current()
            context.target_max_block_size = (
                64
                if case in {"graphsage_node_classifier", "rgcn_node_classifier"}
                else 16 * 1024
            )
            context.target_min_block_size = 1
            record["ray_data_context"] = {
                "target_max_block_size": context.target_max_block_size,
                "target_min_block_size": context.target_min_block_size,
            }
        if case in ALGORITHMS or case in {
            "multi_epoch",
            "token_transformer_large_batch",
        }:
            details = run_algorithm(case)
        elif case in NEGATIVE_CASES:
            details = negative_probe(case)
        elif case in PERFORMANCE_CASES:
            details = performance_probe(case)
        elif case == "stop_target":
            request, inputs, _ = fixture("dnn", ROOT / case, rows=1003, epochs=100_000)
            details = execute(request, inputs)
        else:
            details = {
                "package": package_probe,
                "core_unit": core_unit_probe,
                "unit": unit_probe,
                "integrity": integrity_probe,
                "delivery": delivery_probe,
                "tune": tune_probe,
                "failure": failure_probe,
                "split_probe": split_probe,
                "reference_contract": reference_contract,
                "transformer_unit": transformer_unit_probe,
                "transformer_ddp_alignment": transformer_ddp_probe,
                "core_runtime_profile": core_runtime_profile_probe,
            }[case]()
        record.update(status="passed", details=details)
    except Exception as exc:
        record.update(
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    finally:
        record["elapsed_s"] = time.perf_counter() - begin
        save(EVIDENCE / f"{case}.json", record)
        ray.shutdown()
    print(
        json.dumps(
            {"case": case, "status": record["status"], "elapsed_s": record["elapsed_s"]}
        ),
        flush=True,
    )
    return 0 if record["status"] == "passed" else 1


def orchestrate(cases: list[str]) -> int:
    import requests
    from ray.job_submission import JobStatus, JobSubmissionClient

    EVIDENCE.mkdir(exist_ok=True)
    if os.environ.get("TRIBUTO_E2E_REUSE_ENABLED") == "1":
        source = Path("/opt/tributo-e2e/reuse")
        reused = []
        for proof in source.glob("*-training.json"):
            case = proof.name.removesuffix("-training.json")
            if case not in cases:
                continue
            payload = source / "reusable" / case
            if not payload.is_dir():
                continue
            report = json.loads(proof.read_text())
            require_training_reuse(case, report)
            from tributo.exporting.runtime import BundleModelLoader

            shutil.copytree(payload, ROOT / case)
            with BundleModelLoader().open(
                report["outputs"]["bundle_uri"],
                expected_manifest_sha256=report["outputs"]["manifest_sha256"],
            ):
                pass
            for name in (proof.name, f"{case}.json", f"{case}-data.json"):
                if (source / name).is_file():
                    shutil.copy2(source / name, EVIDENCE / name)
            # Preserve one verified copy until every dependent check completes.
            shutil.copytree(payload, EVIDENCE / "reusable" / case)
            reused.append(case)
        save(EVIDENCE / "reused-training.json", reused)
    client = JobSubmissionClient("http://127.0.0.1:8265")
    summary = []
    stopping = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}
    for case in cases:
        assert case in DEFAULT_CASES, case
        submission = f"pytorch-{case.replace('_', '-')}-{time.time_ns()}"
        target = "stop_target" if case == "stop" else case
        job_id = client.submit_job(
            entrypoint=f"python {JOB} --case {target}", submission_id=submission
        )
        print(f"START {case} {job_id}", flush=True)
        begin = time.monotonic()
        stopped = False
        samples = []
        while client.get_job_status(job_id) not in stopping:
            elapsed = time.monotonic() - begin
            if case == "stop" and elapsed > 20 and not stopped:
                client.stop_job(job_id)
                stopped = True
            if elapsed > 600:
                client.stop_job(job_id)
                stopped = True
                break
            # Ray dashboard provides per-node CPU/memory observations without a host Docker socket.
            try:
                observed = requests.get(
                    "http://127.0.0.1:8265/api/cluster_status", timeout=2
                )
                samples.append(
                    {
                        "elapsed_s": elapsed,
                        "status_code": observed.status_code,
                        "status": observed.json(),
                    }
                )
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2)
        status = client.get_job_status(job_id)
        (EVIDENCE / f"{case}-job.log").write_text(client.get_job_logs(job_id))
        save(EVIDENCE / f"{case}-resource-samples.json", samples)
        if case == "stop":
            deadline = time.monotonic() + 30
            while status not in stopping and time.monotonic() < deadline:
                time.sleep(1)
                status = client.get_job_status(job_id)
            record = {
                "case": case,
                "status": "passed"
                if stopped and status == JobStatus.STOPPED
                else "failed",
                "job_status": str(status),
                "stop_requested": stopped,
            }
            save(EVIDENCE / "stop.json", record)
        elif (EVIDENCE / f"{case}.json").exists():
            record = json.loads((EVIDENCE / f"{case}.json").read_text())
        else:
            record = {
                "case": case,
                "status": "failed",
                "error": "Job ended without a result; inspect job log",
                "job_status": str(status),
            }
            save(EVIDENCE / f"{case}.json", record)
        if status != JobStatus.SUCCEEDED and case != "stop":
            record["status"] = "failed"
        summary.append(
            {
                "case": case,
                "job_id": job_id,
                "job_status": str(status),
                "status": record["status"],
                "elapsed_s": time.monotonic() - begin,
            }
        )
        save(EVIDENCE / "summary.json", summary)
        print("END " + json.dumps(summary[-1]), flush=True)
        if "node running low on memory" in str(record.get("error", "")):
            print("Stopping this diagnostic round after infrastructure OOM", flush=True)
            break
    return 0 if all(item["status"] == "passed" for item in summary) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", action="store_true")
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--cases", nargs="+")
    arguments = parser.parse_args()
    if arguments.ready:
        ready()
        return 0
    if arguments.orchestrate:
        return orchestrate(arguments.cases or list(DEFAULT_CASES))
    assert arguments.case
    return run_case(arguments.case)


if __name__ == "__main__":
    raise SystemExit(main())
