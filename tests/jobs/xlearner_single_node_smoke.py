"""Run X-Learner and held-out effect evaluation in a single-node Ray Job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
from tributo.exporting.models import BundleRef
from tributo.exporting.runtime import BundleModelLoader
from tributo.integrations.algorithm_inputs import (
    INGESTION_RESOLVER_ID,
    IngestionInputInvocation,
)
from tributo_algorithms_causal_xlearner import (
    CATEEvaluationResult,
    RCTEvaluationResult,
    evaluate_cate_predictions,
    evaluate_rct_predictions,
)
from xlearner_evaluation_checks import verify_partitioned_evaluation

ACIC_SOURCE_URL = "https://jenniferhill7.wixsite.com/acic-2016/competition"
EVALUATION_COLUMNS = {"sample_id", "treatment", "outcome", "mu0", "mu1"}
FEATURE_COLUMNS = ("x0", "x1", "x2")
WORKER_COUNT = 2
FOLD_COUNT = 2


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_versions(
    *,
    expected_python_version: str,
    expected_tributo_version: str,
    expected_ray_version: str,
) -> dict[str, str]:
    versions = {
        "python": ".".join(str(part) for part in sys.version_info[:2]),
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


def _synthetic_frame(*, rows: int, seed: int, id_prefix: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = rng.uniform(-1.0, 1.0, size=(rows, len(FEATURE_COLUMNS)))
    treatment = np.tile(np.asarray([0, 1], dtype=np.int8), rows // 2 + 1)[:rows]
    rng.shuffle(treatment)
    mu0 = 1.0 + 0.8 * features[:, 0] - 0.3 * features[:, 1]
    treatment_effect = 0.5 + 0.7 * features[:, 0] - 0.4 * features[:, 2]
    mu1 = mu0 + treatment_effect
    outcome = np.where(treatment == 1, mu1, mu0) + rng.normal(0.0, 0.05, rows)
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    frame.insert(0, "sample_id", [f"{id_prefix}-{index}" for index in range(rows)])
    frame["treatment"] = treatment
    frame["outcome"] = outcome
    frame["mu0"] = mu0
    frame["mu1"] = mu1
    return frame


def _write_train_dataset(root: Path, frame: pd.DataFrame) -> tuple[Path, int]:
    train_path = root / "train"
    train_path.mkdir()
    for index, rows in enumerate(np.array_split(np.arange(len(frame)), 4)):
        frame.iloc[rows].to_parquet(train_path / f"part-{index}.parquet", index=False)
    return train_path, len(frame)


def _fixture_data(
    root: Path,
) -> tuple[Path, int, pd.DataFrame, tuple[str, ...], dict[str, object]]:
    train = _synthetic_frame(rows=512, seed=7, id_prefix="train")
    evaluation = _synthetic_frame(rows=128, seed=17, id_prefix="holdout")
    train_path, training_rows = _write_train_dataset(
        root,
        train.loc[:, [*FEATURE_COLUMNS, "sample_id", "treatment", "outcome"]],
    )
    return (
        train_path,
        training_rows,
        evaluation,
        FEATURE_COLUMNS,
        {
            "kind": "deterministic_fixture",
            "generator_seed": 7,
            "evaluation_seed": 17,
            "split_seed": 7,
            "schema_mapping": {
                "sample_id": "sample_id",
                "treatment": "treatment",
                "outcome": "outcome",
                "mu0": "mu0",
                "mu1": "mu1",
                "features": list(FEATURE_COLUMNS),
            },
            "training_rows": training_rows,
            "evaluation_rows": int(len(evaluation)),
        },
    )


def _read_acic_data(
    path: Path,
    *,
    expected_sha256: str,
    setting_id: int,
    replication: int,
    split_seed: int,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"ACIC input file does not exist: {path}")
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in expected_sha256
    ):
        raise ValueError("ACIC input checksum must be a 64-character SHA-256 digest")
    if setting_id < 1 or replication < 1:
        raise ValueError("ACIC setting and replication IDs must be positive")
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"ACIC input checksum mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("ACIC input must be a prepared CSV or Parquet file")

    if not frame.columns.is_unique or any(
        not isinstance(column, str) for column in frame.columns
    ):
        raise ValueError("ACIC input columns must be unique strings")
    missing = sorted(EVALUATION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"ACIC input is missing required columns: {missing}")
    feature_columns = sorted(
        (
            column
            for column in frame.columns
            if column.startswith("x") and column[1:].isdigit()
        ),
        key=lambda column: int(column[1:]),
    )
    unexpected = sorted(set(frame.columns) - EVALUATION_COLUMNS - set(feature_columns))
    if unexpected:
        raise ValueError(
            "ACIC input contains unmapped columns that could leak truth data: "
            f"{unexpected}"
        )
    if len(feature_columns) < 3:
        raise ValueError(
            "ACIC input requires at least three numeric x0, x1, ... columns"
        )
    if feature_columns != [f"x{index}" for index in range(len(feature_columns))]:
        raise ValueError("ACIC feature columns must be contiguous from x0")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("ACIC sample_id values must be present and unique")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (str, int, np.integer))
        or (isinstance(value, str) and not value)
        for value in frame["sample_id"].tolist()
    ):
        raise ValueError("ACIC sample_id values must be non-empty strings or integers")

    numeric_columns = [*feature_columns, "treatment", "outcome", "mu0", "mu1"]
    for column in numeric_columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"ACIC column {column!r} must be numeric") from error
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"ACIC column {column!r} must contain finite values")
    treatment = frame["treatment"].to_numpy(dtype=np.float64)
    if not np.isin(treatment, (0.0, 1.0)).all():
        raise ValueError("ACIC treatment values must be exactly 0 or 1")
    if len(frame) < 40 or frame["treatment"].value_counts().min() < 10:
        raise ValueError("ACIC input is too small for the fixed train/evaluation split")

    return (
        frame,
        tuple(feature_columns),
        {
            "kind": "acic_2016",
            "source_url": ACIC_SOURCE_URL,
            "setting_id": setting_id,
            "replication": replication,
            "split_seed": split_seed,
            "input_sha256": actual_sha256,
            "input_file": path.name,
            "rows": int(len(frame)),
            "feature_columns": feature_columns,
            "schema_mapping": {
                "sample_id": "sample_id",
                "treatment": "treatment",
                "outcome": "outcome",
                "mu0": "mu0",
                "mu1": "mu1",
                "features": feature_columns,
            },
        },
    )


def _split_acic(
    frame: pd.DataFrame, *, split_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treatment = frame["treatment"].to_numpy(dtype=np.int8)
    rng = np.random.default_rng(split_seed)
    train_indices: list[int] = []
    evaluation_indices: list[int] = []
    for treatment_value in (0, 1):
        arm_indices = np.flatnonzero(treatment == treatment_value)
        arm_indices = rng.permutation(arm_indices)
        evaluation_rows = max(1, int(np.floor(len(arm_indices) * 0.2)))
        evaluation_indices.extend(int(index) for index in arm_indices[:evaluation_rows])
        train_indices.extend(int(index) for index in arm_indices[evaluation_rows:])

    train = frame.iloc[sorted(train_indices)].reset_index(drop=True)
    evaluation = frame.iloc[sorted(evaluation_indices)].reset_index(drop=True)
    train_counts = train["treatment"].value_counts()
    evaluation_counts = evaluation["treatment"].value_counts()
    if min(int(train_counts.get(value, 0)) for value in (0, 1)) < 4:
        raise ValueError(
            "ACIC training split needs at least four rows per treatment arm"
        )
    if min(int(evaluation_counts.get(value, 0)) for value in (0, 1)) < 1:
        raise ValueError("ACIC evaluation split needs both treatment arms")
    return train, evaluation


def _run_fit(
    root: Path,
    train_path: Path,
    feature_columns: tuple[str, ...],
    expected_rows: int,
) -> dict[str, Any]:
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
            algorithm="x_learner",
            operation=AlgorithmOperation.FIT,
            input_binding=InputBinding(
                name="train",
                resolver_id=INGESTION_RESOLVER_ID,
                reference="train-input",
                feature_names=(*feature_columns, "treatment", "sample_id"),
                label_name="outcome",
            ),
            algorithm_config={
                "data": {
                    "feature_columns": list(feature_columns),
                    "identity_col": "sample_id",
                    "outcome_col": "outcome",
                    "treatment_col": "treatment",
                },
                "model": {
                    "outcome": {"eta": 0.2, "max_depth": 2, "nthread": 1, "seed": 7},
                    "effect": {"eta": 0.2, "max_depth": 2, "nthread": 1, "seed": 7},
                    "propensity": {
                        "eta": 0.2,
                        "max_depth": 2,
                        "nthread": 1,
                        "seed": 7,
                    },
                },
                "training": {"cross_fit_folds": FOLD_COUNT, "num_rounds": 2, "seed": 7},
                "ray": {"storage_path": str(root / "checkpoints")},
                "output": {"bundle_uri": str(root / "bundle")},
            },
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=WORKER_COUNT,
    )
    result = build_algorithm_dispatcher().execute(
        request,
        InputExecutionContext(inputs),
        resolution_context=InputResolutionContext(values=inputs),
    )
    if result.execution.status != "succeeded":
        raise AssertionError(
            f"X-Learner fit did not succeed: {result.execution.status}"
        )
    fit_metrics = dict(result.execution.metrics)
    if (
        int(fit_metrics.get("treated_rows", -1))
        + int(fit_metrics.get("control_rows", -1))
        != expected_rows
    ):
        raise AssertionError("X-Learner fit metrics do not cover all training rows")
    if int(fit_metrics.get("cross_fit_folds", -1)) != FOLD_COUNT:
        raise AssertionError("X-Learner used an unexpected cross-fit fold count")
    if result.execution_receipt is None:
        raise AssertionError("X-Learner did not produce an execution receipt")
    receipt = result.execution_receipt.to_dict()
    details = result.execution_receipt.state.details
    expected_stage_rows = {
        "mu0": int(fit_metrics["control_rows"]),
        "mu1": int(fit_metrics["treated_rows"]),
        "tau0": int(fit_metrics["control_rows"]),
        "tau1": int(fit_metrics["treated_rows"]),
        "propensity": expected_rows,
    }
    for stage, expected in expected_stage_rows.items():
        if details.get(f"stage.{stage}.rows") != expected:
            raise AssertionError(
                f"Final-fit receipt has incorrect coverage for {stage}"
            )
        if f"stage.{stage}.input_rows_scope" in details:
            raise AssertionError(
                "Final Bundle receipt must describe final fits, not OOF aggregates"
            )
    outputs = dict(result.execution.outputs)
    bundle_ref = BundleRef(
        canonical_uri=outputs["bundle_uri"],
        bundle_id=outputs["bundle_id"],
        manifest_sha256=outputs["manifest_sha256"],
    )
    return {
        "bundle": bundle_ref,
        "fit_metrics": fit_metrics,
        "outputs": outputs,
        "receipt": receipt,
    }


def _evaluate_bundle(
    bundle: BundleRef,
    evaluation: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[CATEEvaluationResult, RCTEvaluationResult]:
    loader = BundleModelLoader()
    with loader.open(
        bundle.canonical_uri,
        role="inference",
        expected_manifest_sha256=bundle.manifest_sha256,
        use_case="batch",
    ) as model:
        prediction = model.predict(
            {
                "float_input": evaluation.loc[:, list(feature_columns)].to_numpy(
                    dtype=np.float32
                )
            }
        )
        # The separate, randomized binary fixture exercises the public RCT API.
        # It is a protocol check, not a claim about model quality on this cohort.
        rct_frame = _synthetic_frame(rows=128, seed=27, id_prefix="rct-holdout")
        rct_values = rct_frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        # ACIC may have more features; this fixture tests the same Bundle schema.
        if len(feature_columns) != len(FEATURE_COLUMNS):
            rct_values = (
                np.random.default_rng(27)
                .uniform(-1, 1, (128, len(feature_columns)))
                .astype(np.float32)
            )
        rct_prediction = model.predict({"float_input": rct_values})
        rct_treatment = rct_frame["treatment"].to_numpy()
        response_probability = np.clip(
            0.2 + 0.1 * rct_values[:, 0] + 0.2 * rct_treatment, 0.01, 0.99
        )
        rct_outcome = np.random.default_rng(37).binomial(1, response_probability)
        rct_result = evaluate_rct_predictions(
            prediction_ids=rct_frame["sample_id"].to_numpy(),
            cate_predictions=rct_prediction["cate"],
            observation_ids=rct_frame["sample_id"].to_numpy(),
            treatment=rct_treatment,
            outcome=rct_outcome,
            assignment_probability=0.5,
            experiment_id="deterministic-binary-rct-fixture",
            policy_coverage=0.25,
            n_bins=4,
            n_bootstrap=200,
            seed=7,
        )
    truth_result = evaluate_cate_predictions(
        prediction_ids=evaluation["sample_id"].to_numpy(),
        cate_predictions=prediction["cate"],
        truth_ids=evaluation["sample_id"].to_numpy(),
        mu0=evaluation["mu0"].to_numpy(dtype=np.float64),
        mu1=evaluation["mu1"].to_numpy(dtype=np.float64),
    )
    return truth_result, rct_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--expected-tributo-version", required=True)
    parser.add_argument("--expected-ray-version", required=True)
    parser.add_argument("--expected-ray-image", required=True)
    parser.add_argument("--acic-data", type=Path)
    parser.add_argument("--acic-sha256")
    parser.add_argument("--acic-setting-id", type=int)
    parser.add_argument("--acic-replication", type=int)
    parser.add_argument("--split-seed", type=int, default=7)
    args = parser.parse_args()
    if args.split_seed < 0:
        raise ValueError("split seed must be non-negative")

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

    if args.acic_data is None:
        if any(
            value is not None
            for value in (args.acic_sha256, args.acic_setting_id, args.acic_replication)
        ):
            raise ValueError("ACIC metadata flags require --acic-data")
        (
            train_path,
            training_rows,
            evaluation,
            feature_columns,
            dataset_metadata,
        ) = _fixture_data(args.root)
    else:
        if (
            not args.acic_sha256
            or args.acic_setting_id is None
            or args.acic_replication is None
        ):
            raise ValueError(
                "ACIC mode requires --acic-sha256, --acic-setting-id, and --acic-replication"
            )
        frame, feature_columns, dataset_metadata = _read_acic_data(
            args.acic_data,
            expected_sha256=args.acic_sha256,
            setting_id=args.acic_setting_id,
            replication=args.acic_replication,
            split_seed=args.split_seed,
        )
        train_frame, evaluation = _split_acic(frame, split_seed=args.split_seed)
        train_path, training_rows = _write_train_dataset(
            args.root,
            train_frame.loc[:, [*feature_columns, "sample_id", "treatment", "outcome"]],
        )
        dataset_metadata["split_seed"] = args.split_seed
        dataset_metadata["training_rows"] = training_rows
        dataset_metadata["evaluation_rows"] = int(len(evaluation))

    ray.init(address="auto")
    try:
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        if len(alive_nodes) != 1:
            raise AssertionError(f"expected one live Ray node, got {len(alive_nodes)}")
        result = _run_fit(args.root, train_path, feature_columns, training_rows)
        metrics, rct_metrics = _evaluate_bundle(
            result["bundle"], evaluation, feature_columns
        )
        partition_checks = verify_partitioned_evaluation()
        report = {
            "runtime": {"versions": versions, "ray_image": args.expected_ray_image},
            "cluster": {
                "alive_ray_nodes": len(alive_nodes),
                "worker_count": WORKER_COUNT,
            },
            "dataset": {
                **dataset_metadata,
                "training_rows": training_rows,
                "evaluation_rows": int(len(evaluation)),
            },
            "bundle": result["outputs"],
            "execution_receipt": result["receipt"],
            "fit_metrics": result["fit_metrics"],
            "evaluation": metrics,
            "rct_evaluation": rct_metrics,
            "partition_checks": partition_checks,
        }
        (args.root / "result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("RESULT: " + json.dumps(report, sort_keys=True), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
