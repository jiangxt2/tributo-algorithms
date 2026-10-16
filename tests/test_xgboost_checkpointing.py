"""Native continuation and checkpoint/evidence consistency regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xgboost
from tributo.algorithms.api import AlgorithmExecutionError
from tributo_algorithms_boosting.algorithm import DistributedXGBoost, _EvidenceCollector
from tributo_algorithms_boosting.checkpointing import (
    REPORT_KEY,
    check_final_result,
    evidence_callback,
    final_report_group,
    read_checkpoint,
    recovery_identity,
    training_callback,
    write_checkpoint,
)
from tributo_algorithms_boosting.contracts import (
    XGBoostConfigValidator,
    XGBoostOutputValidator,
)


def _matrix() -> Any:
    features = np.linspace(-2, 2, 32, dtype=np.float32).reshape(-1, 1)
    return xgboost.DMatrix(features, label=(features[:, 0] > 0).astype(int))


@pytest.mark.parametrize(
    "scores,split,patience,maximize",
    [
        ([0.5, 0.3, 0.4, 0.4], 3, 2, False),
        ([0.5, 0.7, 0.6, 0.6], 3, 2, True),
        ([0.5, 0.3, 0.4, 0.2, 0.3, 0.4], 3, 2, False),
        ([0.5, 0.5, 0.5], 2, 2, False),
    ],
)
def test_native_history_replay_preserves_patience_and_best(
    scores: list[float], split: int, patience: int, maximize: bool
) -> None:
    matrix = _matrix()
    params = {"objective": "binary:logistic", "nthread": 1, "seed": 7}

    def fit(model: Any, start: int, rounds: int, callback: Any) -> Any:
        values = iter(scores[start:])
        return xgboost.train(
            {**params, "disable_default_eval_metric": True},
            matrix,
            xgb_model=model,
            num_boost_round=rounds,
            evals=[(matrix, "validation")],
            custom_metric=lambda pred, data: ("fixture", next(values)),
            callbacks=[callback],
            verbose_eval=False,
        )

    baseline = training_callback(
        dtrain=matrix, patience=patience, maximize=maximize, interval=0
    )
    expected = fit(None, 0, len(scores), baseline)
    saved: dict[str, Any] = {}

    def save(model: Any, state: Any) -> None:
        saved.update(
            completed_rounds=state.completed,
            scores=list(state.scores),
            metric=state.metric,
            last_metrics=dict(state.last_metrics),
            stopped=state.stopped,
            raw=bytes(model.save_raw(raw_format="ubj")),
        )

    first = training_callback(
        dtrain=matrix, patience=patience, maximize=maximize, interval=split, report=save
    )
    fit(None, 0, split, first)
    full = xgboost.Booster(model_file=bytearray(saved.pop("raw")))
    resumed = training_callback(
        dtrain=matrix, patience=patience, maximize=maximize, interval=0, restored=saved
    )
    actual = fit(full, split, len(scores) - split, resumed)
    assert resumed.completed == baseline.completed
    assert actual.best_iteration == expected.best_iteration
    assert actual.best_score == expected.best_score
    assert bytes(actual.save_raw(raw_format="ubj")) == bytes(
        expected.save_raw(raw_format="ubj")
    )


@pytest.mark.parametrize("early", [False, True])
def test_terminal_restore_adds_zero_rounds(early: bool) -> None:
    matrix = _matrix()
    callback = training_callback(
        dtrain=matrix,
        patience=1 if early else None,
        maximize=False if early else None,
        interval=0,
    )
    params = {"objective": "binary:logistic", "nthread": 1}
    selected = xgboost.train(
        {**params, "disable_default_eval_metric": True},
        matrix,
        num_boost_round=2,
        evals=[(matrix, "validation")],
        custom_metric=lambda pred, data: ("fixture", 0.5),
        callbacks=[callback],
        verbose_eval=False,
    )
    restored = {
        "completed_rounds": callback.completed,
        "scores": callback.scores,
        "metric": callback.metric,
        "last_metrics": callback.last_metrics,
        "stopped": callback.stopped,
    }
    resumed = training_callback(
        dtrain=matrix,
        patience=1 if early else None,
        maximize=False if early else None,
        interval=0,
        restored=restored,
    )
    actual = xgboost.train(
        params,
        matrix,
        xgb_model=bytearray(callback.full_model),
        num_boost_round=0,
        callbacks=[resumed],
        verbose_eval=False,
    )
    assert resumed.completed == resumed.start == 2
    assert actual.num_boosted_rounds() == selected.num_boosted_rounds()
    assert actual.save_raw(raw_format="ubj") == selected.save_raw(raw_format="ubj")


def _reports(digest: str = "digest", rounds: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "training_rounds": 20,
            "model_rounds": 20,
            "model_state_digest": digest,
            "attempt_training_rounds": rounds,
            "resumed_from_round": 20 - rounds,
            REPORT_KEY: {
                "invocation": "run",
                "phase": "final",
                "worker": {
                    "rank": rank,
                    "world_size": 2,
                    "model_state_digest": digest,
                    "collective_steps": rounds,
                    "batch_count": rounds,
                },
            },
        }
        for rank in range(2)
    ]


def test_final_evidence_requires_one_complete_consistent_report() -> None:
    reports = _reports()
    checkpoint = SimpleNamespace(path="/persisted/final")
    group = final_report_group(reports, checkpoint, invocation="run", world_size=2)
    assert group is not None and len(group["workers"]) == 2
    assert group["checkpoint_path"] == checkpoint.path
    assert REPORT_KEY not in group["metrics"]
    mutations = [reports[:1], deepcopy(reports), deepcopy(reports), deepcopy(reports)]
    mutations[1][1][REPORT_KEY]["invocation"] = "other"
    mutations[2][1][REPORT_KEY]["worker"]["model_state_digest"] = "old"
    mutations[3][1]["attempt_training_rounds"] = 1
    for items in mutations:
        with pytest.raises(AlgorithmExecutionError):
            final_report_group(items, checkpoint, invocation="run", world_size=2)
    with pytest.raises(AlgorithmExecutionError):
        final_report_group(reports, None, invocation="run", world_size=2)


def _identity() -> dict[str, Any]:
    return recovery_identity(
        {
            "invocation": "fit",
            "plan_id": "plan",
            "feature_names": ["f0"],
            "label_name": "label",
            "input_roles": ["train"],
        },
        2,
    )


@pytest.fixture
def checkpoint_files(
    tmp_path: Path, request: pytest.FixtureRequest
) -> tuple[Path, dict[str, Any]]:
    matrix = _matrix()
    state = training_callback(dtrain=matrix, patience=None, maximize=None, interval=0)
    model = xgboost.train(
        {
            "objective": "binary:logistic",
            "nthread": 1,
            **(
                {"disable_default_eval_metric": True}
                if getattr(request, "param", False)
                else {}
            ),
        },
        matrix,
        num_boost_round=2,
        evals=[(matrix, "train")],
        callbacks=[state],
        verbose_eval=False,
    )
    identity = _identity()
    write_checkpoint(
        tmp_path,
        full_model=state.full_model,
        state=state,
        identity=identity,
        selected=bytes(model.save_raw(raw_format="ubj")),
        metadata={"schema_version": 1, "training_rounds": 2, "model_rounds": 2},
    )
    return tmp_path, identity


def test_checkpoint_roundtrip_retains_full_and_final_model(
    checkpoint_files: Any,
) -> None:
    root, identity = checkpoint_files
    raw, state = read_checkpoint(root, identity=identity, rounds=2, patience=None)
    assert raw == (root / "training.ubj").read_bytes()
    assert state["phase"] == "final" and state["completed_rounds"] == 2


@pytest.mark.parametrize("phase", ["training", "final"])
def test_checkpoint_roundtrip_without_evaluation_metrics(
    tmp_path: Path, phase: str
) -> None:
    matrix = _matrix()
    state = training_callback(dtrain=matrix, patience=None, maximize=None, interval=0)
    model = xgboost.train(
        {
            "objective": "binary:logistic",
            "nthread": 1,
            "disable_default_eval_metric": True,
        },
        matrix,
        num_boost_round=2,
        evals=[(matrix, "train")],
        callbacks=[state],
        verbose_eval=False,
    )
    assert state.last_metrics == {} and state.completed == 2
    identity = _identity()
    write_checkpoint(
        tmp_path,
        full_model=state.full_model,
        state=state,
        identity=identity,
        selected=bytes(model.save_raw(raw_format="ubj")) if phase == "final" else None,
        metadata={"schema_version": 1, "training_rounds": 2, "model_rounds": 2}
        if phase == "final"
        else None,
    )
    raw, restored = read_checkpoint(
        tmp_path, identity=identity, rounds=2, patience=None
    )
    assert raw == state.full_model
    assert restored["phase"] == phase and restored["last_metrics"] == {}


def test_early_stopping_checkpoint_rejects_missing_selection_metric(
    tmp_path: Path,
) -> None:
    matrix = _matrix()
    state = training_callback(dtrain=matrix, patience=2, maximize=False, interval=0)
    xgboost.train(
        {"objective": "binary:logistic", "nthread": 1},
        matrix,
        num_boost_round=2,
        evals=[(matrix, "train"), (matrix, "validation")],
        callbacks=[state],
        verbose_eval=False,
    )
    identity = {**_identity(), "input_roles": ["train", "validation"]}
    write_checkpoint(
        tmp_path, full_model=state.full_model, state=state, identity=identity
    )
    path = tmp_path / "recovery_state.json"
    saved = json.loads(path.read_text())
    saved["last_metrics"] = {}
    path.write_text(json.dumps(saved))
    with pytest.raises(AlgorithmExecutionError, match="checkpoint is invalid") as error:
        read_checkpoint(
            tmp_path, identity=identity, rounds=2, patience=2, maximize=False
        )
    assert "last selection score mismatch" in str(error.value.__cause__)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("phase", "export"),
        ("completed_rounds", True),
        ("completed_rounds", 3),
        ("completed_rounds", 1),
        ("scores", [0.5]),
        ("stopped", True),
        ("last_metrics", None),
        ("last_metrics", {"train-logloss": float("nan")}),
        ("training_sha256", "corrupt"),
        ("selected_sha256", "corrupt"),
        ("extra", "key"),
        ("identity", {}),
    ],
)
def test_checkpoint_rejects_inconsistent_state(
    checkpoint_files: Any, field: str, value: Any
) -> None:
    root, identity = checkpoint_files
    path = root / "recovery_state.json"
    state = json.loads(path.read_text())
    state[field] = value
    path.write_text(json.dumps(state))
    with pytest.raises(AlgorithmExecutionError, match="checkpoint is invalid"):
        read_checkpoint(root, identity=identity, rounds=2, patience=None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("invocation", "other-fit"),
        ("plan_id", "changed-input-or-config"),
        ("world_size", 3),
        ("feature_names", ["changed"]),
        ("label_name", "changed"),
        ("input_roles", ["train", "validation"]),
        ("versions", {}),
    ],
)
def test_checkpoint_rejects_foreign_identity(
    checkpoint_files: Any, field: str, value: Any
) -> None:
    root, identity = checkpoint_files
    identity[field] = value
    with pytest.raises(AlgorithmExecutionError, match="checkpoint is invalid"):
        read_checkpoint(root, identity=identity, rounds=2, patience=None)


@pytest.mark.parametrize(
    "filename",
    [
        "recovery_state.json",
        "feature_names.json",
        "training.ubj",
        "model.ubj",
        "training_metadata.json",
    ],
)
def test_checkpoint_rejects_corrupt_files(checkpoint_files: Any, filename: str) -> None:
    root, identity = checkpoint_files
    (root / filename).write_bytes(b"corrupt")
    with pytest.raises(AlgorithmExecutionError, match="checkpoint is invalid"):
        read_checkpoint(root, identity=identity, rounds=2, patience=None)


@pytest.mark.parametrize(
    "field,value", [("scores", [0.5]), ("metric", None), ("stopped", True)]
)
def test_native_replay_rejects_inconsistent_early_state(field: str, value: Any) -> None:
    matrix = _matrix()
    state = training_callback(dtrain=matrix, patience=5, maximize=False, interval=0)
    xgboost.train(
        {"nthread": 1},
        matrix,
        num_boost_round=2,
        evals=[(matrix, "validation")],
        callbacks=[state],
        verbose_eval=False,
    )
    saved = {
        "completed_rounds": 2,
        "metric": state.metric,
        "scores": state.scores,
        "last_metrics": state.last_metrics,
        "stopped": False,
    }
    saved[field] = value
    # The persisted-state reader handles shape/type errors; native replay checks
    # consistency with Booster best attrs and the stop decision.
    if field == "scores":
        saved["scores"] = [state.scores[0], state.scores[0] - 0.1]
    callback = training_callback(
        dtrain=matrix, patience=5, maximize=False, interval=0, restored=saved
    )
    with pytest.raises((AlgorithmExecutionError, ValueError)):
        xgboost.train(
            {"nthread": 1},
            matrix,
            num_boost_round=0,
            xgb_model=bytearray(state.full_model),
            callbacks=[callback],
        )


@pytest.mark.parametrize("interval,failures", [(0, 0), (10, 0), (10, 1), (100, 1)])
def test_recovery_config_accepts_bounded_combinations(
    interval: int, failures: int
) -> None:
    config = {
        "data": {"label_col": "label"},
        "output": {"bundle_uri": "/bundle"},
        "training": {"num_rounds": 20, "checkpoint_interval_rounds": interval},
        "ray": {"storage_path": "/shared", "max_failures": failures},
    }
    assert XGBoostConfigValidator().validate(config) == config


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("training", "checkpoint_interval_rounds", True),
        ("training", "checkpoint_interval_rounds", -1),
        ("training", "checkpoint_interval_rounds", 1.5),
        ("ray", "max_failures", True),
        ("ray", "max_failures", -1),
        ("ray", "max_failures", "1"),
        ("model", "booster", "dart"),
    ],
)
def test_recovery_config_rejects_invalid_values(
    section: str, field: str, value: Any
) -> None:
    config: dict[str, Any] = {
        "data": {"label_col": "label"},
        "output": {"bundle_uri": "/bundle"},
        "model": {},
        "training": {"checkpoint_interval_rounds": 10},
        "ray": {"storage_path": "/shared", "max_failures": 1},
    }
    config[section][field] = value
    with pytest.raises(ValueError):
        XGBoostConfigValidator().validate(config)
    config["training"]["checkpoint_interval_rounds"] = 0
    config["ray"]["max_failures"] = 1
    with pytest.raises(ValueError, match="requires checkpoint_interval_rounds"):
        XGBoostConfigValidator().validate(config)


def test_final_callback_replaces_whole_group_and_matches_result(
    monkeypatch: Any,
) -> None:
    import ray

    collector = _EvidenceCollector()
    monkeypatch.setattr(ray, "get", lambda value: value)
    actor = SimpleNamespace(
        replace_final=SimpleNamespace(remote=collector.replace_final)
    )
    callback = evidence_callback(actor, "run", 2)
    checkpoint = SimpleNamespace(path="/final-old")
    callback.after_report(None, _reports("old", 5), checkpoint)
    assert collector.final_snapshot()["metrics"]["model_state_digest"] == "old"
    new_checkpoint = SimpleNamespace(path="/final-new")
    reports = _reports("new", 0)
    callback.after_report(None, reports, new_checkpoint)
    group = collector.final_snapshot()
    assert {worker["model_state_digest"] for worker in group["workers"]} == {"new"}
    check_final_result(
        SimpleNamespace(metrics=reports[0], checkpoint=new_checkpoint), group, "run"
    )
    with pytest.raises(AlgorithmExecutionError, match="does not match"):
        check_final_result(
            SimpleNamespace(metrics=reports[0], checkpoint=checkpoint), group, "run"
        )
    with pytest.raises(AlgorithmExecutionError, match="does not match"):
        check_final_result(
            SimpleNamespace(metrics=_reports("old", 5)[0], checkpoint=new_checkpoint),
            group,
            "run",
        )
    partial = reports[:1]
    with pytest.raises(AlgorithmExecutionError):
        callback.after_report(None, partial, new_checkpoint)
    assert collector.final_snapshot() is group
    periodic = [
        {REPORT_KEY: {"invocation": "run", "phase": "training"}} for _ in range(2)
    ]
    callback.after_report(None, periodic, new_checkpoint)
    assert collector.final_snapshot() is group


@pytest.mark.parametrize("resumed,attempt", [(0, 20), (10, 10), (20, 0)])
def test_output_contract_accepts_actual_attempt_rounds(
    resumed: int, attempt: int
) -> None:
    value = {
        "status": "succeeded",
        "outputs": {"bundle_uri": "/bundle"},
        "metrics": {
            "training_rounds": 20,
            "model_rounds": 20,
            "resumed_from_round": resumed,
            "attempt_training_rounds": attempt,
        },
    }
    assert XGBoostOutputValidator().validate(value) == value
    value["metrics"]["attempt_training_rounds"] = -1
    with pytest.raises(ValueError, match="round counts"):
        XGBoostOutputValidator().validate(value)


def test_trainer_uses_public_failure_and_checkpoint_policies(monkeypatch: Any) -> None:
    import ray
    import ray.train.xgboost as ray_xgboost

    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        ray,
        "remote",
        lambda cls: SimpleNamespace(
            options=lambda **kwargs: SimpleNamespace(remote=lambda: object())
        ),
    )
    monkeypatch.setattr(
        ray_xgboost,
        "XGBoostTrainer",
        lambda **kwargs: captured.append(kwargs) or SimpleNamespace(**kwargs),
    )
    config = {
        "data": {"label_col": "label"},
        "training": {"num_rounds": 20, "checkpoint_interval_rounds": 10},
        "ray": {"storage_path": "/shared", "max_failures": 1},
    }
    plan = SimpleNamespace(
        runtime=SimpleNamespace(
            resume_from=None,
            worker_count=2,
            num_gpus=0,
            num_cpus=1,
            custom_resources={},
        ),
        primary_input_binding=SimpleNamespace(feature_names=("f0",)),
        primary_input_descriptor=SimpleNamespace(binding_digest="binding"),
        plan_id="plan",
        algorithm_config=config,
    )
    adapter = DistributedXGBoost(plan)
    adapter.build_trainer(config, {"train": object()})
    adapter.build_trainer(config, {"train": object()})
    assert captured[0]["run_config"].name != captured[1]["run_config"].name
    for arguments in captured:
        run = arguments["run_config"]
        assert run.failure_config.max_failures == 1
        assert run.failure_config.controller_failure_limit == 0
        assert run.failure_config.max_preemption_failures == 0
        assert run.checkpoint_config.num_to_keep == 1
        assert run.checkpoint_config.checkpoint_score_attribute is None
        assert len(run.callbacks) == 1
        loop = arguments["train_loop_config"]
        assert run.name.endswith(loop["invocation"])
        assert loop["plan_id"] == "plan" and loop["checkpoint_interval_rounds"] == 10
    config["training"]["checkpoint_interval_rounds"] = 0
    config["ray"]["max_failures"] = 0
    adapter.build_trainer(config, {"train": object()})
    assert not captured[-1]["run_config"].callbacks
    assert "checkpoint_interval_rounds" not in captured[-1]["train_loop_config"]


def test_recovery_rejects_train_v1_before_reading_data() -> None:
    script = """
from types import SimpleNamespace
from tributo.algorithms.api import AlgorithmConfigurationError
from tributo_algorithms_boosting.algorithm import DistributedXGBoost
plan = SimpleNamespace(runtime=SimpleNamespace(resume_from=None),
    algorithm_config={'training': {'checkpoint_interval_rounds': 1}})
try:
    DistributedXGBoost(plan).validate_environment()
except AlgorithmConfigurationError as exc:
    assert 'requires Ray Train V2' in str(exc)
else:
    raise AssertionError('Train V1 was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "RAY_TRAIN_V2_ENABLED": "0"},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mismatch", [False, True])
def test_collect_evidence_binds_result_to_the_final_batch(
    monkeypatch: Any, mismatch: bool
) -> None:
    import ray

    checkpoint = SimpleNamespace(path="/persisted/final")
    reports = _reports()
    group = final_report_group(reports, checkpoint, invocation="run", world_size=2)
    adapter = DistributedXGBoost(
        SimpleNamespace(
            runtime=SimpleNamespace(resume_from=None, worker_count=2),
            algorithm_config={"training": {"checkpoint_interval_rounds": 10}},
        )
    )
    actor = SimpleNamespace(final_snapshot=SimpleNamespace(remote=lambda: group))
    adapter.collector, adapter.invocation = actor, "run"
    killed: list[object] = []
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(ray, "kill", lambda actor, **kwargs: killed.append(actor))
    result = SimpleNamespace(
        metrics=reports[0],
        checkpoint=SimpleNamespace(path="/wrong") if mismatch else checkpoint,
    )
    if mismatch:
        with pytest.raises(AlgorithmExecutionError, match="does not match"):
            adapter.collect_evidence(result)
    else:
        assert adapter.collect_evidence(result)["workers"] == group["workers"]
    assert killed == [actor] and adapter.collector is None


@pytest.mark.parametrize(
    "checkpoint_files",
    [False, True],
    indirect=True,
    ids=["default-metrics", "disabled-metrics"],
)
@pytest.mark.parametrize("phase", ["final", "training"])
def test_recovery_export_requires_final_matching_state_and_filters_internal_metrics(
    checkpoint_files: Any, phase: str
) -> None:
    from ray.train import Checkpoint
    from tributo_algorithms_boosting.algorithm import export_result

    root, _ = checkpoint_files
    path = root / "recovery_state.json"
    state = json.loads(path.read_text())
    metrics = {
        "training_rounds": 2,
        "model_rounds": 2,
        "resumed_from_round": 2,
        "attempt_training_rounds": 0,
        "model_state_digest": state["selected_sha256"],
        REPORT_KEY: {"invocation": "fit", "phase": "final", "worker": {}},
    }
    plan = SimpleNamespace(
        plan_id="plan",
        runtime=SimpleNamespace(worker_count=2),
        primary_input_binding=SimpleNamespace(
            feature_names=("f0",), label_name="label"
        ),
        input_bindings=SimpleNamespace(bindings=[SimpleNamespace(name="train")]),
        algorithm_config={
            "training": {"num_rounds": 2, "checkpoint_interval_rounds": 1},
            "output": {"bundle_uri": str(root / "bundle")},
        },
    )
    checkpoint = Checkpoint.from_directory(root)
    state["phase"] = phase
    if phase == "training":
        state["selected_sha256"] = None
    path.write_text(json.dumps(state))
    if phase == "training":
        with pytest.raises(AlgorithmExecutionError, match="diverged"):
            export_result(
                result=SimpleNamespace(metrics=metrics),
                checkpoint=checkpoint,
                plan=plan,
                run_id="export",
            )
        assert not (root / "bundle").exists()
    else:
        result = export_result(
            result=SimpleNamespace(metrics=metrics),
            checkpoint=checkpoint,
            plan=plan,
            run_id="export",
        )
        assert result.status == "succeeded"
        assert REPORT_KEY not in result.metrics
        assert result.metrics["resumed_from_round"] == 2
        assert result.metrics["attempt_training_rounds"] == 0
