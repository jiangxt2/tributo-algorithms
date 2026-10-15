"""Invocation-local XGBoost state and public framework callback integration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from tributo.algorithms.api import AlgorithmExecutionError

REPORT_KEY = "_tributo_xgboost_recovery"


def recovery_identity(config: Mapping[str, Any], world_size: int) -> dict[str, Any]:
    return {
        "invocation": config["invocation"],
        "plan_id": config["plan_id"],
        "world_size": world_size,
        "feature_names": list(config["feature_names"]),
        "label_name": config["label_name"],
        "input_roles": sorted(config["input_roles"]),
        "versions": {
            **{
                name: importlib.metadata.version(name)
                for name in ("tributo", "tributo-algorithms", "ray", "xgboost")
            },
            "python": platform.python_version(),
        },
    }


def write_checkpoint(
    root: Path,
    *,
    full_model: bytes,
    state: Any,
    identity: Mapping[str, Any],
    selected: bytes | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    (root / "training.ubj").write_bytes(full_model)
    (root / "feature_names.json").write_text(
        json.dumps(identity["feature_names"]), encoding="utf-8"
    )
    if selected is not None:
        (root / "model.ubj").write_bytes(selected)
        (root / "training_metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
    payload = {
        "schema_version": 1,
        "identity": dict(identity),
        "phase": "final" if selected is not None else "training",
        "completed_rounds": state.completed,
        "metric": state.metric,
        "scores": state.scores,
        "stopped": state.stopped,
        "patience": state.patience,
        "maximize": state.maximize,
        "best_iteration": state.best_iteration,
        "best_score": state.best_score,
        "last_metrics": state.last_metrics,
        "training_sha256": hashlib.sha256(full_model).hexdigest(),
        "selected_sha256": hashlib.sha256(selected).hexdigest()
        if selected is not None
        else None,
    }
    (root / "recovery_state.json").write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def read_checkpoint(
    root: Path,
    *,
    identity: Mapping[str, Any],
    rounds: int,
    patience: int | None,
    maximize: bool | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Fail closed on a corrupt, foreign, or internally inconsistent checkpoint."""
    import xgboost

    try:
        state = json.loads((root / "recovery_state.json").read_text(encoding="utf-8"))
        fields = {
            "schema_version",
            "identity",
            "phase",
            "completed_rounds",
            "metric",
            "scores",
            "stopped",
            "last_metrics",
            "training_sha256",
            "selected_sha256",
            "patience",
            "maximize",
            "best_iteration",
            "best_score",
        }
        if (
            not isinstance(state, dict)
            or set(state) != fields
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or state["identity"] != dict(identity)
            or state["phase"] not in {"training", "final"}
            or type(state["completed_rounds"]) is not int
            or not 1 <= state["completed_rounds"] <= rounds
            or type(state["stopped"]) is not bool
        ):
            raise ValueError("identity, version, phase or rounds mismatch")
        scores = state["scores"]
        if not isinstance(scores, list) or any(
            not isinstance(score, (float, int))
            or isinstance(score, bool)
            or not math.isfinite(score)
            for score in scores
        ):
            raise ValueError("selection scores are invalid")
        if patience is not None:
            if (
                len(scores) != state["completed_rounds"]
                or not isinstance(state["metric"], str)
                or not state["metric"]
            ):
                raise ValueError("selection history is incomplete")
            best, score = state["best_iteration"], state["best_score"]
            if (
                type(state["patience"]) is not int
                or state["patience"] != patience
                or type(state["maximize"]) is not bool
                or state["maximize"] != maximize
                or type(best) is not int
                or not 0 <= best < state["completed_rounds"]
                or not isinstance(score, (float, int))
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                raise ValueError("selection configuration or best state mismatch")
        elif scores or state["metric"] is not None or state["stopped"]:
            raise ValueError("unexpected early-stopping state")
        elif any(
            state[key] is not None
            for key in ("patience", "maximize", "best_iteration", "best_score")
        ):
            raise ValueError("unexpected selection configuration")
        last = state["last_metrics"]
        if (
            not isinstance(last, dict)
            or not last
            or any(
                not isinstance(key, str)
                or not key.startswith(("train-", "validation-"))
                or not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for key, value in last.items()
            )
        ):
            raise ValueError("last metrics are invalid")
        if (
            patience is not None
            and last.get(f"validation-{state['metric']}") != scores[-1]
        ):
            raise ValueError("last selection score mismatch")
        features = json.loads((root / "feature_names.json").read_text(encoding="utf-8"))
        raw = (root / "training.ubj").read_bytes()
        model = xgboost.Booster(model_file=bytearray(raw))
        if (
            features != identity["feature_names"]
            or model.num_features() != len(features)
            or model.num_boosted_rounds() != state["completed_rounds"]
            or hashlib.sha256(raw).hexdigest() != state["training_sha256"]
        ):
            raise ValueError("training model or features mismatch")
        if patience is not None and (
            model.best_iteration != state["best_iteration"]
            or model.best_score != state["best_score"]
        ):
            raise ValueError("training model best attrs mismatch")
        if state["phase"] == "final":
            from tributo_algorithms_boosting.contracts import (
                _validate_training_metadata,
            )
            from tributo_algorithms_boosting.native import _check_training_metadata

            selected_raw = (root / "model.ubj").read_bytes()
            selected = xgboost.Booster(model_file=bytearray(selected_raw))
            metadata = _validate_training_metadata(
                json.loads(
                    (root / "training_metadata.json").read_text(encoding="utf-8")
                )
            )
            _check_training_metadata(selected, metadata)
            if (
                hashlib.sha256(selected_raw).hexdigest() != state["selected_sha256"]
                or metadata["training_rounds"] != state["completed_rounds"]
                or (not state["stopped"] and state["completed_rounds"] != rounds)
                or (
                    patience is not None
                    and any(
                        metadata.get(key) != state[key]
                        for key in ("best_iteration", "best_score", "maximize")
                    )
                )
            ):
                raise ValueError("final model mismatch")
        elif state["selected_sha256"] is not None:
            raise ValueError("periodic checkpoint contains a selected model digest")
        return raw, state
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        xgboost.core.XGBoostError,
    ) as exc:
        raise AlgorithmExecutionError("XGBoost recovery checkpoint is invalid") from exc


def training_callback(
    *,
    dtrain: Any,
    patience: int | None,
    maximize: bool | None,
    interval: int,
    restored: Mapping[str, Any] | None = None,
    report: Callable[[Any, Any], None] | None = None,
) -> Any:
    """Compose native early stopping with checkpointing in a defined order.

    Rebuild native patience by replaying scalar history through public hooks.
    The auxiliary zero-round Booster keeps historical epochs absolute; calling
    before_training on the real Booster then sets the continuation offset.
    """
    import xgboost

    class TrainingState(xgboost.callback.TrainingCallback):
        def __init__(self) -> None:
            self.matrix: Any = None
            self.patience, self.maximize = patience, maximize
            self.best_iteration: int | None = None
            self.best_score: float | None = None
            self.completed = int(restored["completed_rounds"]) if restored else 0
            self.start = self.completed
            self.scores = list(restored["scores"]) if restored else []
            self.metric = restored["metric"] if restored else None
            self.last_metrics = dict(restored["last_metrics"]) if restored else {}
            self.stopped = False
            self.full_model = b""
            self.early = (
                xgboost.callback.EarlyStopping(
                    rounds=patience,
                    data_name="validation",
                    maximize=maximize,
                    save_best=True,
                )
                if patience is not None
                else None
            )

        def before_training(self, model: Any) -> Any:
            if model.num_boosted_rounds() != self.start:
                raise AlgorithmExecutionError("XGBoost checkpoint round mismatch")
            if self.early is not None:
                if self.scores:
                    if not isinstance(self.metric, str) or not self.metric:
                        raise AlgorithmExecutionError(
                            "XGBoost selection metric is missing"
                        )
                    auxiliary = xgboost.Booster(cache=[self.matrix])
                    self.early.before_training(auxiliary)
                    for epoch, score in enumerate(self.scores):
                        if self.stopped:
                            raise AlgorithmExecutionError(
                                "XGBoost history continues after early stopping"
                            )
                        self.stopped = self.early.after_iteration(
                            auxiliary,
                            epoch,
                            {"validation": {cast(str, self.metric): [score]}},
                        )
                    if (
                        auxiliary.attr("best_iteration") != model.attr("best_iteration")
                        or auxiliary.attr("best_score") != model.attr("best_score")
                        or restored is None
                        or self.stopped != restored["stopped"]
                    ):
                        raise AlgorithmExecutionError(
                            "XGBoost checkpoint early-stopping state mismatch"
                        )
                self.early.before_training(model)
                if self.scores:
                    self.best_iteration, self.best_score = (
                        int(model.best_iteration),
                        float(model.best_score),
                    )
            return model

        def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
            self.completed = self.start + epoch + 1
            for role, values in evals_log.items():
                for metric, scores in values.items():
                    score = float(scores[-1])
                    if not math.isfinite(score):
                        raise AlgorithmExecutionError(
                            "XGBoost evaluation scores must be finite"
                        )
                    self.last_metrics[f"{role}-{metric}"] = score
            if self.early is not None:
                validation = evals_log.get("validation", {})
                if len(validation) != 1:
                    raise AlgorithmExecutionError(
                        "XGBoost selection metric is ambiguous"
                    )
                metric, scores = next(iter(validation.items()))
                if self.metric is not None and self.metric != metric:
                    raise AlgorithmExecutionError("XGBoost selection metric changed")
                self.metric = metric
                self.stopped = self.early.after_iteration(model, epoch, evals_log)
                self.scores.append(float(scores[-1]))
                self.best_iteration, self.best_score = (
                    int(model.best_iteration),
                    float(model.best_score),
                )
            if interval and self.completed % interval == 0 and report is not None:
                report(model, self)
            return self.stopped

        def after_training(self, model: Any) -> Any:
            self.full_model = bytes(model.save_raw(raw_format="ubj"))
            self.matrix = None
            return self.early.after_training(model) if self.early is not None else model

    callback = TrainingState()
    callback.matrix = dtrain
    return callback


def final_report_group(
    metrics: list[dict[str, Any]],
    checkpoint: Any,
    *,
    invocation: str,
    world_size: int,
) -> dict[str, Any] | None:
    """Accept one complete final report, never a mixture of rank records."""
    if len(metrics) != world_size:
        raise AlgorithmExecutionError("XGBoost report is missing Workers")
    raw_reports = [item.get(REPORT_KEY) for item in metrics]
    if any(
        not isinstance(item, Mapping) or item.get("invocation") != invocation
        for item in raw_reports
    ):
        raise AlgorithmExecutionError("XGBoost report invocation mismatch")
    reports = [cast(Mapping[str, Any], item) for item in raw_reports]
    phases = {item.get("phase") for item in reports}
    if phases == {"training"}:
        return None
    if phases != {"final"} or checkpoint is None:
        raise AlgorithmExecutionError("XGBoost final report requires a Checkpoint")
    public = [
        {key: value for key, value in item.items() if key != REPORT_KEY}
        for item in metrics
    ]
    if any(item != public[0] for item in public[1:]):
        raise AlgorithmExecutionError("XGBoost final Worker metrics diverged")
    completed, start, attempt = (
        public[0].get(key)
        for key in ("training_rounds", "resumed_from_round", "attempt_training_rounds")
    )
    if (
        type(completed) is not int
        or completed < 1
        or type(start) is not int
        or start < 0
        or type(attempt) is not int
        or attempt < 0
        or start + attempt != completed
    ):
        raise AlgorithmExecutionError("XGBoost final report round counts are invalid")
    workers = [item.get("worker") for item in reports]
    if any(
        not isinstance(worker, dict)
        or type(worker.get("rank")) is not int
        or worker["rank"] != rank
        or worker.get("world_size") != world_size
        or worker.get("model_state_digest") != public[0].get("model_state_digest")
        or worker.get("collective_steps") != public[0].get("attempt_training_rounds")
        or worker.get("batch_count") != public[0].get("attempt_training_rounds")
        for rank, worker in enumerate(workers)
    ):
        raise AlgorithmExecutionError("XGBoost final Worker evidence is inconsistent")
    return {
        "checkpoint_path": checkpoint.path,
        "metrics": public[0],
        "workers": workers,
    }


def check_final_result(result: Any, group: Mapping[str, Any], invocation: str) -> None:
    metrics = getattr(result, "metrics", {})
    checkpoint = getattr(result, "checkpoint", None)
    if not isinstance(metrics, Mapping):
        raise AlgorithmExecutionError("XGBoost result metrics must be a mapping")
    internal = metrics.get(REPORT_KEY, {})
    if (
        checkpoint is None
        or checkpoint.path != group.get("checkpoint_path")
        or not isinstance(internal, Mapping)
        or internal.get("invocation") != invocation
        or internal.get("phase") != "final"
        or any(
            metrics.get(key) != value for key, value in group.get("metrics", {}).items()
        )
    ):
        raise AlgorithmExecutionError(
            "XGBoost result does not match final Worker evidence"
        )


def evidence_callback(collector: Any, invocation: str, world_size: int) -> Any:
    """Commit final evidence only after Ray persists the complete group report."""
    import ray
    from ray.train import UserCallback

    class FinalEvidence(UserCallback):
        def after_report(
            self, run_context: Any, metrics: list[dict[str, Any]], checkpoint: Any
        ) -> None:
            group = final_report_group(
                metrics, checkpoint, invocation=invocation, world_size=world_size
            )
            if group is not None:
                ray.get(collector.replace_final.remote(group))

    return FinalEvidence()
