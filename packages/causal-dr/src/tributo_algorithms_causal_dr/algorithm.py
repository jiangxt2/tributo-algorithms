"""Three-stage distributed AIPW/Doubly Robust ATE estimator."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ResolvedAlgorithmPlan,
    WorkerResources,
)
from tributo.algorithms.spi import FrameworkNativeAlgorithm
from tributo_algorithms_boosting import XGBoostStageRunner
from tributo_algorithms_causal_core.algorithm import CausalATEModel, export_model

STAGES = ("mu0", "mu1", "propensity")
_LABEL = "__tributo_dr_label"


def _split_rows(materialized: object, fold_count: int) -> tuple[object, ...]:
    """Split a materialized Dataset by row indices without dropping rows."""
    rows = int(cast(Any, materialized).count())
    base, remainder = divmod(rows, fold_count)
    boundaries: list[int] = []
    cursor = 0
    for index in range(fold_count - 1):
        cursor += base + (1 if index < remainder else 0)
        boundaries.append(cursor)
    return tuple(cast(Any, materialized).split_at_indices(boundaries))


def _lossless_folds(
    dataset: object,
    treatment_name: str,
    fold_count: int,
    *,
    seed: int,
) -> tuple[object, ...]:
    """Create stratified disjoint folds while retaining every row."""
    treated = (
        cast(Any, dataset)
        .filter(lambda row: int(row[treatment_name]) == 1)
        .random_shuffle(seed=seed)
        .materialize()
    )
    control = (
        cast(Any, dataset)
        .filter(lambda row: int(row[treatment_name]) == 0)
        .random_shuffle(seed=seed + 1)
        .materialize()
    )
    if int(treated.count()) < fold_count or int(control.count()) < fold_count:
        raise AlgorithmExecutionError(
            "DR cross-fitting requires treated and control rows in every fold"
        )
    treated_folds = _split_rows(treated, fold_count)
    control_folds = _split_rows(control, fold_count)
    return tuple(
        cast(Any, treated_folds[index]).union(control_folds[index])
        for index in range(fold_count)
    )


def _label_batch(
    batch: object,
    *,
    feature_names: tuple[str, ...],
    label_name: str,
) -> object:
    frame = cast(Any, batch)
    result = frame.loc[:, list(feature_names)].copy()
    result[_LABEL] = frame[label_name].to_numpy()
    return result


def _score_batch(
    batch: object,
    *,
    feature_names: tuple[str, ...],
    treatment_name: str,
    outcome_name: str,
    booster_raw: Mapping[str, bytes],
    propensity_clip: tuple[float, float],
) -> dict[str, object]:
    import numpy as np
    import xgboost

    frame = cast(Any, batch)
    matrix = xgboost.DMatrix(frame.loc[:, list(feature_names)].to_numpy())

    def predict(stage: str) -> Any:
        booster = xgboost.Booster()
        booster.load_model(bytearray(booster_raw[stage]))
        return booster.predict(matrix)

    mu0 = predict("mu0")
    mu1 = predict("mu1")
    propensity = np.clip(predict("propensity"), propensity_clip[0], propensity_clip[1])
    treatment = frame[treatment_name].to_numpy(dtype=np.float64)
    outcome = frame[outcome_name].to_numpy(dtype=np.float64)
    score = (
        mu1
        - mu0
        + treatment * (outcome - mu1) / propensity
        - (1.0 - treatment) * (outcome - mu0) / (1.0 - propensity)
    )
    return {"dr_score": score}


@dataclass(frozen=True)
class DRResult:
    model: CausalATEModel
    stage_evidence: Mapping[str, Mapping[str, object]]
    composition_digest: str


class _DRDriver:
    def __init__(self, *, plan: ResolvedAlgorithmPlan, dataset: object) -> None:
        self.plan = plan
        self.dataset = dataset

    def fit(self) -> DRResult:
        config = self.plan.algorithm_config
        data = cast(Mapping[str, Any], config["data"])
        model_config = cast(Mapping[str, Any], config["model"])
        training = cast(Mapping[str, Any], config["training"])
        ray_config = cast(Mapping[str, Any], config["ray"])
        features = tuple(str(name) for name in data["feature_columns"])
        treatment_name = str(data["treatment_col"])
        outcome_name = str(data["outcome_col"])
        dataset = cast(Any, self.dataset)
        treated = dataset.filter(lambda row: int(row[treatment_name]) == 1)
        control = dataset.filter(lambda row: int(row[treatment_name]) == 0)
        treated_rows = int(treated.count())
        control_rows = int(control.count())
        rows = treated_rows + control_rows
        fold_count = int(training.get("cross_fit_folds", 5))
        if fold_count < 2 or fold_count > 20:
            raise AlgorithmConfigurationError(
                "training.cross_fit_folds must be between 2 and 20"
            )
        if treated_rows < max(
            self.plan.runtime.worker_count, fold_count
        ) or control_rows < max(self.plan.runtime.worker_count, fold_count):
            raise AlgorithmExecutionError(
                "DR treatment groups are too small for worker and fold coverage"
            )
        if rows < fold_count * 2:
            raise AlgorithmExecutionError(
                "DR cross-fitting requires at least one treated and control row per fold"
            )
        folds = _lossless_folds(
            dataset,
            treatment_name,
            fold_count,
            seed=int(training.get("seed", 7)),
        )
        if len(folds) != fold_count:
            raise AlgorithmExecutionError(
                "DR cross-fitting produced an invalid fold set"
            )
        runner = XGBoostStageRunner(
            worker_count=self.plan.runtime.worker_count,
            resources_per_worker=WorkerResources(
                num_cpus=self.plan.runtime.num_cpus,
                num_gpus=self.plan.runtime.num_gpus,
                custom=self.plan.runtime.custom_resources,
            ),
            storage_path=str(ray_config["storage_path"]),
            input_binding_digest=self.plan.primary_input_descriptor.binding_digest,
        )

        def labelled(source: object, label: str) -> object:
            return cast(Any, source).map_batches(
                _label_batch,
                batch_format="pandas",
                fn_kwargs={"feature_names": features, "label_name": label},
            )

        outcome_params = dict(
            cast(Mapping[str, object], model_config.get("outcome", {}))
        )
        outcome_params.setdefault("objective", "reg:squarederror")
        propensity_params = dict(
            cast(Mapping[str, object], model_config.get("propensity", {}))
        )
        propensity_params.setdefault("objective", "binary:logistic")
        rounds = int(training.get("num_rounds", 10))
        clip_value = training.get("propensity_clip", (0.01, 0.99))
        propensity_clip = (float(clip_value[0]), float(clip_value[1]))
        scored_folds: list[object] = []
        stage_folds: dict[str, list[dict[str, object]]] = {name: [] for name in STAGES}
        stage_digests: dict[str, list[str]] = {name: [] for name in STAGES}
        for fold_index, heldout in enumerate(folds):
            training_parts = folds[:fold_index] + folds[fold_index + 1 :]
            train_data = cast(
                Any,
                (
                    training_parts[0]
                    if len(training_parts) == 1
                    else cast(Any, training_parts[0]).union(*training_parts[1:])
                ),
            )
            heldout_data = cast(Any, heldout)
            train_treated = train_data.filter(lambda row: int(row[treatment_name]) == 1)
            train_control = train_data.filter(lambda row: int(row[treatment_name]) == 0)
            train_treated_rows = int(train_treated.count())
            train_control_rows = int(train_control.count())
            heldout_treated_rows = int(
                heldout_data.filter(lambda row: int(row[treatment_name]) == 1).count()
            )
            heldout_control_rows = int(
                heldout_data.filter(lambda row: int(row[treatment_name]) == 0).count()
            )
            if (
                train_treated_rows < self.plan.runtime.worker_count
                or train_control_rows < self.plan.runtime.worker_count
                or heldout_treated_rows < 1
                or heldout_control_rows < 1
            ):
                raise AlgorithmExecutionError(
                    "DR cross-fitting fold lacks treatment/control overlap"
                )
            stages = {
                "mu0": runner.fit(
                    f"fold-{fold_index}-mu0",
                    labelled(train_control, outcome_name),
                    feature_names=features,
                    label_name=_LABEL,
                    params=outcome_params,
                    num_boost_round=rounds,
                ),
                "mu1": runner.fit(
                    f"fold-{fold_index}-mu1",
                    labelled(train_treated, outcome_name),
                    feature_names=features,
                    label_name=_LABEL,
                    params=outcome_params,
                    num_boost_round=rounds,
                ),
                "propensity": runner.fit(
                    f"fold-{fold_index}-propensity",
                    labelled(train_data, treatment_name),
                    feature_names=features,
                    label_name=_LABEL,
                    params=propensity_params,
                    num_boost_round=rounds,
                ),
            }
            raw = {name: stages[name].booster_raw for name in STAGES}
            scored_folds.append(
                heldout_data.map_batches(
                    _score_batch,
                    batch_format="pandas",
                    fn_kwargs={
                        "feature_names": features,
                        "treatment_name": treatment_name,
                        "outcome_name": outcome_name,
                        "booster_raw": raw,
                        "propensity_clip": propensity_clip,
                    },
                )
            )
            for name in STAGES:
                stage = stages[name]
                evidence = dict(cast(Mapping[str, object], stage.evidence))
                evidence["heldout_rows"] = int(
                    heldout_treated_rows + heldout_control_rows
                )
                evidence["fold_index"] = fold_index
                stage_folds[name].append(evidence)
                stage_digests[name].append(
                    str(
                        cast(Mapping[str, Any], stage.evidence["state"])[
                            "global_model_digest"
                        ]
                    )
                )
        scores = cast(Any, scored_folds[0]).union(*scored_folds[1:])
        effect = float(scores.mean("dr_score"))
        standard_error = float(scores.std("dr_score")) / math.sqrt(rows)
        z = float(training.get("confidence_z", 1.959963984540054))
        policy_cost = float(training.get("policy_cost", 0.0))
        treated_mean = float(treated.mean(outcome_name))
        control_mean = float(control.mean(outcome_name))
        stage_evidence = {
            name: {
                "workers": stage_folds[name][0]["workers"],
                "state": stage_folds[name][0]["state"],
                "input_complete": True,
                "expected_training_rows": rows,
                "cross_fit_folds": stage_folds[name],
            }
            for name in STAGES
        }
        composition_digest = hashlib.sha256(
            json.dumps(stage_digests, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        causal_model = CausalATEModel(
            method="distributed_aipw_dr",
            treatment=treatment_name,
            outcome=outcome_name,
            feature_names=features,
            treated_count=treated_rows,
            control_count=control_rows,
            treated_mean=treated_mean,
            control_mean=control_mean,
            effect=effect,
            standard_error=standard_error,
            confidence_interval=(
                effect - z * standard_error,
                effect + z * standard_error,
            ),
            policy_cost=policy_cost,
            treat_all_policy=effect > policy_cost,
            diagnostics={
                "propensity_clip_low": propensity_clip[0],
                "propensity_clip_high": propensity_clip[1],
                "cross_fit_folds": fold_count,
            },
        )
        return DRResult(
            model=causal_model,
            stage_evidence=stage_evidence,
            composition_digest=composition_digest,
        )


class DistributedDRLearner(FrameworkNativeAlgorithm):
    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan

    def validate_environment(self) -> None:
        try:
            import xgboost
        except ImportError as exc:
            raise AlgorithmConfigurationError("DR estimator requires XGBoost") from exc
        if not xgboost.__version__:
            raise AlgorithmConfigurationError("DR environment is invalid")

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        if len(datasets) != 1:
            raise AlgorithmConfigurationError("DR estimator requires one Dataset")
        dataset = next(iter(datasets.values()))
        data = cast(Mapping[str, Any], self.plan.algorithm_config["data"])
        required = (
            *tuple(data["feature_columns"]),
            str(data["treatment_col"]),
            str(data["outcome_col"]),
        )
        binding = self.plan.primary_input_binding
        if set(binding.feature_names) != set(required) - {str(data["outcome_col"])}:
            raise AlgorithmConfigurationError("DR InputBinding columns drifted")
        if binding.label_name != data["outcome_col"]:
            raise AlgorithmConfigurationError("DR outcome label drifted")
        return {"train": cast(Any, dataset).select_columns(list(required))}

    def build_trainer(
        self, config: Mapping[str, Any], datasets: Mapping[str, object]
    ) -> object:
        del config
        return _DRDriver(plan=self.plan, dataset=datasets["train"])

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        if not isinstance(result, DRResult):
            raise AlgorithmExecutionError("DR returned an invalid result")
        return {
            "stages": dict(result.stage_evidence),
            "composition_digest": result.composition_digest,
        }

    def checkpoint_source(self, result: object) -> object:
        if not isinstance(result, DRResult):
            raise AlgorithmExecutionError("DR checkpoint result is invalid")
        return result


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedDRLearner:
    del artifacts
    if implementation is not DistributedDRLearner:
        raise AlgorithmConfigurationError("DR implementation drifted")
    return DistributedDRLearner(plan)


def export_result(
    *, result: object, checkpoint: object, plan: ResolvedAlgorithmPlan, run_id: str
) -> AlgorithmExecutionResult:
    del checkpoint
    if not isinstance(result, DRResult):
        raise AlgorithmExecutionError("DR export result is invalid")
    execution = export_model(model=result.model, plan=plan, run_id=run_id)
    return AlgorithmExecutionResult(
        status=execution.status,
        metrics=execution.metrics,
        outputs={
            **dict(execution.outputs),
            "composition_digest": result.composition_digest,
        },
        artifacts=execution.artifacts,
    )


__all__ = [
    "DRResult",
    "DistributedDRLearner",
    "STAGES",
    "create_algorithm",
    "export_result",
]
