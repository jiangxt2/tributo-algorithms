"""Distributed shard-ensemble DoWhy estimate and placebo refutation."""

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
)
from tributo.algorithms.spi import FrameworkNativeAlgorithm
from tributo_algorithms_causal_core.algorithm import CausalATEModel, export_model

STAGES = ("estimate", "refute")


def _run_shard(
    shard: object,
    *,
    treatment_name: str,
    outcome_name: str,
    common_causes: tuple[str, ...],
    seed: int,
    rank: int,
    world_size: int,
    binding_digest: str,
) -> dict[str, object]:
    import ray
    from dowhy import CausalModel

    frame = cast(Any, shard).materialize().to_pandas()
    if frame.empty:
        raise AlgorithmExecutionError("DoWhy shard is empty")
    model = CausalModel(
        data=frame,
        treatment=treatment_name,
        outcome=outcome_name,
        common_causes=list(common_causes),
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
    )
    refutation = model.refute_estimate(
        estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        random_seed=seed + rank,
    )
    runtime = ray.get_runtime_context()
    assigned = runtime.get_assigned_resources()
    treatment = frame[treatment_name]
    treated = treatment == 1
    control = treatment == 0
    return {
        "rank": rank,
        "world_size": world_size,
        "worker_id": str(runtime.get_worker_id()),
        "node_id": str(runtime.get_node_id()),
        "shard_id": hashlib.sha256(
            f"{binding_digest}:{rank}/{world_size}".encode("ascii")
        ).hexdigest(),
        "rows": int(frame.shape[0]),
        "effect": float(estimate.value),
        "placebo_effect": float(refutation.new_effect),
        "treated_count": int(treated.sum()),
        "control_count": int(control.sum()),
        "treated_sum": float(frame.loc[treated, outcome_name].sum()),
        "control_sum": float(frame.loc[control, outcome_name].sum()),
        "resources": {
            "num_cpus": float(assigned.get("CPU", 0.0)),
            "num_gpus": float(assigned.get("GPU", 0.0)),
            "custom": {
                str(name): float(value)
                for name, value in assigned.items()
                if name not in {"CPU", "GPU", "memory", "object_store_memory"}
            },
        },
    }


@dataclass(frozen=True)
class DoWhyResult:
    model: CausalATEModel
    stage_evidence: Mapping[str, Mapping[str, object]]
    composition_digest: str


class _DoWhyDriver:
    def __init__(self, *, plan: ResolvedAlgorithmPlan, dataset: object) -> None:
        self.plan = plan
        self.dataset = dataset

    def fit(self) -> DoWhyResult:
        import numpy as np
        import ray

        config = self.plan.algorithm_config
        data = cast(Mapping[str, Any], config["data"])
        refutation = cast(Mapping[str, Any], config["refutation"])
        treatment_name = str(data["treatment_col"])
        outcome_name = str(data["outcome_col"])
        common_causes = tuple(str(name) for name in data["common_causes"])
        world_size = self.plan.runtime.worker_count
        shards = cast(Any, self.dataset).split(world_size, equal=True)
        remote: Any = ray.remote(cast(Any, _run_shard)).options(
            num_cpus=self.plan.runtime.num_cpus,
            num_gpus=self.plan.runtime.num_gpus,
            resources=dict(self.plan.runtime.custom_resources),
        )
        results = cast(
            list[dict[str, Any]],
            ray.get(
                [
                    remote.remote(
                        shard,
                        treatment_name=treatment_name,
                        outcome_name=outcome_name,
                        common_causes=common_causes,
                        seed=int(refutation.get("seed", 7)),
                        rank=rank,
                        world_size=world_size,
                        binding_digest=self.plan.primary_input_descriptor.binding_digest,
                    )
                    for rank, shard in enumerate(shards)
                ]
            ),
        )
        rows = sum(int(item["rows"]) for item in results)
        effects = np.asarray([float(item["effect"]) for item in results])
        placebo = np.asarray([float(item["placebo_effect"]) for item in results])
        weights = np.asarray([int(item["rows"]) for item in results], dtype=np.float64)
        effect = float(np.average(effects, weights=weights))
        standard_error = (
            float(np.std(effects, ddof=1) / math.sqrt(world_size))
            if world_size > 1
            else 0.0
        )
        treated_count = sum(int(item["treated_count"]) for item in results)
        control_count = sum(int(item["control_count"]) for item in results)
        if treated_count < 2 or control_count < 2:
            raise AlgorithmExecutionError("DoWhy ensemble requires treatment overlap")
        policy_cost = float(refutation.get("policy_cost", 0.0))
        causal_model = CausalATEModel(
            method="distributed_dowhy_linear_ensemble",
            treatment=treatment_name,
            outcome=outcome_name,
            feature_names=common_causes,
            treated_count=treated_count,
            control_count=control_count,
            treated_mean=sum(float(item["treated_sum"]) for item in results)
            / treated_count,
            control_mean=sum(float(item["control_sum"]) for item in results)
            / control_count,
            effect=effect,
            standard_error=standard_error,
            confidence_interval=(
                effect - 1.959963984540054 * standard_error,
                effect + 1.959963984540054 * standard_error,
            ),
            policy_cost=policy_cost,
            treat_all_policy=effect > policy_cost,
            diagnostics={
                "placebo_effect_mean": float(np.average(placebo, weights=weights)),
                "shard_estimate_std": float(np.std(effects)),
                "shard_count": float(world_size),
            },
        )
        estimate_digest = hashlib.sha256(
            json.dumps({"effect": effect, "rows": rows}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        refute_digest = hashlib.sha256(
            json.dumps(
                {"placebo": placebo.tolist(), "rows": rows}, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()

        def workers(digest: str) -> list[dict[str, object]]:
            return [
                {
                    "worker_id": item["worker_id"],
                    "node_id": item["node_id"],
                    "rank": item["rank"],
                    "world_size": item["world_size"],
                    "shard_id": item["shard_id"],
                    "resources": item["resources"],
                    "model_state_digest": digest,
                    "rows_processed": item["rows"],
                    "input_rows": {"train": item["rows"]},
                    "batch_count": 1,
                    "collective_steps": 1,
                }
                for item in results
            ]

        stages = {
            "estimate": {
                "workers": workers(estimate_digest),
                "state": {
                    "coordination": "framework_native",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": estimate_digest,
                    "details": {"framework": "dowhy", "stage": "estimate"},
                },
                "input_complete": True,
                "expected_training_rows": rows,
            },
            "refute": {
                "workers": workers(refute_digest),
                "state": {
                    "coordination": "framework_native",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": refute_digest,
                    "details": {"framework": "dowhy", "stage": "refute"},
                },
                "input_complete": True,
                "expected_training_rows": rows,
            },
        }
        composition_digest = hashlib.sha256(
            f"{estimate_digest}:{refute_digest}".encode("ascii")
        ).hexdigest()
        return DoWhyResult(
            model=causal_model,
            stage_evidence=stages,
            composition_digest=composition_digest,
        )


class DistributedDoWhyRefutation(FrameworkNativeAlgorithm):
    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan

    def validate_environment(self) -> None:
        try:
            import dowhy
        except ImportError as exc:
            raise AlgorithmConfigurationError("DoWhy dependency is required") from exc
        if not dowhy.__version__:
            raise AlgorithmConfigurationError("DoWhy environment is invalid")

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        if len(datasets) != 1:
            raise AlgorithmConfigurationError("DoWhy requires one Dataset")
        data = cast(Mapping[str, Any], self.plan.algorithm_config["data"])
        required = (
            *tuple(data["common_causes"]),
            str(data["treatment_col"]),
            str(data["outcome_col"]),
        )
        binding = self.plan.primary_input_binding
        if set(binding.feature_names) != set(required) - {str(data["outcome_col"])}:
            raise AlgorithmConfigurationError("DoWhy InputBinding columns drifted")
        if binding.label_name != data["outcome_col"]:
            raise AlgorithmConfigurationError("DoWhy outcome label drifted")
        dataset = next(iter(datasets.values()))
        return {"train": cast(Any, dataset).select_columns(list(required))}

    def build_trainer(
        self, config: Mapping[str, Any], datasets: Mapping[str, object]
    ) -> object:
        del config
        return _DoWhyDriver(plan=self.plan, dataset=datasets["train"])

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        if not isinstance(result, DoWhyResult):
            raise AlgorithmExecutionError("DoWhy returned an invalid result")
        return {
            "stages": dict(result.stage_evidence),
            "composition_digest": result.composition_digest,
        }

    def checkpoint_source(self, result: object) -> object:
        if not isinstance(result, DoWhyResult):
            raise AlgorithmExecutionError("DoWhy result is invalid")
        return result


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedDoWhyRefutation:
    del artifacts
    if implementation is not DistributedDoWhyRefutation:
        raise AlgorithmConfigurationError("DoWhy implementation drifted")
    return DistributedDoWhyRefutation(plan)


def export_result(
    *, result: object, checkpoint: object, plan: ResolvedAlgorithmPlan, run_id: str
) -> AlgorithmExecutionResult:
    del checkpoint
    if not isinstance(result, DoWhyResult):
        raise AlgorithmExecutionError("DoWhy export result is invalid")
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
    "DoWhyResult",
    "DistributedDoWhyRefutation",
    "STAGES",
    "create_algorithm",
    "export_result",
]
