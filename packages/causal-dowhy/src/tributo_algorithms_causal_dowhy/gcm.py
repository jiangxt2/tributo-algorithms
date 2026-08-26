"""Distributed DoWhy GCM root-cause attribution and counterfactual analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import FrameworkNativeAlgorithm
from tributo.exporting.models import BundleOutputConfig, ExportSource, ExportTarget
from tributo.exporting.service import BundleExportService

STAGES = ("fit_gcm", "attribute_root_cause")
_PACKAGE = "tributo-algorithms-causal-dowhy"


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_gcm_shard(
    train_shard: object,
    anomaly_shard: object,
    *,
    nodes: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    target_node: str,
    interventions: Mapping[str, float],
    quality: str,
    distribution_samples: int,
    shapley_permutations: int,
    rank: int,
    world_size: int,
    binding_digest: str,
) -> dict[str, object]:
    import networkx as nx
    import numpy as np
    import ray
    from dowhy import gcm
    from dowhy.gcm.shapley import ShapleyConfig

    train = cast(Any, train_shard).materialize().to_pandas()
    anomaly = cast(Any, anomaly_shard).materialize().to_pandas()
    if train.empty or anomaly.empty:
        raise AlgorithmExecutionError("GCM train and anomaly shards must be non-empty")
    train = train.loc[:, list(nodes)]
    anomaly = anomaly.loc[:, list(nodes)]
    if (
        not np.isfinite(train.to_numpy(dtype=np.float64)).all()
        or not np.isfinite(anomaly.to_numpy(dtype=np.float64)).all()
    ):
        raise AlgorithmExecutionError("GCM requires finite numeric values")

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise AlgorithmConfigurationError("GCM causal graph must be acyclic")
    if set(graph.nodes) != set(nodes):
        raise AlgorithmConfigurationError("GCM graph contains undeclared nodes")

    gcm.config.set_default_n_jobs(1)
    causal_model = gcm.InvertibleStructuralCausalModel(graph)
    quality_value = {
        "good": gcm.auto.AssignmentQuality.GOOD,
        "better": gcm.auto.AssignmentQuality.BETTER,
        "best": gcm.auto.AssignmentQuality.BEST,
    }[quality]
    gcm.auto.assign_causal_mechanisms(
        causal_model,
        train,
        quality=quality_value,
    )
    gcm.fit(causal_model, train)
    attributions = gcm.attribute_anomalies(
        causal_model,
        target_node,
        anomaly,
        num_distribution_samples=distribution_samples,
        shapley_config=ShapleyConfig(
            num_permutations=shapley_permutations,
            n_jobs=1,
        ),
    )
    attribution_summary = {
        str(node): {
            "absolute_sum": float(np.abs(np.asarray(values)).sum()),
            "signed_sum": float(np.asarray(values).sum()),
        }
        for node, values in attributions.items()
    }
    counterfactual_delta_sum = 0.0
    counterfactual_absolute_delta_sum = 0.0
    if interventions:
        intervention_functions = {
            node: (lambda values, fixed=value: np.full_like(values, fixed))
            for node, value in interventions.items()
        }
        counterfactual = gcm.counterfactual_samples(
            causal_model,
            intervention_functions,
            observed_data=anomaly,
        )
        counterfactual_deltas = counterfactual[target_node].to_numpy(
            dtype=np.float64
        ) - anomaly[target_node].to_numpy(dtype=np.float64)
        counterfactual_delta_sum = float(counterfactual_deltas.sum())
        counterfactual_absolute_delta_sum = float(np.abs(counterfactual_deltas).sum())

    fit_digest = _canonical_digest(
        {
            "edges": edges,
            "nodes": nodes,
            "rank": rank,
            "train_means": {name: float(train[name].mean()) for name in sorted(nodes)},
            "train_rows": int(train.shape[0]),
        }
    )
    attribution_digest = _canonical_digest(
        {
            "anomaly_rows": int(anomaly.shape[0]),
            "attribution": attribution_summary,
            "counterfactual_delta_sum": counterfactual_delta_sum,
            "counterfactual_absolute_delta_sum": (counterfactual_absolute_delta_sum),
            "rank": rank,
        }
    )
    runtime = ray.get_runtime_context()
    assigned = runtime.get_assigned_resources()
    return {
        "rank": rank,
        "world_size": world_size,
        "worker_id": str(runtime.get_worker_id()),
        "node_id": str(runtime.get_node_id()),
        "shard_id": hashlib.sha256(
            f"{binding_digest}:gcm:{rank}/{world_size}".encode("ascii")
        ).hexdigest(),
        "train_rows": int(train.shape[0]),
        "anomaly_rows": int(anomaly.shape[0]),
        "attribution": attribution_summary,
        "counterfactual_delta_sum": counterfactual_delta_sum,
        "counterfactual_absolute_delta_sum": counterfactual_absolute_delta_sum,
        "fit_digest": fit_digest,
        "attribution_digest": attribution_digest,
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
class GCMRootCauseResult:
    report: Mapping[str, object]
    stage_evidence: Mapping[str, Mapping[str, object]]
    composition_digest: str


class _GCMDriver:
    def __init__(
        self,
        *,
        plan: ResolvedAlgorithmPlan,
        datasets: Mapping[str, object],
    ) -> None:
        self.plan = plan
        self.datasets = dict(datasets)

    def fit(self) -> GCMRootCauseResult:
        import ray

        config = self.plan.algorithm_config
        data = cast(Mapping[str, Any], config["data"])
        gcm_config = cast(Mapping[str, Any], config["gcm"])
        nodes = tuple(str(name) for name in data["nodes"])
        edges = tuple(
            (str(cast(Sequence[object], edge)[0]), str(cast(Sequence[object], edge)[1]))
            for edge in cast(Sequence[object], data["edges"])
        )
        target_node = str(data["target_node"])
        interventions = {
            str(name): float(cast(int | float, value))
            for name, value in cast(
                Mapping[str, object], data.get("interventions", {})
            ).items()
        }
        world_size = self.plan.runtime.worker_count
        train_shards = cast(Any, self.datasets["train"]).split(world_size, equal=True)
        anomaly_shards = cast(Any, self.datasets["anomaly"]).split(
            world_size, equal=True
        )
        remote: Any = ray.remote(cast(Any, _run_gcm_shard)).options(
            num_cpus=self.plan.runtime.num_cpus,
            num_gpus=self.plan.runtime.num_gpus,
            resources=dict(self.plan.runtime.custom_resources),
            scheduling_strategy="SPREAD",
        )
        results = cast(
            list[dict[str, Any]],
            ray.get(
                [
                    remote.remote(
                        train_shard,
                        anomaly_shard,
                        nodes=nodes,
                        edges=edges,
                        target_node=target_node,
                        interventions=interventions,
                        quality=str(gcm_config.get("quality", "good")),
                        distribution_samples=int(
                            gcm_config.get("distribution_samples", 300)
                        ),
                        shapley_permutations=int(
                            gcm_config.get("shapley_permutations", 10)
                        ),
                        rank=rank,
                        world_size=world_size,
                        binding_digest=(
                            self.plan.primary_input_descriptor.binding_digest
                        ),
                    )
                    for rank, (train_shard, anomaly_shard) in enumerate(
                        zip(train_shards, anomaly_shards, strict=True)
                    )
                ]
            ),
        )
        train_rows = sum(int(item["train_rows"]) for item in results)
        anomaly_rows = sum(int(item["anomaly_rows"]) for item in results)
        attribution = {
            node: {
                "mean_absolute": sum(
                    float(
                        cast(Mapping[str, Mapping[str, float]], item["attribution"])[
                            node
                        ]["absolute_sum"]
                    )
                    for item in results
                )
                / anomaly_rows,
                "mean_signed": sum(
                    float(
                        cast(Mapping[str, Mapping[str, float]], item["attribution"])[
                            node
                        ]["signed_sum"]
                    )
                    for item in results
                )
                / anomaly_rows,
            }
            for node in nodes
        }
        counterfactual_delta = (
            sum(float(item["counterfactual_delta_sum"]) for item in results)
            / anomaly_rows
        )
        counterfactual_absolute_delta = (
            sum(float(item["counterfactual_absolute_delta_sum"]) for item in results)
            / anomaly_rows
        )
        fit_digest = _canonical_digest(
            sorted(str(item["fit_digest"]) for item in results)
        )
        attribution_digest = _canonical_digest(
            {
                "attribution": attribution,
                "counterfactual_target_delta": counterfactual_delta,
                "counterfactual_target_absolute_delta": (counterfactual_absolute_delta),
            }
        )

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
                    "rows_processed": item["train_rows"],
                    "input_rows": {
                        "train": item["train_rows"],
                        "coverage.anomaly": item["anomaly_rows"],
                    },
                    "batch_count": 1,
                    "collective_steps": 1,
                }
                for item in results
            ]

        stages = {
            "fit_gcm": {
                "workers": workers(fit_digest),
                "state": {
                    "coordination": "framework_native",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": fit_digest,
                    "details": {"framework": "dowhy-gcm", "stage": "fit_gcm"},
                },
                "input_complete": True,
                "expected_training_rows": train_rows,
            },
            "attribute_root_cause": {
                "workers": workers(attribution_digest),
                "state": {
                    "coordination": "framework_native",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": attribution_digest,
                    "details": {
                        "framework": "dowhy-gcm",
                        "stage": "attribute_root_cause",
                    },
                },
                "input_complete": True,
                "expected_training_rows": train_rows,
            },
        }
        composition_digest = _canonical_digest(
            {"fit": fit_digest, "root_cause": attribution_digest}
        )
        report: dict[str, object] = {
            "api_version": 1,
            "kind": "causal_gcm_root_cause",
            "method": "distributed_dowhy_gcm_shard_ensemble",
            "nodes": list(nodes),
            "edges": [list(edge) for edge in edges],
            "target_node": target_node,
            "train_rows": train_rows,
            "anomaly_rows": anomaly_rows,
            "root_cause_attribution": attribution,
            "interventions": interventions,
            "counterfactual_target_delta": counterfactual_delta,
            "counterfactual_target_absolute_delta": counterfactual_absolute_delta,
            "shard_count": world_size,
            "composition_digest": composition_digest,
            "limitations": [
                "shard GCM attribution is aggregated and is not identical to one centralized GCM fit"
            ],
        }
        return GCMRootCauseResult(
            report=report,
            stage_evidence=stages,
            composition_digest=composition_digest,
        )


class DistributedGCMRootCause(FrameworkNativeAlgorithm):
    """Fit shard GCMs and aggregate anomaly and counterfactual evidence."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan

    def validate_environment(self) -> None:
        try:
            import networkx
            from dowhy import gcm
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "DoWhy GCM and NetworkX dependencies are required"
            ) from exc
        if not networkx.__version__ or gcm is None:
            raise AlgorithmConfigurationError("DoWhy GCM environment is invalid")

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        if set(datasets) != {"train", "anomaly"}:
            raise AlgorithmConfigurationError(
                "GCM requires train and anomaly Dataset roles"
            )
        data = cast(Mapping[str, Any], self.plan.algorithm_config["data"])
        nodes = tuple(str(name) for name in data["nodes"])
        binding_names = {
            binding.name: binding for binding in self.plan.input_bindings.bindings
        }
        if any(
            set(binding_names[role].feature_names) != set(nodes)
            or binding_names[role].label_name is not None
            for role in ("train", "anomaly")
        ):
            raise AlgorithmConfigurationError("GCM InputBinding columns drifted")
        return {
            role: cast(Any, datasets[role]).select_columns(list(nodes))
            for role in ("train", "anomaly")
        }

    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        del config
        return _GCMDriver(plan=self.plan, datasets=datasets)

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        if not isinstance(result, GCMRootCauseResult):
            raise AlgorithmExecutionError("GCM returned an invalid result")
        return {
            "stages": dict(result.stage_evidence),
            "composition_digest": result.composition_digest,
        }

    def checkpoint_source(self, result: object) -> object:
        if not isinstance(result, GCMRootCauseResult):
            raise AlgorithmExecutionError("GCM checkpoint result is invalid")
        return result


def create_gcm_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedGCMRootCause:
    del artifacts
    if implementation is not DistributedGCMRootCause:
        raise AlgorithmConfigurationError("GCM implementation drifted")
    return DistributedGCMRootCause(plan)


def export_gcm_result(
    *,
    result: object,
    checkpoint: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    del checkpoint
    if not isinstance(result, GCMRootCauseResult):
        raise AlgorithmExecutionError("GCM export result is invalid")
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError("GCM output.bundle_uri is required")
    payload = json.dumps(result.report, sort_keys=True).encode("utf-8")
    artifact = ArtifactDraft.from_payload(
        name="gcm-root-cause-report",
        kind="report",
        format="application/json",
        payload=payload,
    )
    source = ExportSource(
        source_kind="causal_gcm",
        model_object=dict(result.report),
        metadata={
            "causal_study": dict(result.report),
            "framework": "dowhy-gcm",
            "producer_distribution": _PACKAGE,
        },
        source_fingerprint=artifact.sha256,
    )
    bundle = BundleExportService().export_bundle(
        source,
        BundleOutputConfig(
            bundle_uri=str(output["bundle_uri"]),
            request_id=run_id,
            run_id=run_id,
            targets=[
                ExportTarget(
                    name="gcm-root-cause-report",
                    format="json",
                    exporter_id="official-causal-gcm-report-v1",
                )
            ],
            roles={"report": "gcm-root-cause-report"},
        ),
    )
    attribution = cast(
        Mapping[str, Mapping[str, float]], result.report["root_cause_attribution"]
    )
    metrics = {
        f"root_cause.{name}.mean_absolute": float(values["mean_absolute"])
        for name, values in attribution.items()
    }
    metrics["counterfactual_target_delta"] = float(
        cast(int | float, result.report["counterfactual_target_delta"])
    )
    metrics["counterfactual_target_absolute_delta"] = float(
        cast(int | float, result.report["counterfactual_target_absolute_delta"])
    )
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics=metrics,
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
            "composition_digest": result.composition_digest,
        },
        artifacts=(artifact,),
    )


__all__ = [
    "DistributedGCMRootCause",
    "GCMRootCauseResult",
    "STAGES",
    "create_gcm_algorithm",
    "export_gcm_result",
]
