"""Distributed stability-selection PC causal discovery."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    ArtifactDraft,
    ContractBinding,
    ContractBindingSet,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MapReducePolicy,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    StateField,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.spi import AlgorithmExecutionContext, MapReduceAlgorithm
from tributo.exporting.models import BundleOutputConfig, ExportSource, ExportTarget
from tributo.exporting.service import BundleExportService
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

_PACKAGE = "tributo-algorithms-causal-discovery"
_VERSION = "0.1.0"
_STATE_SCHEMA = (
    StateField("adjacency_votes", "int64", (None, None)),
    StateField("arrow_votes", "int64", (None, None)),
    StateField("tail_votes", "int64", (None, None)),
    StateField("run_count", "int64", ()),
    StateField("row_count", "int64", ()),
)


@dataclass(frozen=True)
class CausalGraphModel:
    variables: tuple[str, ...]
    endpoint_matrix: tuple[tuple[int, ...], ...]
    adjacency_vote_fraction: tuple[tuple[float, ...], ...]
    run_count: int
    row_count: int
    alpha: float
    vote_threshold: float

    def report(self) -> dict[str, object]:
        edges = []
        for left in range(len(self.variables)):
            for right in range(left + 1, len(self.variables)):
                left_endpoint = self.endpoint_matrix[left][right]
                right_endpoint = self.endpoint_matrix[right][left]
                if left_endpoint == 0 or right_endpoint == 0:
                    continue
                edges.append(
                    {
                        "left": self.variables[left],
                        "right": self.variables[right],
                        "left_endpoint": left_endpoint,
                        "right_endpoint": right_endpoint,
                        "vote_fraction": self.adjacency_vote_fraction[left][right],
                    }
                )
        return {
            "api_version": 1,
            "kind": "causal_discovery",
            "method": "distributed_shard_stability_pc",
            "variables": list(self.variables),
            "edges": edges,
            "endpoint_encoding": {"tail": -1, "arrow": 1, "absent": 0},
            "alpha": self.alpha,
            "vote_threshold": self.vote_threshold,
            "shard_runs": self.run_count,
            "row_count": self.row_count,
            "limitations": [
                "stability-selected shard PC is not bitwise equivalent to centralized PC"
            ],
        }


def _query_onnx(model: CausalGraphModel) -> bytes:
    """Build a bounded ONNX graph for exact causal-edge report queries."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    endpoints = np.asarray(model.endpoint_matrix, dtype=np.int64)
    votes = np.asarray(model.adjacency_vote_fraction, dtype=np.float32)
    node_count = len(model.variables)
    graph = helper.make_graph(
        [
            helper.make_node("GreaterOrEqual", ["edge_index", "min_index"], ["ge"]),
            helper.make_node("Less", ["edge_index", "max_exclusive"], ["lt"]),
            helper.make_node("And", ["ge", "lt"], ["element_valid"]),
            helper.make_node(
                "Cast",
                ["element_valid"],
                ["element_valid_int"],
                to=TensorProto.INT64,
            ),
            helper.make_node(
                "ReduceMin",
                ["element_valid_int", "row_axis"],
                ["valid_int"],
                keepdims=0,
            ),
            helper.make_node("Cast", ["valid_int"], ["valid"], to=TensorProto.BOOL),
            helper.make_node(
                "Clip",
                ["edge_index", "min_index", "max_index"],
                ["safe_edge_index"],
            ),
            helper.make_node(
                "GatherND",
                ["endpoint_matrix", "safe_edge_index"],
                ["raw_left_endpoint"],
            ),
            helper.make_node(
                "Gather",
                ["safe_edge_index", "reverse_order"],
                ["reverse_edge_index"],
                axis=1,
            ),
            helper.make_node(
                "GatherND",
                ["endpoint_matrix", "reverse_edge_index"],
                ["raw_right_endpoint"],
            ),
            helper.make_node(
                "GatherND",
                ["vote_matrix", "safe_edge_index"],
                ["raw_vote_fraction"],
            ),
            helper.make_node(
                "Where",
                ["valid", "raw_left_endpoint", "invalid_endpoint"],
                ["left_endpoint"],
            ),
            helper.make_node(
                "Where",
                ["valid", "raw_right_endpoint", "invalid_endpoint"],
                ["right_endpoint"],
            ),
            helper.make_node(
                "Where",
                ["valid", "raw_vote_fraction", "invalid_vote"],
                ["vote_fraction"],
            ),
        ],
        "tributo_pc_edge_query",
        [helper.make_tensor_value_info("edge_index", TensorProto.INT64, [None, 2])],
        [
            helper.make_tensor_value_info("left_endpoint", TensorProto.INT64, [None]),
            helper.make_tensor_value_info("right_endpoint", TensorProto.INT64, [None]),
            helper.make_tensor_value_info("vote_fraction", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("valid", TensorProto.BOOL, [None]),
        ],
        initializer=[
            numpy_helper.from_array(endpoints, name="endpoint_matrix"),
            numpy_helper.from_array(votes, name="vote_matrix"),
            numpy_helper.from_array(np.asarray(0, dtype=np.int64), name="min_index"),
            numpy_helper.from_array(
                np.asarray(node_count, dtype=np.int64), name="max_exclusive"
            ),
            numpy_helper.from_array(
                np.asarray(node_count - 1, dtype=np.int64), name="max_index"
            ),
            numpy_helper.from_array(
                np.asarray([1, 0], dtype=np.int64), name="reverse_order"
            ),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="row_axis"),
            numpy_helper.from_array(
                np.asarray(0, dtype=np.int64), name="invalid_endpoint"
            ),
            numpy_helper.from_array(
                np.asarray(0.0, dtype=np.float32), name="invalid_vote"
            ),
        ],
    )
    built = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="tributo-algorithms-causal-discovery",
    )
    onnx.checker.check_model(built)
    return cast(bytes, built.SerializeToString())


class DistributedPCStability(
    MapReduceAlgorithm[Mapping[str, object], Mapping[str, object], CausalGraphModel]
):
    """Run causal-learn PC per shard and aggregate stable endpoint votes."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        self.variables = plan.primary_input_binding.feature_names
        self.alpha = float(plan.algorithm_config.get("alpha", 0.05))
        self.vote_threshold = float(plan.algorithm_config.get("vote_threshold", 0.5))
        raw_max_k = plan.algorithm_config.get("max_condition_set")
        self.max_k = int(raw_max_k) if raw_max_k is not None else None

    def map_partition(
        self,
        batches: Iterable[Mapping[str, object]],
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        del context
        import numpy as np
        from causallearn.search.ConstraintBased.PC import pc

        pieces = []
        for batch in batches:
            missing = [name for name in self.variables if name not in batch]
            if missing:
                raise AlgorithmInputError(f"PC batch is missing variables: {missing}")
            values = np.column_stack(
                [np.asarray(batch[name], dtype=np.float64) for name in self.variables]
            )
            if not np.isfinite(values).all():
                raise AlgorithmInputError("PC discovery requires finite values")
            pieces.append(values)
        if not pieces:
            return self.empty_partition()
        data = np.vstack(pieces)
        if data.shape[0] < max(8, data.shape[1] + 2):
            raise AlgorithmInputError("PC shard has insufficient rows")
        try:
            discovered = pc(
                data,
                alpha=self.alpha,
                stable=True,
                show_progress=False,
                node_names=list(self.variables),
                max_k=self.max_k,
            )
            graph = np.asarray(discovered.G.graph, dtype=np.int64)
        except Exception as exc:
            raise AlgorithmExecutionError(
                f"causal-learn PC failed: {type(exc).__name__}"
            ) from exc
        if graph.shape != (len(self.variables), len(self.variables)):
            raise AlgorithmExecutionError("PC returned an invalid graph matrix")
        return {
            "adjacency_votes": np.asarray(graph != 0, dtype=np.int64),
            "arrow_votes": np.asarray(graph == 1, dtype=np.int64),
            "tail_votes": np.asarray(graph == -1, dtype=np.int64),
            "run_count": np.asarray(1, dtype=np.int64),
            "row_count": np.asarray(data.shape[0], dtype=np.int64),
        }

    def merge_states(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        import numpy as np

        return {
            item.name: np.asarray(
                np.asarray(left[item.name], dtype=item.dtype)
                + np.asarray(right[item.name], dtype=item.dtype),
                dtype=item.dtype,
            )
            for item in _STATE_SCHEMA
        }

    def finalize_model(self, state: Mapping[str, object]) -> CausalGraphModel:
        import numpy as np

        runs = int(np.asarray(state["run_count"]))
        if runs < 1:
            raise AlgorithmInputError("PC discovery produced no shard graph")
        adjacency = np.asarray(state["adjacency_votes"], dtype=np.int64)
        arrows = np.asarray(state["arrow_votes"], dtype=np.int64)
        tails = np.asarray(state["tail_votes"], dtype=np.int64)
        required_votes = math.ceil(runs * self.vote_threshold)
        stable = adjacency >= required_votes
        endpoints = np.where(arrows > tails, 1, -1).astype(np.int64)
        endpoints[~stable] = 0
        np.fill_diagonal(endpoints, 0)
        fractions = adjacency.astype(np.float64) / runs
        return CausalGraphModel(
            variables=self.variables,
            endpoint_matrix=tuple(
                tuple(int(value) for value in row) for row in endpoints
            ),
            adjacency_vote_fraction=tuple(
                tuple(float(value) for value in row) for row in fractions
            ),
            run_count=runs,
            row_count=int(np.asarray(state["row_count"])),
            alpha=self.alpha,
            vote_threshold=self.vote_threshold,
        )

    def state_schema(self) -> tuple[StateField, ...]:
        return _STATE_SCHEMA

    def empty_partition(self) -> Mapping[str, object]:
        import numpy as np

        width = len(self.variables)
        return {
            "adjacency_votes": np.zeros((width, width), dtype=np.int64),
            "arrow_votes": np.zeros((width, width), dtype=np.int64),
            "tail_votes": np.zeros((width, width), dtype=np.int64),
            "run_count": np.asarray(0, dtype=np.int64),
            "row_count": np.asarray(0, dtype=np.int64),
        }

    def coverage_counts(self, state: Mapping[str, object]) -> Mapping[str, int]:
        import numpy as np

        return {"discovery_rows": int(np.asarray(state["row_count"]))}

    @property
    def retry_safe(self) -> bool:
        return True


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedPCStability:
    del artifacts
    if implementation is not DistributedPCStability:
        raise AlgorithmConfigurationError("PC implementation identity drifted")
    return DistributedPCStability(plan)


def export_graph(
    *,
    model: CausalGraphModel,
    plan: ResolvedAlgorithmPlan,
    run_id: str | None = None,
) -> AlgorithmExecutionResult:
    report = model.report()
    payload = json.dumps(report, sort_keys=True).encode("utf-8")
    artifact = ArtifactDraft.from_payload(
        name="causal-discovery-report",
        kind="report",
        format="application/json",
        payload=payload,
    )
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError("PC output.bundle_uri is required")
    from tributo.exporting.models import CheckpointField, ExportCheckpointV1

    source = ExportSource(
        source_kind="prebuilt_onnx",
        model_object=_query_onnx(model),
        architecture_id=plan.resolution.implementation_id,
        metadata={
            "causal_study": report,
            "framework": "causal-learn",
            "framework_versions": {"onnx_opset": "18"},
            "producer_distribution": _PACKAGE,
        },
        source_fingerprint=artifact.sha256,
        checkpoint_contract=ExportCheckpointV1(
            trainer_type="pc_stability_discovery",
            architecture_id=plan.resolution.implementation_id,
            input_schema=(
                CheckpointField(name="edge_index", dtype="int64", shape=("batch", 2)),
            ),
            output_schema=(
                CheckpointField(name="left_endpoint", dtype="int64", shape=("batch",)),
                CheckpointField(name="right_endpoint", dtype="int64", shape=("batch",)),
                CheckpointField(
                    name="vote_fraction", dtype="float32", shape=("batch",)
                ),
                CheckpointField(name="valid", dtype="bool", shape=("batch",)),
            ),
            preprocessing={
                "type": "variable_index_pair",
                "variables": list(model.variables),
                "invalid_index_policy": "zero_output_with_valid_false",
            },
            task_type="causal_graph_query",
            framework="onnx",
            framework_version="18",
        ),
    )
    bundle = BundleExportService().export_bundle(
        source,
        BundleOutputConfig(
            bundle_uri=str(output["bundle_uri"]),
            request_id=run_id,
            run_id=run_id,
            targets=[
                ExportTarget(
                    name="causal-graph-query",
                    format="onnx",
                    exporter_id="prebuilt-onnx-v1",
                ),
                ExportTarget(
                    name="causal-graph-report",
                    format="json",
                    exporter_id="official-causal-discovery-report-v1",
                ),
            ],
            roles={
                "inference": "causal-graph-query",
                "report": "causal-graph-report",
            },
        ),
    )
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics={"shard_runs": model.run_count, "row_count": model.row_count},
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
            "report_artifact_sha256": artifact.sha256,
        },
        artifacts=(artifact,),
    )


def _contract(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_causal_discovery.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="pc_stability_discovery",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="causal_discovery",
    model_family="pc_stability",
    data_modalities=("tabular",),
    lifecycle_kind="discover_validate_report",
    allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
    config_contract_ref="tributo.official.causal.pc.config.v1",
    input_contract_ref="tributo.official.causal.discovery-table.v1",
    output_contract_ref="tributo.official.causal.discovery-report.v1",
)

PC_DISCOVERY_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.causal_discovery.pc_stability",
    implementation_version="1.0.0",
    implementation=(
        "tributo_algorithms_causal_discovery.algorithm:DistributedPCStability"
    ),
    executable_factory="tributo_algorithms_causal_discovery.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework="causal-learn",
    environment=EnvironmentSpec(
        environment_id="tributo.official.causal-discovery.v1",
        dependencies=(
            "causal-learn>=0.1.4,<0.2",
            "numpy>=2,<3",
            "onnx>=1.16",
            "onnxruntime>=1.20",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=("alpha", "max_condition_set", "output", "vote_threshold"),
    strategy=DistributionStrategy.RAY_MAP_REDUCE,
    supported_worker_range=WorkerRange(2, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=MapReducePolicy(
        state_schema=_STATE_SCHEMA,
        max_partial_state_bytes=64 * 1024 * 1024,
        reducer_ref=(
            "tributo_algorithms_causal_discovery.algorithm:"
            "DistributedPCStability.merge_states"
        ),
        finalizer_ref=(
            "tributo_algorithms_causal_discovery.algorithm:"
            "DistributedPCStability.finalize_model"
        ),
        commutative=True,
        max_retries=0,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_causal_discovery.algorithm:export_graph",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_contract(_SPEC.config_contract_ref or "", "6", "PCConfigValidator"),
        input=_contract(_SPEC.input_contract_ref or "", "7", "DiscoveryInputValidator"),
        output=_contract(
            _SPEC.output_contract_ref or "", "8", "DiscoveryOutputValidator"
        ),
        coverage=_contract(
            "tributo.official.causal.discovery-coverage.v1",
            "9",
            "DiscoveryCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["PC_DISCOVERY_DESCRIPTOR", "CausalGraphModel", "DistributedPCStability"]
