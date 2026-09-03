"""Descriptors for GraphSAGE and R-GCN RayTorchAdapter algorithms."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MetricReduction,
    QualifiedReference,
    SingleStageTorchPlan,
    TorchDatasetRoute,
    TorchPolicy,
    TorchStageSpec,
    WorkerRange,
    WorkerResources,
)
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

from tributo_algorithms_graph_pyg.contracts import (
    GraphConfigValidator,
    GraphOutputValidator,
    GraphSAGETorchCoverageValidator,
    GraphSAGETorchInputValidator,
    RelationalGraphConfigValidator,
    RGCNTorchCoverageValidator,
    RGCNTorchInputValidator,
)

_PACKAGE = "tributo-algorithms-graph-pyg"
_VERSION = "0.1.0"
_ROOT = Path(__file__).resolve().parent


def _code_digest() -> str:
    return hashlib.sha256((_ROOT / "algorithm.py").read_bytes()).hexdigest()


def _binding(
    contract_id: str, validator: type[Any], version: int = 2
) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=version,
        schema_digest=str(validator.schema_digest),
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_graph_pyg.contracts:{validator.__name__}"
        ),
    )


def _policy(*, relational: bool) -> TorchPolicy:
    return TorchPolicy(
        torch_runtime_api_version=1,
        loop_owner="adapter",
        parallelism_id="torch.ddp.replicated",
        dataset_routing=(
            TorchDatasetRoute("train", "split_exact", True, 1, 1, "reject"),
            TorchDatasetRoute(
                "nodes", "replicate", True, 1, 1, "reject", 1_000_000, 268_435_456
            ),
            TorchDatasetRoute(
                "edges", "replicate", True, 1, 1, "reject", 5_000_000, 268_435_456
            ),
        ),
        execution_plan=SingleStageTorchPlan(
            stage=TorchStageSpec(
                "train",
                "tributo.integrations.algorithm_runtimes.ray_train_torch:ray_torch_adapter_train_loop_per_worker",
                ("train", "nodes", "edges"),
                metric_mapping={"train_loss": "train_loss", "accuracy": "accuracy"},
            )
        ),
        state_layout="replicated",
        metric_reducers={
            "train_loss": MetricReduction.SUM_COUNT,
            "accuracy": MetricReduction.SUM_COUNT,
        },
        backend="auto",
        resume_supported=False,
        same_world_size_resume=None,
        max_replicated_bytes_per_worker=536_870_912,
    )


def _spec(*, relational: bool) -> AlgorithmSpec:
    name = "rgcn_node_classifier" if relational else "graphsage_node_classifier"
    return AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(ProblemType.NODE_CLASSIFICATION,),
        capabilities=(
            Capability.DISTRIBUTED,
            Capability.EXPORTABLE,
            Capability.TUNABLE,
        ),
        learning_paradigm="supervised",
        model_family="relational_gcn" if relational else "graphsage",
        data_modalities=("relational_graph" if relational else "graph",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(ExecutionMode.RAY_TRAIN_TORCH.value,),
        config_contract_ref=(
            "tributo.official.rgcn.config.v1"
            if relational
            else "tributo.official.graphsage.config.v1"
        ),
        input_contract_ref=(
            "tributo.official.graph.relational-torch.v2"
            if relational
            else "tributo.official.graph.homogeneous-torch.v2"
        ),
        output_contract_ref=(
            "tributo.official.rgcn.bundle.v1"
            if relational
            else "tributo.official.graphsage.bundle.v1"
        ),
    )


_ENVIRONMENT = EnvironmentSpec(
    environment_id="tributo.official.graph-pyg.v2",
    dependencies=(
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "safetensors>=0.4.3",
        "torch>=2.5",
        "torch-geometric>=2.5",
        "tributo>=1,<2",
        f"{_PACKAGE}=={_VERSION}",
    ),
)


def _descriptor(*, relational: bool) -> object:
    spec = _spec(relational=relational)
    input_validator = (
        RGCNTorchInputValidator if relational else GraphSAGETorchInputValidator
    )
    coverage_validator = (
        RGCNTorchCoverageValidator if relational else GraphSAGETorchCoverageValidator
    )
    implementation_id = (
        "tributo.official.graph_pyg.rgcn"
        if relational
        else "tributo.official.graph_pyg.graphsage"
    )
    adapter = (
        "tributo_algorithms_graph_pyg.algorithm:DistributedRGCN"
        if relational
        else "tributo_algorithms_graph_pyg.algorithm:DistributedGraphSAGE"
    )
    contracts = ContractBindingSet(
        config=_binding(
            spec.config_contract_ref or "",
            RelationalGraphConfigValidator if relational else GraphConfigValidator,
            1,
        ),
        input=_binding(spec.input_contract_ref or "", input_validator),
        output=_binding(spec.output_contract_ref or "", GraphOutputValidator, 1),
        coverage=_binding(
            "tributo.official.rgcn.torch-coverage.v2"
            if relational
            else "tributo.official.graphsage.torch-coverage.v2",
            coverage_validator,
        ),
    )
    return AlgorithmBuilder.from_torch_adapter(
        spec=spec,
        implementation_id=implementation_id,
        implementation_version="2.0.0",
        adapter=adapter,
        environment=_ENVIRONMENT,
        metric_reducers={
            "train_loss": MetricReduction.SUM_COUNT,
            "accuracy": MetricReduction.SUM_COUNT,
        },
        supported_worker_range=WorkerRange(2, 64),
        supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
        resources_per_worker=WorkerResources(num_cpus=1),
        package_name=_PACKAGE,
        package_version=_VERSION,
        tributo_version_spec=">=1,<2",
        policy=_policy(relational=relational),
        code_digest=_code_digest(),
        contract_bindings=contracts,
        descriptor_api_version=2,
        is_default=True,
    )


GRAPHSAGE_DESCRIPTOR = _descriptor(relational=False)
RGCN_DESCRIPTOR = _descriptor(relational=True)

__all__ = ["GRAPHSAGE_DESCRIPTOR", "RGCN_DESCRIPTOR"]
