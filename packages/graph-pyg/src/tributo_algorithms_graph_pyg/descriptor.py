"""Descriptor for the official distributed homogeneous GraphSAGE classifier."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    FrameworkNativePolicy,
    QualifiedReference,
    WorkerRange,
    WorkerResources,
)
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

_PACKAGE = "tributo-algorithms-graph-pyg"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_graph_pyg.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="graphsage_node_classifier",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.NODE_CLASSIFICATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="supervised",
    model_family="graphsage",
    data_modalities=("graph",),
    lifecycle_kind="batch_fit",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.official.graphsage.config.v1",
    input_contract_ref="tributo.official.graph.homogeneous.v1",
    output_contract_ref="tributo.official.graphsage.safetensors.v1",
)

_RGCN_SPEC = AlgorithmSpec(
    name="rgcn_node_classifier",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.NODE_CLASSIFICATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="supervised",
    model_family="relational_gcn",
    data_modalities=("relational_graph",),
    lifecycle_kind="batch_fit",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.official.rgcn.config.v1",
    input_contract_ref="tributo.official.graph.relational.v1",
    output_contract_ref="tributo.official.rgcn.safetensors.v1",
)

_CONTRACTS = ContractBindingSet(
    config=_binding(_SPEC.config_contract_ref or "", "1", "GraphConfigValidator"),
    input=_binding(
        _SPEC.input_contract_ref or "", "2", "HomogeneousGraphInputValidator"
    ),
    output=_binding(_SPEC.output_contract_ref or "", "3", "GraphOutputValidator"),
    coverage=_binding(
        "tributo.official.graph.full-neighborhood.coverage.v1",
        "4",
        "GraphCoverageValidator",
    ),
)

GRAPHSAGE_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.graph_pyg.graphsage",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_graph_pyg.algorithm:DistributedGraphSAGE",
    executable_factory="tributo_algorithms_graph_pyg.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework="pytorch-geometric",
    environment=EnvironmentSpec(
        environment_id="tributo.official.graph-pyg.v1",
        dependencies=(
            "safetensors>=0.4.3",
            "torch>=2.5",
            "torch-geometric>=2.5",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=("model", "output", "ray", "training"),
    strategy=DistributionStrategy.FRAMEWORK_NATIVE,
    supported_worker_range=WorkerRange(2, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=FrameworkNativePolicy(
        framework="pytorch-geometric-ddp",
        evidence_collector_ref=(
            "tributo_algorithms_graph_pyg.algorithm:"
            "DistributedGraphSAGE.collect_evidence"
        ),
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_graph_pyg.algorithm:export_result",
    flavor_id="safetensors-v1",
    contract_bindings=_CONTRACTS,
    descriptor_api_version=2,
    is_default=True,
)

RGCN_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_RGCN_SPEC,
    implementation_id="tributo.official.graph_pyg.rgcn",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_graph_pyg.algorithm:DistributedRGCN",
    executable_factory="tributo_algorithms_graph_pyg.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework="pytorch-geometric",
    environment=GRAPHSAGE_DESCRIPTOR.registration.environment,
    allowed_config_keys=("model", "output", "ray", "training"),
    strategy=DistributionStrategy.FRAMEWORK_NATIVE,
    supported_worker_range=WorkerRange(2, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=FrameworkNativePolicy(
        framework="pytorch-geometric-rgcn-ddp",
        evidence_collector_ref=(
            "tributo_algorithms_graph_pyg.algorithm:DistributedRGCN.collect_evidence"
        ),
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_graph_pyg.algorithm:export_result",
    flavor_id="safetensors-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(
            _RGCN_SPEC.config_contract_ref or "", "5", "RelationalGraphConfigValidator"
        ),
        input=_binding(
            _RGCN_SPEC.input_contract_ref or "", "6", "RelationalGraphInputValidator"
        ),
        output=_binding(
            _RGCN_SPEC.output_contract_ref or "", "3", "GraphOutputValidator"
        ),
        coverage=_binding(
            "tributo.official.graph.relational-coverage.v1",
            "7",
            "RelationalGraphCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["GRAPHSAGE_DESCRIPTOR", "RGCN_DESCRIPTOR"]
