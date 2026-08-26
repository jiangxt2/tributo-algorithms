"""Lightweight descriptor for distributed PC stability discovery."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MapReducePolicy,
    QualifiedReference,
    StateField,
    WorkerRange,
    WorkerResources,
)
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
    flavor_id="report",
    contract_bindings=ContractBindingSet(
        config=_contract(_SPEC.config_contract_ref or "", "6", "PCConfigValidator"),
        input=_contract(
            _SPEC.input_contract_ref or "",
            "7",
            "DiscoveryInputValidator",
        ),
        output=_contract(
            _SPEC.output_contract_ref or "",
            "8",
            "DiscoveryOutputValidator",
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


__all__ = ["PC_DISCOVERY_DESCRIPTOR"]
