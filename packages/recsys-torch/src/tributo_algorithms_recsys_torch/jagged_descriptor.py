"""Descriptor for distributed jagged-history recommendation."""

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

_PACKAGE = "tributo-algorithms-recsys-torch"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_recsys_torch.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="jagged_embedding_recommender",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.RANKING,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="supervised_implicit_feedback",
    model_family="jagged_embedding_ddp_alltoall_routing",
    data_modalities=("interaction", "jagged_sequence"),
    lifecycle_kind="batch_fit",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.official.jagged-recsys.config.v2",
    input_contract_ref="tributo.official.jagged-recsys.interactions.v1",
    output_contract_ref="tributo.official.jagged-recsys.bundle.v1",
)

JAGGED_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.recsys_torch.jagged_embedding",
    implementation_version="1.0.0",
    implementation=(
        "tributo_algorithms_recsys_torch.jagged:DistributedJaggedEmbedding"
    ),
    executable_factory=(
        "tributo_algorithms_recsys_torch.jagged:create_jagged_algorithm"
    ),
    distribution=_PACKAGE,
    framework="pytorch",
    environment=EnvironmentSpec(
        environment_id="tributo.official.recsys-torch-jagged.v1",
        dependencies=(
            "onnx>=1.16",
            "onnxruntime>=1.20",
            "safetensors>=0.4.3",
            "torch>=2.5",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=("data", "model", "output", "ray", "training"),
    strategy=DistributionStrategy.FRAMEWORK_NATIVE,
    supported_worker_range=WorkerRange(2, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=FrameworkNativePolicy(
        framework="pytorch-ddp-jagged-alltoall",
        evidence_collector_ref=(
            "tributo_algorithms_recsys_torch.jagged:"
            "DistributedJaggedEmbedding.collect_evidence"
        ),
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_recsys_torch.jagged:export_jagged_result",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(_SPEC.config_contract_ref or "", "9", "JaggedConfigValidator"),
        input=_binding(_SPEC.input_contract_ref or "", "6", "JaggedInputValidator"),
        output=_binding(_SPEC.output_contract_ref or "", "7", "JaggedOutputValidator"),
        coverage=_binding(
            "tributo.official.jagged-recsys.coverage.v1",
            "8",
            "JaggedCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["JAGGED_DESCRIPTOR"]
