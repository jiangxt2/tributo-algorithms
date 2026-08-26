"""Descriptor for distributed DoWhy estimate and refutation."""

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

STAGES = ("estimate", "refute")

_PACKAGE = "tributo-algorithms-causal-dowhy"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_causal_dowhy.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="dowhy_linear_refutation",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="causal_inference",
    model_family="dowhy_linear_refutation_ensemble",
    data_modalities=("tabular",),
    lifecycle_kind="identify_estimate_refute_policy",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.official.causal.dowhy.config.v1",
    input_contract_ref="tributo.official.causal.binary-treatment.v1",
    output_contract_ref="tributo.official.causal.report-bundle.v1",
)

DOWHY_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.causal_dowhy.linear_refutation",
    implementation_version="1.0.0",
    implementation=(
        "tributo_algorithms_causal_dowhy.algorithm:DistributedDoWhyRefutation"
    ),
    executable_factory="tributo_algorithms_causal_dowhy.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework="dowhy",
    environment=EnvironmentSpec(
        environment_id="tributo.official.causal-dowhy.v1",
        dependencies=(
            "dowhy>=0.13,<0.15",
            "tributo-algorithms-causal-core==0.1.0",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=("data", "output", "refutation", "runtime"),
    strategy=DistributionStrategy.FRAMEWORK_NATIVE,
    supported_worker_range=WorkerRange(2, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=FrameworkNativePolicy(
        framework="dowhy-shard-ensemble",
        evidence_collector_ref=(
            "tributo_algorithms_causal_dowhy.algorithm:"
            "DistributedDoWhyRefutation.collect_evidence"
        ),
        component_stages=STAGES,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_causal_dowhy.algorithm:export_result",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(_SPEC.config_contract_ref or "", "e", "DoWhyConfigValidator"),
        input=_binding(_SPEC.input_contract_ref or "", "f", "DoWhyInputValidator"),
        output=_binding(_SPEC.output_contract_ref or "", "0", "DoWhyOutputValidator"),
        coverage=_binding(
            "tributo.official.causal.dowhy-stage-coverage.v1",
            "1",
            "DoWhyCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["DOWHY_DESCRIPTOR"]
