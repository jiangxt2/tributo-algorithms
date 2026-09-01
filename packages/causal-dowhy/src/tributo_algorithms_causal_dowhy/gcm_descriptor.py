"""Descriptor for distributed DoWhy GCM root-cause analysis."""

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

STAGES = ("fit_gcm", "attribute_root_cause")

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
    name="gcm_root_cause",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(
        ProblemType.ANOMALY_DETECTION,
        ProblemType.CAUSAL_EFFECT_ESTIMATION,
    ),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="causal_root_cause_and_counterfactual",
    model_family="dowhy_gcm_shard_ensemble",
    data_modalities=("tabular",),
    lifecycle_kind="fit_attribute_counterfactual_report",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.official.causal.gcm.config.v1",
    input_contract_ref="tributo.official.causal.gcm.train-anomaly.v1",
    output_contract_ref="tributo.official.causal.gcm.report-bundle.v1",
)

GCM_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.causal_dowhy.gcm_root_cause",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_causal_dowhy.gcm:DistributedGCMRootCause",
    executable_factory=("tributo_algorithms_causal_dowhy.gcm:create_gcm_algorithm"),
    distribution=_PACKAGE,
    framework="dowhy-gcm",
    environment=EnvironmentSpec(
        environment_id="tributo.official.causal-dowhy-gcm.v1",
        dependencies=(
            "dowhy>=0.13,<0.15",
            "networkx>=3.0",
            "onnx>=1.16",
            "onnxruntime>=1.20",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=("data", "gcm", "output", "runtime"),
    strategy=DistributionStrategy.FRAMEWORK_NATIVE,
    supported_worker_range=WorkerRange(2, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=FrameworkNativePolicy(
        framework="dowhy-gcm-shard-ensemble",
        evidence_collector_ref=(
            "tributo_algorithms_causal_dowhy.gcm:"
            "DistributedGCMRootCause.collect_evidence"
        ),
        component_stages=STAGES,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_causal_dowhy.gcm:export_gcm_result",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(_SPEC.config_contract_ref or "", "2", "GCMConfigValidator"),
        input=_binding(_SPEC.input_contract_ref or "", "3", "GCMInputValidator"),
        output=_binding(_SPEC.output_contract_ref or "", "4", "GCMOutputValidator"),
        coverage=_binding(
            "tributo.official.causal.gcm-coverage.v1",
            "5",
            "GCMCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["GCM_DESCRIPTOR"]
