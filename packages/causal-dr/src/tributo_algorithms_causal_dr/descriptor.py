"""Descriptor for official distributed doubly robust ATE."""

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

STAGES = ("mu0", "mu1", "propensity")

_PACKAGE = "tributo-algorithms-causal-dr"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_causal_dr.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="doubly_robust_ate",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="causal_inference",
    model_family="aipw_doubly_robust",
    data_modalities=("tabular",),
    lifecycle_kind="nuisance_fit_estimate_refute_policy",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.official.causal.dr.config.v1",
    input_contract_ref="tributo.official.causal.binary-treatment.v1",
    output_contract_ref="tributo.official.causal.report-bundle.v1",
)

DR_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.causal_dr.aipw",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_causal_dr.algorithm:DistributedDRLearner",
    executable_factory="tributo_algorithms_causal_dr.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework="xgboost",
    environment=EnvironmentSpec(
        environment_id="tributo.official.causal-dr.v1",
        dependencies=(
            "tributo-algorithms-boosting==0.1.0",
            "tributo-algorithms-causal-core==0.1.0",
            "xgboost>=2.1,<4",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=("data", "model", "output", "ray", "training"),
    strategy=DistributionStrategy.FRAMEWORK_NATIVE,
    supported_worker_range=WorkerRange(2, 128),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=FrameworkNativePolicy(
        framework="xgboost-aipw",
        evidence_collector_ref=(
            "tributo_algorithms_causal_dr.algorithm:"
            "DistributedDRLearner.collect_evidence"
        ),
        component_stages=STAGES,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_causal_dr.algorithm:export_result",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(_SPEC.config_contract_ref or "", "e", "DRConfigValidator"),
        input=_binding(_SPEC.input_contract_ref or "", "b", "DRInputValidator"),
        output=_binding(_SPEC.output_contract_ref or "", "c", "DROutputValidator"),
        coverage=_binding(
            "tributo.official.causal.dr-stage-coverage.v1",
            "d",
            "DRCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["DR_DESCRIPTOR"]
