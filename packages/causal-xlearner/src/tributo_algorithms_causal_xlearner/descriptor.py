"""Descriptor for official distributed five-stage X-Learner."""

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
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    Capability,
    DataLoadingMode,
    ProblemType,
    ResourceHints,
)

STAGES = ("mu0", "mu1", "tau0", "tau1", "propensity")

_PACKAGE = "tributo-algorithms-causal-xlearner"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_causal_xlearner.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="x_learner",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    resource_hints=ResourceHints(gpu_required=False),
    extras_group="training",
    problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
    data_modality=("tabular",),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE),
    data_loading=DataLoadingMode.LEGACY_DRIVER,
    operations=("fit",),
    learning_paradigm="causal_meta_learner",
    model_family="x_learner",
    data_modalities=("tabular",),
    lifecycle_kind="causal_estimate",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.x_learner.config.v1",
    input_contract_ref="tributo.causal.binary_tabular.v1",
    output_contract_ref="tributo.causal.x_learner_bundle.v1",
)

X_LEARNER_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.causal_xlearner.xgboost",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_causal_xlearner.algorithm:DistributedXLearner",
    executable_factory="tributo_algorithms_causal_xlearner.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework="xgboost",
    environment=EnvironmentSpec(
        environment_id="tributo.official.causal-xlearner.v1",
        dependencies=(
            "tributo-algorithms-boosting==0.1.0",
            "xgboost>=2.1,<4",
            "onnx>=1.16",
            "onnxmltools>=1.13",
            "onnxruntime>=1.20",
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
        framework="xgboost-x-learner",
        evidence_collector_ref=(
            "tributo_algorithms_causal_xlearner.algorithm:"
            "DistributedXLearner.collect_evidence"
        ),
        component_stages=STAGES,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_causal_xlearner.exporter:export_result",
    flavor_id="official-x-learner-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(
            _SPEC.config_contract_ref or "", "6", "XLearnerConfigValidator"
        ),
        input=_binding(_SPEC.input_contract_ref or "", "2", "XLearnerInputValidator"),
        output=_binding(
            _SPEC.output_contract_ref or "", "3", "XLearnerOutputValidator"
        ),
        coverage=_binding(
            "tributo.official.x-learner.stage-coverage.v1",
            "4",
            "XLearnerCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["X_LEARNER_DESCRIPTOR"]
