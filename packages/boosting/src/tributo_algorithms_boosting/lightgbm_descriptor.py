"""Descriptor for official distributed LightGBM."""

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
)

_PACKAGE = "tributo-algorithms-boosting"
_VERSION = "0.1.0"
_INPUT = "tributo.algorithm-input.ray-tabular.v1"
_OUTPUT = "tributo.official.lightgbm.onnx.v1"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_boosting.lightgbm_contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="lightgbm",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    problem_types=(
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTI_CLASS_CLASSIFICATION,
        ProblemType.REGRESSION,
    ),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    data_loading=DataLoadingMode.CANONICAL_DRIVER,
    operations=("fit",),
    learning_paradigm="supervised",
    model_family="gradient_boosted_trees",
    data_modalities=("tabular",),
    lifecycle_kind="bounded_training",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.algorithm-config.lightgbm.v1",
    input_contract_ref=_INPUT,
    output_contract_ref=_OUTPUT,
)

LIGHTGBM_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.boosting.lightgbm",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_boosting.lightgbm:DistributedLightGBM",
    executable_factory="tributo_algorithms_boosting.lightgbm:create_algorithm",
    distribution=_PACKAGE,
    framework="lightgbm",
    environment=EnvironmentSpec(
        environment_id="tributo.official.boosting.lightgbm.v1",
        dependencies=(
            "lightgbm>=4.5,<5",
            "onnxmltools>=1.13",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=("data", "model", "output", "ray", "resource", "training"),
    strategy=DistributionStrategy.FRAMEWORK_NATIVE,
    supported_worker_range=WorkerRange(2, 1024),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=FrameworkNativePolicy(
        framework="lightgbm-data-parallel",
        evidence_collector_ref=(
            "tributo_algorithms_boosting.lightgbm:DistributedLightGBM.collect_evidence"
        ),
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_boosting.lightgbm:export_result",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(
            _SPEC.config_contract_ref or "", "5", "LightGBMConfigValidator"
        ),
        input=_binding(_INPUT, "6", "LightGBMInputValidator"),
        output=_binding(_OUTPUT, "7", "LightGBMOutputValidator"),
        coverage=_binding(
            "tributo.official.lightgbm.coverage.v1", "8", "LightGBMCoverageValidator"
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["LIGHTGBM_DESCRIPTOR"]
