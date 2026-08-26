"""Descriptor for official out-of-tree distributed XGBoost."""

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

_PACKAGE = "tributo-algorithms-boosting"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_boosting.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="xgboost",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    resource_hints=ResourceHints(gpu_required=False),
    extras_group="training",
    problem_types=(
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTI_CLASS_CLASSIFICATION,
        ProblemType.REGRESSION,
    ),
    data_modality=("tabular",),
    capabilities=(
        Capability.TUNABLE,
        Capability.EXPORTABLE,
        Capability.DISTRIBUTED,
    ),
    data_loading=DataLoadingMode.CANONICAL_DRIVER,
    operations=("fit",),
    learning_paradigm="supervised",
    model_family="gradient_boosted_trees",
    data_modalities=("tabular",),
    lifecycle_kind="bounded_training",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.algorithm-config.xgboost.v1",
    input_contract_ref="tributo.algorithm-input.ray-tabular.v1",
    output_contract_ref="tributo.algorithm-result.execution-only.v1",
)

XGBOOST_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.boosting.xgboost",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_boosting.algorithm:DistributedXGBoost",
    executable_factory="tributo_algorithms_boosting.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework="xgboost",
    environment=EnvironmentSpec(
        environment_id="tributo.official.boosting.v1",
        dependencies=(
            "onnxmltools>=1.13",
            "xgboost>=2.1,<4",
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
        framework="xgboost-rabit",
        evidence_collector_ref=(
            "tributo_algorithms_boosting.algorithm:DistributedXGBoost.collect_evidence"
        ),
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_boosting.algorithm:export_result",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(_SPEC.config_contract_ref or "", "1", "XGBoostConfigValidator"),
        input=_binding(_SPEC.input_contract_ref or "", "2", "XGBoostInputValidator"),
        output=_binding(_SPEC.output_contract_ref or "", "3", "XGBoostOutputValidator"),
        coverage=_binding(
            "tributo.official.xgboost.coverage.v1",
            "4",
            "XGBoostCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["XGBOOST_DESCRIPTOR"]
