"""Descriptor for the official conditional CatBoost ensemble."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    DistributedExactness,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    ParallelEnsemblePolicy,
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

_PACKAGE = "tributo-algorithms-catboost"
_VERSION = "0.1.0"
_INPUT = "tributo.official.tabular.labeled.v1"
_OUTPUT = "tributo.official.catboost.native.v1"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_catboost.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="catboost",
    trainer_cls=None,
    version="1.0.0",
    default_config={"task": "classification"},
    supported_tasks=("fit",),
    operations=("fit",),
    data_loading=DataLoadingMode.CANONICAL_DRIVER,
    problem_types=(
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTI_CLASS_CLASSIFICATION,
        ProblemType.REGRESSION,
    ),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE),
    learning_paradigm="supervised",
    model_family="categorical_gradient_boosted_trees",
    data_modalities=("tabular",),
    lifecycle_kind="batch_fit",
    allowed_execution_modes=(ExecutionMode.PARALLEL_ENSEMBLE.value,),
    config_contract_ref="tributo.official.catboost.config.v1",
    input_contract_ref=_INPUT,
    output_contract_ref=_OUTPUT,
)

CATBOOST_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.catboost.parallel_ensemble",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_catboost.algorithm:CatBoostEnsemble",
    executable_factory="tributo_algorithms_catboost.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework="catboost",
    environment=EnvironmentSpec(
        environment_id="tributo.official.catboost.v1",
        dependencies=(
            "catboost>=1.2,<2",
            "onnxruntime>=1.20",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=(
        "model",
        "n_estimators",
        "output",
        "runtime",
        "seed",
        "task",
        "unit_count",
    ),
    strategy=DistributionStrategy.RAY_PARALLEL_ENSEMBLE,
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    policy=ParallelEnsemblePolicy(
        max_units=256,
        max_retries=1,
        exactness=DistributedExactness.CONDITIONAL,
    ),
    exporter="tributo_algorithms_catboost.algorithm:export_result",
    flavor_id="catboost-native-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(
            _SPEC.config_contract_ref or "", "5", "CatBoostConfigValidator"
        ),
        input=_binding(_INPUT, "6", "CatBoostInputValidator"),
        output=_binding(_OUTPUT, "7", "CatBoostOutputValidator"),
        coverage=_binding(
            "tributo.official.catboost.coverage.v1", "8", "CatBoostCoverageValidator"
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["CATBOOST_DESCRIPTOR"]
