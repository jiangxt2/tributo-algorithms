"""Official descriptors for distributed classical algorithms."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    DistributedExactness,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    IterativeOptimizationPolicy,
    JoblibEstimatorPolicy,
    ParallelEnsemblePolicy,
    QualifiedReference,
    WorkerRange,
    WorkerResources,
)
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

_PACKAGE = "tributo-algorithms-classical"
_VERSION = "0.1.0"
_COMPATIBILITY = ">=1,<2"
_DEPENDENCIES = (
    "numpy>=2,<3",
    "onnx>=1.16",
    "onnxruntime>=1.20",
    "scikit-learn>=1.4,<2",
    "skl2onnx>=1.17",
    "tributo>=1,<2",
    f"{_PACKAGE}=={_VERSION}",
)


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_classical.contracts:{validator}"
        ),
    )


def _contracts(config_id: str, output_id: str) -> ContractBindingSet:
    return ContractBindingSet(
        config=_binding(config_id, "5", "ClassicalConfigValidator"),
        input=_binding(
            "tributo.official.tabular.labeled.v1",
            "2",
            "TabularInputValidator",
        ),
        output=_binding(output_id, "3", "SklearnOutputValidator"),
        coverage=_binding(
            "tributo.official.distributed.coverage.v1",
            "4",
            "DistributedCoverageValidator",
        ),
    )


def _spec(
    name: str,
    modes: tuple[ExecutionMode, ...],
    config_id: str,
    output_id: str,
) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(
            (ProblemType.REGRESSION,)
            if name == "linear_regression"
            else (ProblemType.BINARY_CLASSIFICATION,)
            if name == "logistic_regression"
            else (
                ProblemType.BINARY_CLASSIFICATION,
                ProblemType.MULTI_CLASS_CLASSIFICATION,
                ProblemType.REGRESSION,
            )
        ),
        capabilities=(
            Capability.DISTRIBUTED,
            Capability.EXPORTABLE,
            Capability.TUNABLE,
        ),
        learning_paradigm="supervised",
        model_family=(
            "tree_ensemble"
            if name in {"random_forest", "extra_trees"}
            else "linear_model"
        ),
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=tuple(mode.value for mode in modes),
        config_contract_ref=config_id,
        input_contract_ref="tributo.official.tabular.labeled.v1",
        output_contract_ref=output_id,
    )


_ENVIRONMENT = EnvironmentSpec(
    environment_id="tributo.official.classical.v1",
    dependencies=_DEPENDENCIES,
)
_PROFILES = (ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER)
_RESOURCES = WorkerResources(num_cpus=1)
_EXPORTER = "tributo_algorithms_classical.exporter:export_sklearn_model"

RANDOM_FOREST_JOBLIB_DESCRIPTOR = AlgorithmBuilder.from_joblib_estimator_recipe(
    spec=_spec(
        "random_forest",
        (ExecutionMode.JOBLIB_ESTIMATOR, ExecutionMode.PARALLEL_ENSEMBLE),
        "tributo.official.random_forest.config.v1",
        "tributo.official.random_forest.onnx.v1",
    ),
    implementation_id="tributo.official.random_forest.joblib",
    implementation_version="1.0.0",
    recipe=("tributo_algorithms_classical.random_forest:RandomForestJoblibRecipe"),
    environment=_ENVIRONMENT,
    allowed_config_keys=(
        "class_weight",
        "max_depth",
        "max_features",
        "n_estimators",
        "output",
        "seed",
        "task",
        "task_count",
    ),
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=_PROFILES,
    resources_per_worker=_RESOURCES,
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=_COMPATIBILITY,
    policy=JoblibEstimatorPolicy(max_materialized_rows=1_000_000),
    exporter=_EXPORTER,
    flavor_id="onnx-runtime-v1",
    contract_bindings=_contracts(
        "tributo.official.random_forest.config.v1",
        "tributo.official.random_forest.onnx.v1",
    ),
    descriptor_api_version=2,
    is_default=False,
)

RANDOM_FOREST_NATIVE_DESCRIPTOR = AlgorithmBuilder.from_parallel_ensemble(
    spec=_spec(
        "random_forest",
        (ExecutionMode.JOBLIB_ESTIMATOR, ExecutionMode.PARALLEL_ENSEMBLE),
        "tributo.official.random_forest.config.v1",
        "tributo.official.random_forest.onnx.v1",
    ),
    implementation_id="tributo.official.random_forest.native_ensemble",
    implementation_version="1.0.0",
    algorithm="tributo_algorithms_classical.random_forest:RandomForestEnsemble",
    environment=_ENVIRONMENT,
    allowed_config_keys=(
        "class_weight",
        "max_depth",
        "max_features",
        "n_estimators",
        "output",
        "runtime",
        "seed",
        "task",
        "unit_count",
    ),
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=_PROFILES,
    resources_per_worker=_RESOURCES,
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=_COMPATIBILITY,
    policy=ParallelEnsemblePolicy(
        max_units=4096,
        max_retries=1,
        exactness=DistributedExactness.CONDITIONAL,
    ),
    exporter=_EXPORTER,
    flavor_id="onnx-runtime-v1",
    contract_bindings=_contracts(
        "tributo.official.random_forest.config.v1",
        "tributo.official.random_forest.onnx.v1",
    ),
    descriptor_api_version=2,
    is_default=True,
)

EXTRA_TREES_JOBLIB_DESCRIPTOR = AlgorithmBuilder.from_joblib_estimator_recipe(
    spec=_spec(
        "extra_trees",
        (ExecutionMode.JOBLIB_ESTIMATOR, ExecutionMode.PARALLEL_ENSEMBLE),
        "tributo.official.extra_trees.config.v1",
        "tributo.official.extra_trees.onnx.v1",
    ),
    implementation_id="tributo.official.extra_trees.joblib",
    implementation_version="1.0.0",
    recipe="tributo_algorithms_classical.random_forest:ExtraTreesJoblibRecipe",
    environment=_ENVIRONMENT,
    allowed_config_keys=(
        "class_weight",
        "max_depth",
        "max_features",
        "n_estimators",
        "output",
        "seed",
        "task",
        "task_count",
    ),
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=_PROFILES,
    resources_per_worker=_RESOURCES,
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=_COMPATIBILITY,
    policy=JoblibEstimatorPolicy(max_materialized_rows=1_000_000),
    exporter=_EXPORTER,
    flavor_id="onnx-runtime-v1",
    contract_bindings=_contracts(
        "tributo.official.extra_trees.config.v1",
        "tributo.official.extra_trees.onnx.v1",
    ),
    descriptor_api_version=2,
    is_default=False,
)

EXTRA_TREES_NATIVE_DESCRIPTOR = AlgorithmBuilder.from_parallel_ensemble(
    spec=_spec(
        "extra_trees",
        (ExecutionMode.JOBLIB_ESTIMATOR, ExecutionMode.PARALLEL_ENSEMBLE),
        "tributo.official.extra_trees.config.v1",
        "tributo.official.extra_trees.onnx.v1",
    ),
    implementation_id="tributo.official.extra_trees.native_ensemble",
    implementation_version="1.0.0",
    algorithm="tributo_algorithms_classical.random_forest:ExtraTreesEnsemble",
    environment=_ENVIRONMENT,
    allowed_config_keys=(
        "class_weight",
        "max_depth",
        "max_features",
        "n_estimators",
        "output",
        "runtime",
        "seed",
        "task",
        "unit_count",
    ),
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=_PROFILES,
    resources_per_worker=_RESOURCES,
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=_COMPATIBILITY,
    policy=ParallelEnsemblePolicy(
        max_units=4096,
        max_retries=1,
        exactness=DistributedExactness.CONDITIONAL,
    ),
    exporter=_EXPORTER,
    flavor_id="onnx-runtime-v1",
    contract_bindings=_contracts(
        "tributo.official.extra_trees.config.v1",
        "tributo.official.extra_trees.onnx.v1",
    ),
    descriptor_api_version=2,
    is_default=True,
)

LOGISTIC_REGRESSION_DESCRIPTOR = AlgorithmBuilder.from_iterative_optimization(
    spec=_spec(
        "logistic_regression",
        (ExecutionMode.ITERATIVE_OPTIMIZATION,),
        "tributo.official.logistic_regression.config.v1",
        "tributo.official.logistic_regression.onnx.v1",
    ),
    implementation_id="tributo.official.logistic_regression.binary_l2",
    implementation_version="1.0.0",
    algorithm=(
        "tributo_algorithms_classical.logistic_regression:BinaryL2LogisticRegression"
    ),
    environment=_ENVIRONMENT,
    allowed_config_keys=(
        "C",
        "feature_count",
        "learning_rate",
        "output",
        "runtime",
        "seed",
        "tolerance",
    ),
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=_PROFILES,
    resources_per_worker=_RESOURCES,
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=_COMPATIBILITY,
    policy=IterativeOptimizationPolicy(
        max_rounds=200,
        checkpoint_interval=1,
        max_retries=1,
    ),
    exporter=_EXPORTER,
    flavor_id="onnx-runtime-v1",
    contract_bindings=_contracts(
        "tributo.official.logistic_regression.config.v1",
        "tributo.official.logistic_regression.onnx.v1",
    ),
    descriptor_api_version=2,
    is_default=True,
)

LINEAR_REGRESSION_DESCRIPTOR = AlgorithmBuilder.from_iterative_optimization(
    spec=_spec(
        "linear_regression",
        (ExecutionMode.ITERATIVE_OPTIMIZATION,),
        "tributo.official.linear_regression.config.v1",
        "tributo.official.linear_regression.onnx.v1",
    ),
    implementation_id="tributo.official.linear_regression.squared_l2",
    implementation_version="1.0.0",
    algorithm=(
        "tributo_algorithms_classical.linear_regression:DistributedLinearRegression"
    ),
    environment=_ENVIRONMENT,
    allowed_config_keys=(
        "feature_count",
        "learning_rate",
        "output",
        "runtime",
        "seed",
        "tolerance",
    ),
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=_PROFILES,
    resources_per_worker=_RESOURCES,
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=_COMPATIBILITY,
    policy=IterativeOptimizationPolicy(
        max_rounds=200,
        checkpoint_interval=1,
        max_retries=1,
    ),
    exporter=_EXPORTER,
    flavor_id="onnx-runtime-v1",
    contract_bindings=_contracts(
        "tributo.official.linear_regression.config.v1",
        "tributo.official.linear_regression.onnx.v1",
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = [
    "EXTRA_TREES_JOBLIB_DESCRIPTOR",
    "EXTRA_TREES_NATIVE_DESCRIPTOR",
    "LINEAR_REGRESSION_DESCRIPTOR",
    "LOGISTIC_REGRESSION_DESCRIPTOR",
    "RANDOM_FOREST_JOBLIB_DESCRIPTOR",
    "RANDOM_FOREST_NATIVE_DESCRIPTOR",
]
