"""Descriptors for binary and regression synchronous SGD."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    IterativeOptimizationPolicy,
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

_PACKAGE = "tributo-algorithms-classical"
_VERSION = "0.1.0"
_INPUT = "tributo.official.tabular.labeled.v1"
_COVERAGE = "tributo.official.distributed.coverage.v1"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_classical.sgd_contracts:{validator}"
        ),
    )


def _spec(name: str, problem_type: ProblemType, output: str) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        data_loading=DataLoadingMode.CANONICAL_DRIVER,
        problem_types=(problem_type,),
        capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE),
        learning_paradigm="supervised",
        model_family="linear_stochastic_optimizer",
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(ExecutionMode.ITERATIVE_OPTIMIZATION.value,),
        config_contract_ref=f"tributo.official.sgd.{name}.config.v1",
        input_contract_ref=_INPUT,
        output_contract_ref=output,
    )


def _descriptor(
    spec: AlgorithmSpec, implementation_id: str, implementation: str
) -> object:
    return AlgorithmBuilder.from_distributed_algorithm(
        spec=spec,
        implementation_id=implementation_id,
        implementation_version="1.0.0",
        implementation=implementation,
        executable_factory="tributo_algorithms_classical.sgd:create_algorithm",
        distribution=_PACKAGE,
        framework="sklearn",
        environment=EnvironmentSpec(
            environment_id="tributo.official.classical.sgd.v1",
            dependencies=(
                "numpy>=2,<3",
                "onnx>=1.16",
                "onnxruntime>=1.20",
                "scikit-learn>=1.4,<2",
                "skl2onnx>=1.17",
                "tributo>=1,<2",
                f"{_PACKAGE}=={_VERSION}",
            ),
        ),
        allowed_config_keys=(
            "alpha",
            "learning_rate",
            "learning_rate_decay",
            "loss",
            "max_iter",
            "output",
            "runtime",
            "seed",
            "tolerance",
        ),
        strategy=DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION,
        supported_worker_range=WorkerRange(2, 256),
        supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
        resources_per_worker=WorkerResources(num_cpus=1),
        policy=IterativeOptimizationPolicy(
            max_rounds=20, checkpoint_interval=5, max_retries=1
        ),
        package_name=_PACKAGE,
        package_version=_VERSION,
        tributo_version_spec=">=1,<2",
        exporter="tributo_algorithms_classical.exporter:export_sklearn_model",
        flavor_id="onnx-runtime-v1",
        contract_bindings=ContractBindingSet(
            config=_binding(spec.config_contract_ref or "", "c", "SGDConfigValidator"),
            input=_binding(_INPUT, "d", "SGDInputValidator"),
            output=_binding(spec.output_contract_ref or "", "e", "SGDOutputValidator"),
            coverage=_binding(_COVERAGE, "f", "SGDCoverageValidator"),
        ),
        descriptor_api_version=2,
        is_default=True,
    )


_CLASSIFIER_SPEC = _spec(
    "sgd_classifier",
    ProblemType.BINARY_CLASSIFICATION,
    "tributo.official.sgd.classifier.onnx.v1",
)
_REGRESSOR_SPEC = _spec(
    "sgd_regressor",
    ProblemType.REGRESSION,
    "tributo.official.sgd.regressor.onnx.v1",
)

SGD_CLASSIFIER_DESCRIPTOR = _descriptor(
    _CLASSIFIER_SPEC,
    "tributo.official.classical.sgd_classifier.iterative",
    "tributo_algorithms_classical.sgd:DistributedSGDClassifier",
)
SGD_REGRESSOR_DESCRIPTOR = _descriptor(
    _REGRESSOR_SPEC,
    "tributo.official.classical.sgd_regressor.iterative",
    "tributo_algorithms_classical.sgd:DistributedSGDRegressor",
)

__all__ = ["SGD_CLASSIFIER_DESCRIPTOR", "SGD_REGRESSOR_DESCRIPTOR"]
