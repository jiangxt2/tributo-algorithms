"""Descriptors for official distributed dense DNN and PU algorithms."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MetricReduction,
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

_PACKAGE = "tributo-algorithms-tabular-torch"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_tabular_torch.contracts:{validator}"
        ),
    )


def _spec(name: str) -> AlgorithmSpec:
    pu = name == "pu"
    return AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        resource_hints=ResourceHints(gpu_required=False),
        extras_group="identity",
        problem_types=(
            (ProblemType.PU_LEARNING,) if pu else (ProblemType.BINARY_CLASSIFICATION,)
        ),
        data_modality=("tabular",),
        capabilities=(
            Capability.TUNABLE,
            Capability.EXPORTABLE,
            Capability.DISTRIBUTED,
        ),
        data_loading=(
            DataLoadingMode.CANONICAL_TRAINER
            if pu
            else DataLoadingMode.CANONICAL_DRIVER
        ),
        learning_paradigm="positive_unlabeled" if pu else "supervised",
        model_family="deep_neural_network",
        data_modalities=("tabular",),
        lifecycle_kind="bounded_training",
        allowed_execution_modes=(ExecutionMode.TRAINING_RECIPE_V2.value,),
        config_contract_ref=f"tributo.algorithm-config.{name}.v1",
        input_contract_ref="tributo.algorithm-input.ray-tabular.v1",
        output_contract_ref="tributo.algorithm-result.execution-only.v1",
    )


def _contracts(
    spec: AlgorithmSpec,
    config_digest: str,
    validator: str,
    *,
    coverage_digest: str = "9",
    coverage_validator: str = "TabularTorchCoverageValidator",
) -> ContractBindingSet:
    return ContractBindingSet(
        config=_binding(spec.config_contract_ref or "", config_digest, validator),
        input=_binding(
            spec.input_contract_ref or "", "7", "LabeledDenseInputValidator"
        ),
        output=_binding(
            spec.output_contract_ref or "", "8", "TabularTorchOutputValidator"
        ),
        coverage=_binding(
            "tributo.official.tabular-torch.coverage.v1",
            coverage_digest,
            coverage_validator,
        ),
    )


_DNN_SPEC = _spec("dnn")
_PU_SPEC = _spec("pu")
_ENVIRONMENT = EnvironmentSpec(
    environment_id="tributo.official.tabular-torch.v1",
    dependencies=(
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "torch>=2.5",
        "tributo>=1,<2",
        f"{_PACKAGE}=={_VERSION}",
    ),
)

DNN_DESCRIPTOR = AlgorithmBuilder.from_training_recipe_v2(
    spec=_DNN_SPEC,
    implementation_id="tributo.official.tabular_torch.dnn",
    implementation_version="1.0.0",
    recipe="tributo_algorithms_tabular_torch.recipe:DNNRecipe",
    environment=_ENVIRONMENT,
    metric_reducers={"accuracy": MetricReduction.SUM_COUNT},
    supported_worker_range=WorkerRange(1, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    contract_bindings=_contracts(_DNN_SPEC, "5", "TabularTorchConfigValidator"),
    descriptor_api_version=2,
    is_default=True,
)

PU_DESCRIPTOR = AlgorithmBuilder.from_training_recipe_v2(
    spec=_PU_SPEC,
    implementation_id="tributo.official.tabular_torch.pu",
    implementation_version="1.0.0",
    recipe="tributo_algorithms_tabular_torch.recipe:PURecipe",
    environment=_ENVIRONMENT,
    metric_reducers={"observed_positive_recall": MetricReduction.SUM_COUNT},
    supported_worker_range=WorkerRange(1, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    contract_bindings=_contracts(
        _PU_SPEC,
        "6",
        "PUConfigValidator",
        coverage_digest="a",
        coverage_validator="PUCoverageValidator",
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["DNN_DESCRIPTOR", "PU_DESCRIPTOR"]
