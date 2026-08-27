"""Descriptor for the official distributed Isolation Forest."""

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

_PACKAGE = "tributo-algorithms-classical"
_VERSION = "0.1.0"
_INPUT = "tributo.official.tabular.unlabeled_dense.v1"
_OUTPUT = "tributo.official.unsupervised.isolation_forest.anomaly.v1"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_classical.unsupervised_contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="isolation_forest",
    trainer_cls=None,
    version="1.0.0",
    default_config={"n_estimators": 100, "contamination": "auto"},
    supported_tasks=("fit",),
    operations=("fit",),
    data_loading=DataLoadingMode.CANONICAL_DRIVER,
    problem_types=(ProblemType.ANOMALY_DETECTION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="unsupervised",
    model_family="isolation_forest",
    data_modalities=("tabular",),
    lifecycle_kind="batch_fit",
    allowed_execution_modes=(ExecutionMode.PARALLEL_ENSEMBLE.value,),
    config_contract_ref="tributo.official.unsupervised.isolation_forest.config.v1",
    input_contract_ref=_INPUT,
    output_contract_ref=_OUTPUT,
)

ISOLATION_FOREST_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.classical.isolation_forest.parallel_ensemble",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_classical.isolation_forest:IsolationForestEnsemble",
    executable_factory="tributo_algorithms_classical.isolation_forest:create_algorithm",
    distribution=_PACKAGE,
    framework="sklearn",
    environment=EnvironmentSpec(
        environment_id="tributo.official.classical.unsupervised.v1",
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
        "contamination",
        "max_samples",
        "n_estimators",
        "output",
        "runtime",
        "seed",
        "unit_count",
    ),
    strategy=DistributionStrategy.RAY_PARALLEL_ENSEMBLE,
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=ParallelEnsemblePolicy(
        max_units=4096,
        max_retries=1,
        exactness=DistributedExactness.CONDITIONAL,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_classical.unsupervised_exporter:export_isolation_forest_model",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(
            _SPEC.config_contract_ref or "", "6", "UnsupervisedConfigValidator"
        ),
        input=_binding(_INPUT, "7", "UnlabeledDenseInputValidator"),
        output=_binding(_OUTPUT, "b", "IsolationForestOutputValidator"),
        coverage=_binding(
            "tributo.official.distributed.coverage.v1",
            "a",
            "UnsupervisedCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["ISOLATION_FOREST_DESCRIPTOR"]
