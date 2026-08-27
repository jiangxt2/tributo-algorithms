"""Descriptors for the official KMeans algorithms."""

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
_INPUT = "tributo.official.tabular.unlabeled_dense.v1"
_OUTPUT = "tributo.official.unsupervised.kmeans.clustering.v1"
_COVERAGE = "tributo.official.distributed.coverage.v1"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_classical.unsupervised_contracts:{validator}"
        ),
    )


def _spec(name: str) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        default_config={"n_clusters": 8},
        supported_tasks=("fit",),
        operations=("fit",),
        data_loading=DataLoadingMode.CANONICAL_DRIVER,
        capabilities=(
            Capability.DISTRIBUTED,
            Capability.EXPORTABLE,
            Capability.TUNABLE,
        ),
        problem_types=(ProblemType.CLUSTERING,),
        learning_paradigm="unsupervised",
        model_family="centroid_clustering",
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(ExecutionMode.ITERATIVE_OPTIMIZATION.value,),
        config_contract_ref=f"tributo.official.unsupervised.{name}.config.v1",
        input_contract_ref=_INPUT,
        output_contract_ref=_OUTPUT,
    )


def _descriptor(
    spec: AlgorithmSpec,
    implementation_id: str,
    implementation: str,
    exactness: DistributedExactness,
) -> object:
    return AlgorithmBuilder.from_distributed_algorithm(
        spec=spec,
        implementation_id=implementation_id,
        implementation_version="1.0.0",
        implementation=implementation,
        executable_factory="tributo_algorithms_classical.kmeans:create_algorithm",
        distribution=_PACKAGE,
        framework="numpy",
        environment=EnvironmentSpec(
            environment_id="tributo.official.classical.unsupervised.v1",
            dependencies=(
                "numpy>=2,<3",
                "onnx>=1.16",
                "onnxruntime>=1.20",
                "tributo>=1,<2",
                f"{_PACKAGE}=={_VERSION}",
            ),
        ),
        allowed_config_keys=(
            "batch_size",
            "feature_count",
            "learning_rate",
            "max_iter",
            "n_clusters",
            "output",
            "runtime",
            "seed",
            "tolerance",
        ),
        strategy=DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION,
        supported_worker_range=WorkerRange(1, 1024),
        supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
        resources_per_worker=WorkerResources(num_cpus=1),
        policy=IterativeOptimizationPolicy(
            max_rounds=100,
            checkpoint_interval=10,
            max_retries=1,
            exactness=exactness,
        ),
        package_name=_PACKAGE,
        package_version=_VERSION,
        tributo_version_spec=">=1,<2",
        exporter="tributo_algorithms_classical.unsupervised_exporter:export_kmeans_model",
        flavor_id="onnx-runtime-v1",
        contract_bindings=ContractBindingSet(
            config=_binding(
                spec.config_contract_ref or "", "6", "UnsupervisedConfigValidator"
            ),
            input=_binding(_INPUT, "7", "UnlabeledDenseInputValidator"),
            output=_binding(_OUTPUT, "9", "KMeansOutputValidator"),
            coverage=_binding(_COVERAGE, "a", "UnsupervisedCoverageValidator"),
        ),
        descriptor_api_version=2,
        is_default=True,
    )


KMEANS_DESCRIPTOR = _descriptor(
    _spec("kmeans"),
    "tributo.official.classical.kmeans.iterative",
    "tributo_algorithms_classical.kmeans:DistributedKMeans",
    DistributedExactness.EXACT,
)

MINIBATCH_KMEANS_DESCRIPTOR = _descriptor(
    _spec("kmeans_minibatch"),
    "tributo.official.classical.kmeans_minibatch.iterative",
    "tributo_algorithms_classical.kmeans:MiniBatchKMeans",
    DistributedExactness.APPROXIMATE,
)

__all__ = ["KMEANS_DESCRIPTOR", "MINIBATCH_KMEANS_DESCRIPTOR"]
