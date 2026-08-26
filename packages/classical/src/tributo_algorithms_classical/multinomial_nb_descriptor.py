"""Lightweight descriptor for the official distributed MultinomialNB."""

from __future__ import annotations

from tributo.algorithms.api import (
    AlgorithmOperation,
    AlgorithmRegistration,
    BackendInputCompatibility,
    ContractBinding,
    ContractBindingSet,
    DistributedAlgorithmDescriptor,
    DistributionSpec,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    ImplementationDescriptor,
    InputDistribution,
    MapReducePolicy,
    QualifiedReference,
    RuntimeTopology,
    StateCoordination,
    StateField,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

_STATE_SCHEMA = (
    StateField("classes", "int64", (None,)),
    StateField("class_count", "float64", (None,)),
    StateField("feature_count", "float64", (None, None)),
    StateField("row_count", "int64", ()),
)
_MAP_REDUCE_CONTRACT = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.RAY_MAP_REDUCE
]
_INPUT_ADAPTER = QualifiedReference.parse(_MAP_REDUCE_CONTRACT.worker_input_adapter_ref)


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_classical.contracts:{validator}"
        ),
    )


_CONTRACTS = ContractBindingSet(
    config=_binding(
        "tributo.multinomial_nb.config.v1",
        "5",
        "ClassicalConfigValidator",
    ),
    input=_binding(
        "tributo.tabular.nonnegative.v1",
        "2",
        "TabularInputValidator",
    ),
    output=_binding(
        "tributo.classification.onnx.v1",
        "3",
        "SklearnOutputValidator",
    ),
    coverage=_binding(
        "tributo.official.distributed.coverage.v1",
        "4",
        "DistributedCoverageValidator",
    ),
)

MULTINOMIAL_NB_REGISTRATION = AlgorithmRegistration(
    spec=AlgorithmSpec(
        name="multinomial_nb",
        trainer_cls=None,
        version="1.0.0",
        default_config={"alpha": 1.0, "force_alpha": True, "fit_prior": True},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTI_CLASS_CLASSIFICATION,
        ),
        capabilities=(Capability.EXPORTABLE, Capability.DISTRIBUTED),
        extras_group="training",
        learning_paradigm="supervised",
        model_family="naive_bayes",
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
        config_contract_ref="tributo.multinomial_nb.config.v1",
        input_contract_ref="tributo.tabular.nonnegative.v1",
        output_contract_ref="tributo.classification.onnx.v1",
    ),
    implementation=ImplementationDescriptor(
        implementation_id="tributo.official.multinomial_nb.map_reduce",
        version="1.0.0",
        execution_mode=ExecutionMode.MAP_REDUCE,
        implementation_ref=QualifiedReference.parse(
            "tributo_algorithms_classical.multinomial_nb:DistributedMultinomialNB"
        ),
        executable_factory_ref=QualifiedReference.parse(
            "tributo_algorithms_classical.multinomial_nb:create_algorithm"
        ),
        operations=(AlgorithmOperation.FIT,),
        input_compatibility=BackendInputCompatibility(
            accepted_input_views=("ray_data",),
            accepted_ingestion_engines=("tributo.ray_data",),
            required_input_capabilities=("shardable",),
            supported_explicit_adapters=(_INPUT_ADAPTER,),
            distribution_policy=(RuntimeTopology.RAY_MAP_REDUCE,),
        ),
        distribution="tributo-algorithms-classical",
        framework="sklearn",
        artifact_format="none",
        allowed_config_keys=(
            "alpha",
            "class_prior",
            "fit_prior",
            "force_alpha",
            "output",
        ),
        runtime_id=_MAP_REDUCE_CONTRACT.runtime_id,
        worker_input_adapter_ref=_INPUT_ADAPTER,
        exporter_ref=QualifiedReference.parse(
            "tributo_algorithms_classical.multinomial_nb:export_model"
        ),
        flavor_id="onnx-runtime-v1",
    ),
    environment=EnvironmentSpec(
        environment_id="tributo.multinomial_nb.v1",
        dependencies=(
            "tributo-algorithms-classical==0.1.0",
            "tributo>=1,<2",
            "onnx>=1.16.0",
            "onnxruntime>=1.20.0",
            "scikit-learn>=1.4,<2",
            "skl2onnx>=1.17",
        ),
    ),
    distribution_spec=DistributionSpec(
        strategy=DistributionStrategy.RAY_MAP_REDUCE,
        supported_worker_range=WorkerRange(1, 1024),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.CLUSTER,
        ),
        resources_per_worker=WorkerResources(num_cpus=1, num_gpus=0),
        input_distribution=InputDistribution.SHARDED,
        state_coordination=StateCoordination.ASSOCIATIVE_REDUCE,
        policy=MapReducePolicy(
            state_schema=_STATE_SCHEMA,
            max_partial_state_bytes=64 * 1024 * 1024,
            reducer_ref=(
                "tributo_algorithms_classical.multinomial_nb:"
                "DistributedMultinomialNB.merge_states"
            ),
            finalizer_ref=(
                "tributo_algorithms_classical.multinomial_nb:"
                "DistributedMultinomialNB.finalize_model"
            ),
            commutative=True,
            max_retries=0,
        ),
    ),
    contract_bindings=_CONTRACTS,
    is_default=True,
)

MULTINOMIAL_NB_DESCRIPTOR = DistributedAlgorithmDescriptor(
    registration=MULTINOMIAL_NB_REGISTRATION,
    package_name="tributo-algorithms-classical",
    package_version="0.1.0",
    tributo_version_spec=">=1,<2",
    api_version=2,
)


__all__ = ["MULTINOMIAL_NB_DESCRIPTOR", "MULTINOMIAL_NB_REGISTRATION"]
