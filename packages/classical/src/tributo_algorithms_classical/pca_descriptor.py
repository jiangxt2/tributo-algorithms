"""Descriptor for the official distributed PCA algorithm."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MapReducePolicy,
    QualifiedReference,
    StateField,
    WorkerRange,
    WorkerResources,
)
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, DataLoadingMode

_PACKAGE = "tributo-algorithms-classical"
_VERSION = "0.1.0"
_INPUT = "tributo.official.tabular.unlabeled_dense.v1"
_OUTPUT = "tributo.official.unsupervised.pca.transform.v1"
_COVERAGE = "tributo.official.distributed.coverage.v1"
_STATE_SCHEMA = (
    StateField("count", "int64", ()),
    StateField("mean", "float64", (None,)),
    StateField("M2", "float64", (None, None)),
)


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
    name="pca",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    data_loading=DataLoadingMode.CANONICAL_DRIVER,
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE),
    learning_paradigm="unsupervised",
    model_family="principal_component_analysis",
    data_modalities=("tabular",),
    lifecycle_kind="batch_fit",
    allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
    config_contract_ref="tributo.official.unsupervised.pca.config.v1",
    input_contract_ref=_INPUT,
    output_contract_ref=_OUTPUT,
)

PCA_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.classical.pca.map_reduce",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_classical.pca:DistributedPCA",
    executable_factory="tributo_algorithms_classical.pca:create_algorithm",
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
    allowed_config_keys=("feature_count", "n_components", "output", "runtime", "seed"),
    strategy=DistributionStrategy.RAY_MAP_REDUCE,
    supported_worker_range=WorkerRange(1, 1024),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=MapReducePolicy(
        state_schema=_STATE_SCHEMA,
        max_partial_state_bytes=64 * 1024 * 1024,
        reducer_ref="tributo_algorithms_classical.pca:DistributedPCA.merge_states",
        finalizer_ref="tributo_algorithms_classical.pca:DistributedPCA.finalize_model",
        commutative=True,
        max_retries=0,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_classical.unsupervised_exporter:export_pca_model",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(
            _SPEC.config_contract_ref or "", "6", "UnsupervisedConfigValidator"
        ),
        input=_binding(_INPUT, "7", "UnlabeledDenseInputValidator"),
        output=_binding(_OUTPUT, "8", "PCAOutputValidator"),
        coverage=_binding(_COVERAGE, "a", "UnsupervisedCoverageValidator"),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["PCA_DESCRIPTOR"]
