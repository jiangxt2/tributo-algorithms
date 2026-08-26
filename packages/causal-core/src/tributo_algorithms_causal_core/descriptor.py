"""Lightweight descriptors for official distributed causal estimators."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    DistributedAlgorithmDescriptor,
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
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

_PACKAGE = "tributo-algorithms-causal-core"
_VERSION = "0.1.0"
_COMMON_STATE = (
    StateField("treated_count", "int64", ()),
    StateField("control_count", "int64", ()),
    StateField("treated_sum", "float64", ()),
    StateField("control_sum", "float64", ()),
    StateField("row_count", "int64", ()),
)
_STATE_SCHEMA = (
    *_COMMON_STATE[:-1],
    StateField("treated_sum_squares", "float64", ()),
    StateField("control_sum_squares", "float64", ()),
    _COMMON_STATE[-1],
)
_DML_STATE_SCHEMA = (
    StateField("xtx", "float64", (None, None)),
    StateField("xty", "float64", (None,)),
    StateField("xtt", "float64", (None,)),
    StateField("yty", "float64", ()),
    StateField("ytt", "float64", ()),
    StateField("ttt", "float64", ()),
    *_COMMON_STATE,
    StateField("fold_xtx", "float64", (None, None, None)),
    StateField("fold_xty", "float64", (None, None)),
    StateField("fold_xtt", "float64", (None, None)),
    StateField("fold_yty", "float64", (None,)),
    StateField("fold_ytt", "float64", (None,)),
    StateField("fold_ttt", "float64", (None,)),
)
_IV_STATE_SCHEMA = (
    StateField("xtx", "float64", (None, None)),
    StateField("xty", "float64", (None,)),
    StateField("xtt", "float64", (None,)),
    StateField("xtz", "float64", (None,)),
    StateField("yty", "float64", ()),
    StateField("ytt", "float64", ()),
    StateField("ttt", "float64", ()),
    StateField("ytz", "float64", ()),
    StateField("ttz", "float64", ()),
    StateField("ztz", "float64", ()),
    *_COMMON_STATE,
)


def _contract(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_causal_core.contracts:{validator}"
        ),
    )


def _spec(
    *,
    name: str,
    model_family: str,
    lifecycle_kind: str,
    config_contract_ref: str,
    input_contract_ref: str,
) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
        capabilities=(
            Capability.DISTRIBUTED,
            Capability.EXPORTABLE,
            Capability.TUNABLE,
        ),
        learning_paradigm="causal_inference",
        model_family=model_family,
        data_modalities=("tabular",),
        lifecycle_kind=lifecycle_kind,
        allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
        config_contract_ref=config_contract_ref,
        input_contract_ref=input_contract_ref,
        output_contract_ref="tributo.official.causal.report-bundle.v1",
    )


_ATE_SPEC = _spec(
    name="difference_in_means_ate",
    model_family="difference_in_means",
    lifecycle_kind="identify_estimate_policy",
    config_contract_ref="tributo.official.causal.ate.config.v1",
    input_contract_ref="tributo.official.causal.binary-treatment.v1",
)
_DML_SPEC = _spec(
    name="linear_dml_ate",
    model_family="linear_dml",
    lifecycle_kind="identify_crossfit_estimate_policy",
    config_contract_ref="tributo.official.causal.linear-dml.config.v1",
    input_contract_ref="tributo.official.causal.binary-treatment.v1",
)
_IV_SPEC = _spec(
    name="linear_iv_ate",
    model_family="linear_instrumental_variables",
    lifecycle_kind="identify_first_stage_estimate_policy",
    config_contract_ref="tributo.official.causal.linear-iv.config.v1",
    input_contract_ref="tributo.official.causal.binary-treatment-instrument.v1",
)
_ENVIRONMENT = EnvironmentSpec(
    environment_id="tributo.official.causal-core.v1",
    dependencies=(
        "numpy>=2,<3",
        "scikit-learn>=1.4,<2",
        "skl2onnx>=1.17",
        "tributo>=1,<2",
        f"{_PACKAGE}=={_VERSION}",
    ),
)


def _bindings(
    spec: AlgorithmSpec,
    *,
    config_digest: str,
    config_validator: str,
    coverage_id: str,
) -> ContractBindingSet:
    return ContractBindingSet(
        config=_contract(
            spec.config_contract_ref or "",
            config_digest,
            config_validator,
        ),
        input=_contract(
            spec.input_contract_ref or "",
            "2",
            "CausalInputValidator",
        ),
        output=_contract(
            spec.output_contract_ref or "",
            "3",
            "CausalOutputValidator",
        ),
        coverage=_contract(coverage_id, "4", "TreatmentCoverageValidator"),
    )


def _descriptor(
    *,
    spec: AlgorithmSpec,
    implementation_id: str,
    implementation_class: str,
    state_schema: tuple[StateField, ...],
    allowed_config_keys: tuple[str, ...],
    contract_bindings: ContractBindingSet,
) -> DistributedAlgorithmDescriptor:
    implementation = f"tributo_algorithms_causal_core.algorithm:{implementation_class}"
    return AlgorithmBuilder.from_distributed_algorithm(
        spec=spec,
        implementation_id=implementation_id,
        implementation_version="1.0.0",
        implementation=implementation,
        executable_factory="tributo_algorithms_causal_core.algorithm:create_algorithm",
        distribution=_PACKAGE,
        framework=None,
        environment=_ENVIRONMENT,
        allowed_config_keys=allowed_config_keys,
        strategy=DistributionStrategy.RAY_MAP_REDUCE,
        supported_worker_range=WorkerRange(2, 256),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.CLUSTER,
        ),
        resources_per_worker=WorkerResources(num_cpus=1),
        policy=MapReducePolicy(
            state_schema=state_schema,
            max_partial_state_bytes=64 * 1024 * 1024,
            reducer_ref=f"{implementation}.merge_states",
            finalizer_ref=f"{implementation}.finalize_model",
            commutative=True,
            max_retries=0,
        ),
        package_name=_PACKAGE,
        package_version=_VERSION,
        tributo_version_spec=">=1,<2",
        exporter="tributo_algorithms_causal_core.algorithm:export_model",
        flavor_id="onnx-runtime-v1",
        contract_bindings=contract_bindings,
        descriptor_api_version=2,
        is_default=True,
    )


ATE_DESCRIPTOR = _descriptor(
    spec=_ATE_SPEC,
    implementation_id="tributo.official.causal.difference_in_means",
    implementation_class="DifferenceInMeansATE",
    state_schema=_STATE_SCHEMA,
    allowed_config_keys=("confidence_z", "output", "policy_cost", "treatment_col"),
    contract_bindings=_bindings(
        _ATE_SPEC,
        config_digest="6",
        config_validator="ATEConfigValidator",
        coverage_id="tributo.official.causal.treatment-coverage.v1",
    ),
)
LINEAR_DML_DESCRIPTOR = _descriptor(
    spec=_DML_SPEC,
    implementation_id="tributo.official.causal.linear_dml",
    implementation_class="LinearDMLATE",
    state_schema=_DML_STATE_SCHEMA,
    allowed_config_keys=(
        "confidence_z",
        "cross_fit_folds",
        "fold_column",
        "output",
        "policy_cost",
        "treatment_col",
    ),
    contract_bindings=_bindings(
        _DML_SPEC,
        config_digest="6",
        config_validator="ATEConfigValidator",
        coverage_id="tributo.official.causal.dml-treatment-coverage.v1",
    ),
)
LINEAR_IV_DESCRIPTOR = _descriptor(
    spec=_IV_SPEC,
    implementation_id="tributo.official.causal.linear_iv",
    implementation_class="LinearIVATE",
    state_schema=_IV_STATE_SCHEMA,
    allowed_config_keys=(
        "confidence_z",
        "instrument_col",
        "output",
        "policy_cost",
        "treatment_col",
    ),
    contract_bindings=_bindings(
        _IV_SPEC,
        config_digest="5",
        config_validator="IVConfigValidator",
        coverage_id="tributo.official.causal.iv-treatment-coverage.v1",
    ),
)


__all__ = ["ATE_DESCRIPTOR", "LINEAR_DML_DESCRIPTOR", "LINEAR_IV_DESCRIPTOR"]
