"""Descriptor for the official bounded Transformer classifier."""

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
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

_PACKAGE = "tributo-algorithms-transformers-nlp"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_transformers_nlp.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="token_transformer_classifier",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.BINARY_CLASSIFICATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="supervised",
    model_family="transformer_encoder",
    data_modalities=("text",),
    lifecycle_kind="batch_fit",
    allowed_execution_modes=(ExecutionMode.TRAINING_RECIPE_V2.value,),
    config_contract_ref="tributo.official.transformer.config.v1",
    input_contract_ref="tributo.official.transformer.tokens.v1",
    output_contract_ref="tributo.official.transformer.onnx.v1",
)

_CONTRACTS = ContractBindingSet(
    config=_binding(_SPEC.config_contract_ref or "", "d", "TransformerConfigValidator"),
    input=_binding(_SPEC.input_contract_ref or "", "e", "TokenInputValidator"),
    output=_binding(_SPEC.output_contract_ref or "", "f", "TransformerOutputValidator"),
    coverage=_binding(
        "tributo.official.transformer.token.coverage.v1",
        "0",
        "TokenCoverageValidator",
    ),
)

TOKEN_TRANSFORMER_DESCRIPTOR = AlgorithmBuilder.from_training_recipe_v2(
    spec=_SPEC,
    implementation_id="tributo.official.transformer.token_classifier",
    implementation_version="1.0.0",
    recipe=("tributo_algorithms_transformers_nlp.recipe:TokenTransformerRecipe"),
    environment=EnvironmentSpec(
        environment_id="tributo.official.transformer.v1",
        dependencies=(
            "onnx>=1.16",
            "onnxruntime>=1.20",
            "torch>=2.5",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    metric_reducers={"accuracy": MetricReduction.SUM_COUNT},
    supported_worker_range=WorkerRange(1, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    contract_bindings=_CONTRACTS,
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["TOKEN_TRANSFORMER_DESCRIPTOR"]
