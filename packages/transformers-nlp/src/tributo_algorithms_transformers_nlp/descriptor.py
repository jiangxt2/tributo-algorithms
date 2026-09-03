"""Descriptor for the official pre-tokenized Transformer TorchRecipe."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MetricReduction,
    QualifiedReference,
    SingleStageTorchPlan,
    TorchDatasetRoute,
    TorchPolicy,
    TorchStageSpec,
    WorkerRange,
    WorkerResources,
)
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

from tributo_algorithms_transformers_nlp.contracts import (
    TokenTensorInputValidator,
    TransformerConfigValidator,
    TransformerOutputValidator,
    TransformerTorchCoverageValidator,
)

_PACKAGE = "tributo-algorithms-transformers-nlp"
_VERSION = "0.1.0"
_ROOT = Path(__file__).resolve().parent


def _code_digest(filename: str) -> str:
    return hashlib.sha256((_ROOT / filename).read_bytes()).hexdigest()


def _binding(
    contract_id: str, validator: type[Any], version: int = 2
) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=version,
        schema_digest=str(validator.schema_digest),
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_transformers_nlp.contracts:{validator.__name__}"
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
    allowed_execution_modes=(ExecutionMode.RAY_TRAIN_TORCH.value,),
    config_contract_ref="tributo.official.transformer.config.v1",
    input_contract_ref="tributo.official.transformer.tokens.v2",
    output_contract_ref="tributo.official.transformer.onnx.v1",
)

_POLICY = TorchPolicy(
    torch_runtime_api_version=1,
    loop_owner="core_recipe",
    parallelism_id="torch.ddp.replicated",
    dataset_routing=(
        TorchDatasetRoute("train", "split_exact", True, 1, 1, "reject"),
        TorchDatasetRoute("val", "split_exact", False, 1, 0, "zero_contribution"),
        TorchDatasetRoute("test", "split_exact", False, 1, 0, "zero_contribution"),
    ),
    execution_plan=SingleStageTorchPlan(
        stage=TorchStageSpec(
            "train",
            "tributo.integrations.algorithm_runtimes.ray_train_torch:torch_recipe_train_loop_per_worker",
            ("train", "val", "test"),
            metric_mapping={"accuracy": "accuracy", "train_loss": "train_loss"},
        )
    ),
    state_layout="replicated",
    metric_reducers={
        "accuracy": MetricReduction.SUM_COUNT,
        "train_loss": MetricReduction.SUM_COUNT,
    },
    backend="auto",
    resume_supported=True,
    same_world_size_resume=True,
)

_CONTRACTS = ContractBindingSet(
    config=_binding(_SPEC.config_contract_ref or "", TransformerConfigValidator, 1),
    input=_binding(_SPEC.input_contract_ref or "", TokenTensorInputValidator),
    output=_binding(_SPEC.output_contract_ref or "", TransformerOutputValidator, 1),
    coverage=_binding(
        "tributo.official.transformer.torch-coverage.v2",
        TransformerTorchCoverageValidator,
    ),
)

TOKEN_TRANSFORMER_DESCRIPTOR = AlgorithmBuilder.from_torch(
    spec=_SPEC,
    implementation_id="tributo.official.transformer.token_classifier",
    implementation_version="2.0.0",
    recipe="tributo_algorithms_transformers_nlp.recipe:TokenTransformerRecipe",
    environment=EnvironmentSpec(
        environment_id="tributo.official.transformer.v2",
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
    policy=_POLICY,
    code_digest=_code_digest("recipe.py"),
    contract_bindings=_CONTRACTS,
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["TOKEN_TRANSFORMER_DESCRIPTOR"]
