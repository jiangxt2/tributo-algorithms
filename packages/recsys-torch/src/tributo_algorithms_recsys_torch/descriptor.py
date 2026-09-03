"""Descriptor for the official Two-Tower TorchRecipe recommender."""

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

from tributo_algorithms_recsys_torch.contracts import (
    TwoTowerConfigValidator,
    TwoTowerOutputValidator,
    TwoTowerTensorInputValidator,
    TwoTowerTorchCoverageValidator,
)

_PACKAGE = "tributo-algorithms-recsys-torch"
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
            f"tributo_algorithms_recsys_torch.contracts:{validator.__name__}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="two_tower_recommender",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.RANKING,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="supervised_pairwise",
    model_family="two_tower_embedding",
    data_modalities=("interaction",),
    lifecycle_kind="batch_fit",
    allowed_execution_modes=(ExecutionMode.RAY_TRAIN_TORCH.value,),
    config_contract_ref="tributo.official.two-tower.config.v1",
    input_contract_ref="tributo.official.two-tower.pairs.v2",
    output_contract_ref="tributo.official.two-tower.onnx.v1",
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
            metric_mapping={
                "pair_accuracy": "pair_accuracy",
                "train_loss": "train_loss",
            },
        )
    ),
    state_layout="replicated",
    metric_reducers={
        "pair_accuracy": MetricReduction.SUM_COUNT,
        "train_loss": MetricReduction.SUM_COUNT,
    },
    backend="auto",
    resume_supported=True,
    same_world_size_resume=True,
)

_CONTRACTS = ContractBindingSet(
    config=_binding(_SPEC.config_contract_ref or "", TwoTowerConfigValidator, 1),
    input=_binding(_SPEC.input_contract_ref or "", TwoTowerTensorInputValidator),
    output=_binding(_SPEC.output_contract_ref or "", TwoTowerOutputValidator, 1),
    coverage=_binding(
        "tributo.official.two-tower.torch-coverage.v2", TwoTowerTorchCoverageValidator
    ),
)

TWO_TOWER_DESCRIPTOR = AlgorithmBuilder.from_torch(
    spec=_SPEC,
    implementation_id="tributo.official.recsys_torch.two_tower",
    implementation_version="2.0.0",
    recipe="tributo_algorithms_recsys_torch.recipe:TwoTowerRecipe",
    environment=EnvironmentSpec(
        environment_id="tributo.official.recsys-torch.v2",
        dependencies=(
            "onnx>=1.16",
            "onnxruntime>=1.20",
            "torch>=2.5",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    metric_reducers={"pair_accuracy": MetricReduction.SUM_COUNT},
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

__all__ = ["TWO_TOWER_DESCRIPTOR"]
