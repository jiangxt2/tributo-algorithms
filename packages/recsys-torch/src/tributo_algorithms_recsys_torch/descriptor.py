"""Descriptor for the official distributed Two-Tower recommender."""

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

_PACKAGE = "tributo-algorithms-recsys-torch"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_recsys_torch.contracts:{validator}"
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
    allowed_execution_modes=(ExecutionMode.TRAINING_RECIPE_V2.value,),
    config_contract_ref="tributo.official.two-tower.config.v1",
    input_contract_ref="tributo.official.interaction.pairs.v1",
    output_contract_ref="tributo.official.two-tower.onnx.v1",
)

TWO_TOWER_DESCRIPTOR = AlgorithmBuilder.from_training_recipe_v2(
    spec=_SPEC,
    implementation_id="tributo.official.recsys_torch.two_tower",
    implementation_version="1.0.0",
    recipe="tributo_algorithms_recsys_torch.recipe:TwoTowerRecipe",
    environment=EnvironmentSpec(
        environment_id="tributo.official.recsys-torch.v1",
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
    contract_bindings=ContractBindingSet(
        config=_binding(
            _SPEC.config_contract_ref or "", "b", "TwoTowerConfigValidator"
        ),
        input=_binding(_SPEC.input_contract_ref or "", "c", "PairInputValidator"),
        output=_binding(
            _SPEC.output_contract_ref or "", "d", "TwoTowerOutputValidator"
        ),
        coverage=_binding(
            "tributo.official.interaction.pair-coverage.v1",
            "e",
            "PairCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["TWO_TOWER_DESCRIPTOR"]
