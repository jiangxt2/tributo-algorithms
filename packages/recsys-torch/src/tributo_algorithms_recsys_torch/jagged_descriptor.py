"""Descriptor for the jagged-history RayTorchAdapter."""

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
    JaggedConfigValidator,
    JaggedOutputValidator,
    JaggedTorchCoverageValidator,
    JaggedTorchInputValidator,
)

_PACKAGE = "tributo-algorithms-recsys-torch"
_VERSION = "0.1.0"
_ROOT = Path(__file__).resolve().parent


def _code_digest() -> str:
    return hashlib.sha256((_ROOT / "jagged.py").read_bytes()).hexdigest()


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
    name="jagged_embedding_recommender",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.RANKING,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="supervised_implicit_feedback",
    model_family="jagged_embedding_ddp_alltoall_routing",
    data_modalities=("interaction", "jagged_sequence"),
    lifecycle_kind="batch_fit",
    allowed_execution_modes=(ExecutionMode.RAY_TRAIN_TORCH.value,),
    config_contract_ref="tributo.official.jagged-recsys.config.v2",
    input_contract_ref="tributo.official.jagged-recsys.interactions.v2",
    output_contract_ref="tributo.official.jagged-recsys.bundle.v1",
)

_POLICY = TorchPolicy(
    torch_runtime_api_version=1,
    loop_owner="adapter",
    parallelism_id="torch.ddp.replicated",
    dataset_routing=(TorchDatasetRoute("train", "split_exact", True, 1, 1, "reject"),),
    execution_plan=SingleStageTorchPlan(
        stage=TorchStageSpec(
            "train",
            "tributo.integrations.algorithm_runtimes.ray_train_torch:ray_torch_adapter_train_loop_per_worker",
            ("train",),
            metric_mapping={"train_loss": "train_loss"},
        )
    ),
    state_layout="replicated",
    metric_reducers={"train_loss": MetricReduction.SUM_COUNT},
    backend="auto",
    resume_supported=False,
    same_world_size_resume=None,
)

_CONTRACTS = ContractBindingSet(
    config=_binding(_SPEC.config_contract_ref or "", JaggedConfigValidator, 1),
    input=_binding(_SPEC.input_contract_ref or "", JaggedTorchInputValidator),
    output=_binding(_SPEC.output_contract_ref or "", JaggedOutputValidator, 1),
    coverage=_binding(
        "tributo.official.jagged-recsys.torch-coverage.v2", JaggedTorchCoverageValidator
    ),
)

JAGGED_DESCRIPTOR = AlgorithmBuilder.from_torch_adapter(
    spec=_SPEC,
    implementation_id="tributo.official.recsys_torch.jagged_embedding",
    implementation_version="2.0.0",
    adapter="tributo_algorithms_recsys_torch.jagged:DistributedJaggedEmbedding",
    environment=EnvironmentSpec(
        environment_id="tributo.official.recsys-torch-jagged.v2",
        dependencies=(
            "onnx>=1.16",
            "onnxruntime>=1.20",
            "safetensors>=0.4.3",
            "torch>=2.5",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    metric_reducers={"train_loss": MetricReduction.SUM_COUNT},
    supported_worker_range=WorkerRange(2, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    policy=_POLICY,
    code_digest=_code_digest(),
    contract_bindings=_CONTRACTS,
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["JAGGED_DESCRIPTOR"]
