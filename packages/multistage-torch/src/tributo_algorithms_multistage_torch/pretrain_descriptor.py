"""Descriptor for the Core-orchestrated pretrain/finetune Component plan."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ComponentStageTorchPlan,
    ContractBinding,
    ContractBindingSet,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MetricReduction,
    QualifiedReference,
    TorchDatasetRoute,
    TorchPolicy,
    TorchStageSpec,
    WorkerRange,
    WorkerResources,
)
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

from tributo_algorithms_multistage_torch.contracts import (
    PretrainFinetuneConfigValidator,
    PretrainFinetuneOutputValidator,
    PretrainFinetuneTorchCoverageValidator,
    PretrainFinetuneTorchInputValidator,
)

STAGES = ("pretrain", "finetune")
_PACKAGE = "tributo-algorithms-multistage-torch"
_VERSION = "0.1.0"
_ROOT = Path(__file__).resolve().parent


def _code_digest() -> str:
    return hashlib.sha256((_ROOT / "pretrain.py").read_bytes()).hexdigest()


def _binding(contract_id: str, validator: type[Any], version: int) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=version,
        schema_digest=str(validator.schema_digest),
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_multistage_torch.contracts:{validator.__name__}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="pretrain_finetune_classifier",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.BINARY_CLASSIFICATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="self_supervised_pretrain_then_supervised_finetune",
    model_family="encoder_classifier",
    data_modalities=("tabular",),
    lifecycle_kind="finite_multistage_fit",
    allowed_execution_modes=(ExecutionMode.RAY_TRAIN_TORCH.value,),
    config_contract_ref="tributo.official.pretrain-finetune.config.v1",
    input_contract_ref="tributo.official.pretrain-finetune.dense-labeled.v2",
    output_contract_ref="tributo.official.pretrain-finetune.bundle.v1",
)

_POLICY = TorchPolicy(
    torch_runtime_api_version=1,
    loop_owner="adapter",
    parallelism_id="torch.ddp.replicated",
    dataset_routing=(TorchDatasetRoute("train", "split_exact", True, 1, 1, "reject"),),
    execution_plan=ComponentStageTorchPlan(
        stages=(
            TorchStageSpec(
                "pretrain",
                "tributo.integrations.algorithm_runtimes.ray_train_torch:ray_torch_adapter_train_loop_per_worker",
                ("train",),
                metric_mapping={
                    "pretrain_loss": "pretrain_loss",
                    "train_loss": "train_loss",
                },
            ),
            TorchStageSpec(
                "finetune",
                "tributo.integrations.algorithm_runtimes.ray_train_torch:ray_torch_adapter_train_loop_per_worker",
                ("train",),
                depends_on=("pretrain",),
                checkpoint_from_stage="pretrain",
                metric_mapping={
                    "finetune_loss": "finetune_loss",
                    "train_loss": "train_loss",
                },
            ),
        ),
        final_stage_id="finetune",
    ),
    state_layout="component",
    metric_reducers={
        "pretrain_loss": MetricReduction.SUM_COUNT,
        "finetune_loss": MetricReduction.SUM_COUNT,
        "train_loss": MetricReduction.SUM_COUNT,
    },
    backend="auto",
    resume_supported=False,
    same_world_size_resume=None,
)

_CONTRACTS = ContractBindingSet(
    config=_binding(
        _SPEC.config_contract_ref or "", PretrainFinetuneConfigValidator, 1
    ),
    input=_binding(
        _SPEC.input_contract_ref or "", PretrainFinetuneTorchInputValidator, 2
    ),
    output=_binding(
        _SPEC.output_contract_ref or "", PretrainFinetuneOutputValidator, 1
    ),
    coverage=_binding(
        "tributo.official.pretrain-finetune.torch-component-coverage.v2",
        PretrainFinetuneTorchCoverageValidator,
        2,
    ),
)

PRETRAIN_FINETUNE_DESCRIPTOR = AlgorithmBuilder.from_torch_adapter(
    spec=_SPEC,
    implementation_id="tributo.official.multistage_torch.pretrain_finetune",
    implementation_version="2.0.0",
    adapter="tributo_algorithms_multistage_torch.pretrain:DistributedPretrainFinetune",
    environment=EnvironmentSpec(
        environment_id="tributo.official.pretrain-finetune.v2",
        dependencies=(
            "onnx>=1.16",
            "onnxruntime>=1.20",
            "safetensors>=0.4.3",
            "torch>=2.5",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    metric_reducers={
        "pretrain_loss": MetricReduction.SUM_COUNT,
        "finetune_loss": MetricReduction.SUM_COUNT,
        "train_loss": MetricReduction.SUM_COUNT,
    },
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

__all__ = ["PRETRAIN_FINETUNE_DESCRIPTOR", "STAGES"]
