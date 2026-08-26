"""Descriptor for distributed self-supervised pretraining and finetuning."""

from __future__ import annotations

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    ContractBinding,
    ContractBindingSet,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    FrameworkNativePolicy,
    QualifiedReference,
    WorkerRange,
    WorkerResources,
)
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

STAGES = ("pretrain", "finetune")

_PACKAGE = "tributo-algorithms-multistage-torch"
_VERSION = "0.1.0"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_multistage_torch.contracts:{validator}"
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
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.official.pretrain-finetune.config.v1",
    input_contract_ref="tributo.official.pretrain-finetune.dense-labeled.v1",
    output_contract_ref="tributo.official.pretrain-finetune.bundle.v1",
)

PRETRAIN_FINETUNE_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.multistage_torch.pretrain_finetune",
    implementation_version="1.0.0",
    implementation=(
        "tributo_algorithms_multistage_torch.pretrain:DistributedPretrainFinetune"
    ),
    executable_factory=(
        "tributo_algorithms_multistage_torch.pretrain:"
        "create_pretrain_finetune_algorithm"
    ),
    distribution=_PACKAGE,
    framework="pytorch",
    environment=EnvironmentSpec(
        environment_id="tributo.official.pretrain-finetune.v1",
        dependencies=(
            "safetensors>=0.4.3",
            "torch>=2.5",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=("model", "output", "ray", "training"),
    strategy=DistributionStrategy.FRAMEWORK_NATIVE,
    supported_worker_range=WorkerRange(2, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=FrameworkNativePolicy(
        framework="pytorch-ddp-pretrain-finetune",
        evidence_collector_ref=(
            "tributo_algorithms_multistage_torch.pretrain:"
            "DistributedPretrainFinetune.collect_evidence"
        ),
        component_stages=STAGES,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter=(
        "tributo_algorithms_multistage_torch.pretrain:export_pretrain_finetune_result"
    ),
    flavor_id="safetensors-v1",
    contract_bindings=ContractBindingSet(
        config=_binding(
            _SPEC.config_contract_ref or "", "e", "PretrainFinetuneConfigValidator"
        ),
        input=_binding(
            _SPEC.input_contract_ref or "", "f", "PretrainFinetuneInputValidator"
        ),
        output=_binding(
            _SPEC.output_contract_ref or "", "0", "PretrainFinetuneOutputValidator"
        ),
        coverage=_binding(
            "tributo.official.pretrain-finetune.stage-coverage.v1",
            "1",
            "PretrainFinetuneCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["PRETRAIN_FINETUNE_DESCRIPTOR"]
