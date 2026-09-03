"""Descriptors for official dense DNN and PU TorchRecipe algorithms."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

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
from tributo.training.algorithm_spec import (
    AlgorithmSpec,
    Capability,
    DataLoadingMode,
    ProblemType,
    ResourceHints,
)

from tributo_algorithms_tabular_torch.contracts import (
    DNNTensorInputValidator,
    DNNTorchBundleOutputValidator,
    DNNTorchCoverageValidator,
    PUConfigValidator,
    PUCoverageValidator,
    PUTensorInputValidator,
    PUTorchBundleOutputValidator,
    TabularTorchConfigValidator,
)

_PACKAGE = "tributo-algorithms-tabular-torch"
_VERSION = "0.1.0"
_ROOT = Path(__file__).resolve().parent


def _code_digest(filename: str) -> str:
    return hashlib.sha256((_ROOT / filename).read_bytes()).hexdigest()


def _binding(
    contract_id: str, validator: type[Any], *, version: int | None = None
) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=version or 2,
        schema_digest=str(validator.schema_digest),
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_tabular_torch.contracts:{validator.__name__}"
        ),
    )


def _spec(name: str, *, pu: bool) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        resource_hints=ResourceHints(gpu_required=False),
        extras_group="identity",
        problem_types=(
            (ProblemType.PU_LEARNING,) if pu else (ProblemType.BINARY_CLASSIFICATION,)
        ),
        data_loading=DataLoadingMode.CANONICAL_TRAINER,
        learning_paradigm="positive_unlabeled" if pu else "supervised",
        model_family="deep_neural_network",
        data_modalities=("tabular",),
        lifecycle_kind="bounded_training",
        capabilities=(
            Capability.TUNABLE,
            Capability.EXPORTABLE,
            Capability.DISTRIBUTED,
        ),
        allowed_execution_modes=(ExecutionMode.RAY_TRAIN_TORCH.value,),
        config_contract_ref=f"tributo.algorithm-config.{name}.v1",
        input_contract_ref=(
            "tributo.official.pu.named-tensor.v2"
            if pu
            else "tributo.official.dnn.named-tensor.v2"
        ),
        output_contract_ref=(f"tributo.official.{name}.onnx-bundle.v2"),
    )


def _policy(*, pu: bool) -> TorchPolicy:
    routes = (
        TorchDatasetRoute("train", "split_exact", True, 1, 1, "reject"),
        TorchDatasetRoute("val", "split_exact", False, 1, 0, "zero_contribution"),
        TorchDatasetRoute("test", "split_exact", False, 1, 0, "zero_contribution"),
    )
    metrics = (
        {"observed_positive_recall": MetricReduction.SUM_COUNT}
        if pu
        else {"accuracy": MetricReduction.SUM_COUNT}
    )
    metrics["train_loss"] = MetricReduction.SUM_COUNT
    policy_kwargs: dict[str, object] = {
        "torch_runtime_api_version": 1,
        "loop_owner": "core_recipe",
        "parallelism_id": "torch.ddp.replicated",
        "dataset_routing": routes,
        "execution_plan": SingleStageTorchPlan(
            stage=TorchStageSpec(
                "train",
                "tributo.integrations.algorithm_runtimes.ray_train_torch:torch_recipe_train_loop_per_worker",
                ("train", "val", "test"),
                metric_mapping={name: name for name in metrics},
            )
        ),
        "state_layout": "replicated",
        "metric_reducers": metrics,
        "backend": "auto",
        "resume_supported": True,
        "same_world_size_resume": True,
    }
    if pu:
        policy_kwargs.update(
            {
                "global_loss_reducer_ref": "tributo_algorithms_tabular_torch.pu_reducer:PUGlobalLossReducer",
                "global_loss_reducer_api_version": 1,
                "global_loss_reducer_code_digest": _code_digest("pu_reducer.py"),
                "composite_loss_schema_id": "tributo.official.tabular_torch.pu-risk-components.v1",
            }
        )
    return cast(TorchPolicy, cast(Any, TorchPolicy)(**policy_kwargs))


def _contracts(*, pu: bool) -> ContractBindingSet:
    if pu:
        return ContractBindingSet(
            config=_binding(
                "tributo.algorithm-config.pu.v1", PUConfigValidator, version=1
            ),
            input=_binding(
                "tributo.official.pu.named-tensor.v2", PUTensorInputValidator
            ),
            output=_binding(
                "tributo.official.pu.onnx-bundle.v2", PUTorchBundleOutputValidator
            ),
            coverage=_binding(
                "tributo.official.pu.torch-coverage.v2", PUCoverageValidator
            ),
        )
    return ContractBindingSet(
        config=_binding(
            "tributo.algorithm-config.dnn.v1", TabularTorchConfigValidator, version=1
        ),
        input=_binding("tributo.official.dnn.named-tensor.v2", DNNTensorInputValidator),
        output=_binding(
            "tributo.official.dnn.onnx-bundle.v2", DNNTorchBundleOutputValidator
        ),
        coverage=_binding(
            "tributo.official.dnn.torch-coverage.v2", DNNTorchCoverageValidator
        ),
    )


_ENVIRONMENT = EnvironmentSpec(
    environment_id="tributo.official.tabular-torch.v2",
    dependencies=(
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "torch>=2.5",
        "tributo>=1,<2",
        f"{_PACKAGE}=={_VERSION}",
    ),
)

_DNN_SPEC = _spec("dnn", pu=False)
_PU_SPEC = _spec("pu", pu=True)

DNN_DESCRIPTOR = AlgorithmBuilder.from_torch(
    spec=_DNN_SPEC,
    implementation_id="tributo.official.tabular_torch.dnn",
    implementation_version="2.0.0",
    recipe="tributo_algorithms_tabular_torch.recipe:DNNRecipe",
    environment=_ENVIRONMENT,
    metric_reducers={"accuracy": MetricReduction.SUM_COUNT},
    supported_worker_range=WorkerRange(1, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    policy=_policy(pu=False),
    code_digest=_code_digest("recipe.py"),
    contract_bindings=_contracts(pu=False),
    descriptor_api_version=2,
    is_default=True,
)

PU_DESCRIPTOR = AlgorithmBuilder.from_torch(
    spec=_PU_SPEC,
    implementation_id="tributo.official.tabular_torch.pu",
    implementation_version="2.0.0",
    recipe="tributo_algorithms_tabular_torch.recipe:PURecipe",
    environment=_ENVIRONMENT,
    metric_reducers={"observed_positive_recall": MetricReduction.SUM_COUNT},
    supported_worker_range=WorkerRange(1, 64),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    policy=_policy(pu=True),
    code_digest=_code_digest("recipe.py"),
    contract_bindings=_contracts(pu=True),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = ["DNN_DESCRIPTOR", "PU_DESCRIPTOR"]
