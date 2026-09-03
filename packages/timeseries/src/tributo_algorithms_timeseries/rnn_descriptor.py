"""Descriptors for fixed-window LSTM and GRU TorchRecipe algorithms."""

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

from tributo_algorithms_timeseries.rnn_contracts import (
    GRUTensorInputValidator,
    GRUTorchCoverageValidator,
    LSTMTensorInputValidator,
    LSTMTorchCoverageValidator,
    RNNConfigValidator,
    RNNOutputValidator,
)

_PACKAGE = "tributo-algorithms-timeseries"
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
            f"tributo_algorithms_timeseries.rnn_contracts:{validator.__name__}"
        ),
    )


def _descriptor(
    *,
    name: str,
    implementation_id: str,
    recipe: str,
    output_contract: str,
    input_validator: type[object],
    coverage_validator: type[object],
) -> object:
    spec = AlgorithmSpec(
        name=name,
        trainer_cls=None,
        version="1.0.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(ProblemType.BINARY_CLASSIFICATION,),
        capabilities=(
            Capability.DISTRIBUTED,
            Capability.EXPORTABLE,
            Capability.TUNABLE,
        ),
        learning_paradigm="supervised",
        model_family="recurrent_neural_network",
        data_modalities=("time_series",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(ExecutionMode.RAY_TRAIN_TORCH.value,),
        config_contract_ref=f"tributo.official.timeseries.{name}.config.v1",
        input_contract_ref=f"tributo.official.timeseries.{name.removesuffix('_classifier')}.window-tensor.v2",
        output_contract_ref=output_contract,
    )
    policy = TorchPolicy(
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
    contracts = ContractBindingSet(
        config=_binding(spec.config_contract_ref or "", RNNConfigValidator, 1),
        input=_binding(spec.input_contract_ref or "", input_validator),
        output=_binding(output_contract, RNNOutputValidator, 1),
        coverage=_binding(
            f"tributo.official.timeseries.{name}.torch-coverage.v2",
            coverage_validator,
        ),
    )
    return AlgorithmBuilder.from_torch(
        spec=spec,
        implementation_id=implementation_id,
        implementation_version="2.0.0",
        recipe=recipe,
        environment=EnvironmentSpec(
            environment_id="tributo.official.timeseries.rnn.v2",
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
        policy=policy,
        code_digest=_code_digest("rnn_recipe.py"),
        contract_bindings=contracts,
        descriptor_api_version=2,
        is_default=True,
    )


LSTM_DESCRIPTOR = _descriptor(
    name="lstm_classifier",
    implementation_id="tributo.official.timeseries.lstm",
    recipe="tributo_algorithms_timeseries.rnn_recipe:LSTMRecipe",
    output_contract="tributo.official.timeseries.lstm.onnx.v1",
    input_validator=LSTMTensorInputValidator,
    coverage_validator=LSTMTorchCoverageValidator,
)
GRU_DESCRIPTOR = _descriptor(
    name="gru_classifier",
    implementation_id="tributo.official.timeseries.gru",
    recipe="tributo_algorithms_timeseries.rnn_recipe:GRURecipe",
    output_contract="tributo.official.timeseries.gru.onnx.v1",
    input_validator=GRUTensorInputValidator,
    coverage_validator=GRUTorchCoverageValidator,
)

__all__ = ["GRU_DESCRIPTOR", "LSTM_DESCRIPTOR"]
