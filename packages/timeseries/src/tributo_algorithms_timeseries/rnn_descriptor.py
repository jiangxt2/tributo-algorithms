"""Descriptors for official fixed-window LSTM and GRU algorithms."""

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

_PACKAGE = "tributo-algorithms-timeseries"
_VERSION = "0.1.0"
_INPUT = "tributo.official.timeseries.fixed_window.v1"
_COVERAGE = "tributo.official.timeseries.window.coverage.v1"


def _binding(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_timeseries.rnn_contracts:{validator}"
        ),
    )


def _spec(name: str, output: str) -> AlgorithmSpec:
    return AlgorithmSpec(
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
        allowed_execution_modes=(ExecutionMode.TRAINING_RECIPE_V2.value,),
        config_contract_ref=f"tributo.official.timeseries.{name}.config.v1",
        input_contract_ref=_INPUT,
        output_contract_ref=output,
    )


def _descriptor(spec: AlgorithmSpec, implementation_id: str, recipe: str) -> object:
    return AlgorithmBuilder.from_training_recipe_v2(
        spec=spec,
        implementation_id=implementation_id,
        implementation_version="1.0.0",
        recipe=recipe,
        environment=EnvironmentSpec(
            environment_id="tributo.official.timeseries.rnn.v1",
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
        contract_bindings=ContractBindingSet(
            config=_binding(spec.config_contract_ref or "", "9", "RNNConfigValidator"),
            input=_binding(_INPUT, "a", "FixedWindowInputValidator"),
            output=_binding(spec.output_contract_ref or "", "b", "RNNOutputValidator"),
            coverage=_binding(_COVERAGE, "c", "RNNCoverageValidator"),
        ),
        descriptor_api_version=2,
        is_default=True,
    )


_LSTM_SPEC = _spec("lstm_classifier", "tributo.official.timeseries.lstm.onnx.v1")
_GRU_SPEC = _spec("gru_classifier", "tributo.official.timeseries.gru.onnx.v1")

LSTM_DESCRIPTOR = _descriptor(
    _LSTM_SPEC,
    "tributo.official.timeseries.lstm.recipe_v2",
    "tributo_algorithms_timeseries.rnn_recipe:LSTMRecipe",
)
GRU_DESCRIPTOR = _descriptor(
    _GRU_SPEC,
    "tributo.official.timeseries.gru.recipe_v2",
    "tributo_algorithms_timeseries.rnn_recipe:GRURecipe",
)

__all__ = ["GRU_DESCRIPTOR", "LSTM_DESCRIPTOR"]
