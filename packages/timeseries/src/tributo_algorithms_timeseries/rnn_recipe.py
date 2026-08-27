"""Fixed-window LSTM and GRU TrainingRecipeV2 implementations."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any, cast

from tributo.algorithms import (
    MetricPlan,
    OptimizationPlan,
    TrainingRecipeV2,
    TrainingStepResult,
)


def _accuracy(predictions: object, targets: object) -> object:
    import torch

    values = cast(Any, predictions)
    expected = cast(Any, targets)
    predicted = (torch.sigmoid(values) >= 0.5).to(dtype=expected.dtype)
    return (predicted == expected).to(dtype=torch.float32).mean()


class _CheckpointCodec:
    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


def _model(config: Mapping[str, Any], *, recurrent_kind: str) -> object:
    import torch

    input_features = int(config.get("input_features", 4))
    hidden_size = int(config.get("hidden_size", 16))
    num_layers = int(config.get("num_layers", 1))
    if input_features < 1 or hidden_size < 1 or num_layers < 1:
        raise ValueError("RNN dimensions must be positive")

    class RecurrentClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            recurrent = torch.nn.LSTM if recurrent_kind == "lstm" else torch.nn.GRU
            self.recurrent = recurrent(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, values: object) -> object:
            sequence = cast(Any, values)
            if sequence.ndim == 2:
                sequence = sequence.unsqueeze(-1)
            if sequence.ndim != 3:
                raise ValueError("RNN input must have shape [batch, window, features]")
            encoded, _ = self.recurrent(sequence)
            return self.output(encoded[:, -1, :])

    return RecurrentClassifier()


class _BaseRNNRecipe(TrainingRecipeV2):
    def batch_adapter(
        self,
        batch: object,
        *,
        feature_names: tuple[str, ...],
        label_name: str | None,
        weight_name: str | None,
        config: Mapping[str, Any],
    ) -> tuple[object, object, object | None, int]:
        import torch

        if not isinstance(batch, Mapping) or label_name is None:
            raise ValueError("RNN batch requires columnar features and a label")
        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("RNN model config must be a mapping")
        if int(model_config.get("input_features", len(feature_names))) != len(
            feature_names
        ):
            raise ValueError("RNN input_features must match the fixed window width")
        sequence = torch.stack(
            [batch[name].to(dtype=torch.float32) for name in feature_names], dim=1
        ).unsqueeze(-1)
        targets = batch[label_name].to(dtype=torch.float32).reshape(-1, 1)
        if not bool(((targets == 0) | (targets == 1)).all()):
            raise ValueError("RNN classifier labels must be 0 or 1")
        weights = batch.get(weight_name) if weight_name is not None else None
        return sequence, targets, weights, int(sequence.shape[0])

    def training_step(
        self,
        modules: Mapping[str, object],
        features: object,
        targets: object,
        weights: object | None,
        config: Mapping[str, Any],
    ) -> TrainingStepResult:
        del weights, config
        model = cast(Any, modules["model"])
        loss = cast(Any, modules["loss"])
        predictions = model(features)
        return TrainingStepResult(predictions, loss(predictions, targets))

    def validation_step(
        self,
        modules: Mapping[str, object],
        features: object,
        targets: object,
        weights: object | None,
        config: Mapping[str, Any],
    ) -> TrainingStepResult:
        return self.training_step(modules, features, targets, weights, config)

    def optimization_plan(
        self,
        model: object,
        config: Mapping[str, Any],
    ) -> OptimizationPlan:
        import torch

        optimizer_config = config.get("optimizer", {})
        if not isinstance(optimizer_config, Mapping):
            raise ValueError("RNN optimizer config must be a mapping")
        learning_rate = float(optimizer_config.get("learning_rate", 0.001))
        weight_decay = float(optimizer_config.get("weight_decay", 0.0))
        accumulation_steps = int(optimizer_config.get("accumulation_steps", 1))
        max_gradient_norm = float(optimizer_config.get("max_gradient_norm", 1.0))
        if learning_rate <= 0 or weight_decay < 0 or accumulation_steps < 1:
            raise ValueError("RNN optimizer parameters are invalid")
        if max_gradient_norm <= 0:
            raise ValueError("RNN max_gradient_norm must be positive")
        return OptimizationPlan(
            optimizer=torch.optim.Adam(
                cast(Any, model).parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            ),
            gradient_accumulation_steps=accumulation_steps,
            max_gradient_norm=max_gradient_norm,
        )

    def metric_plan(self, config: Mapping[str, Any]) -> MetricPlan:
        del config
        return MetricPlan(factories={"accuracy": _accuracy})

    def checkpoint_codec(self) -> object:
        return _CheckpointCodec()


class LSTMRecipe(_BaseRNNRecipe):
    """Train a fixed-window LSTM binary classifier."""

    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        import torch

        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("LSTM model config must be a mapping")
        return {
            "model": _model(model_config, recurrent_kind="lstm"),
            "loss": torch.nn.BCEWithLogitsLoss(),
        }


class GRURecipe(_BaseRNNRecipe):
    """Train a fixed-window GRU binary classifier."""

    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        import torch

        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("GRU model config must be a mapping")
        return {
            "model": _model(model_config, recurrent_kind="gru"),
            "loss": torch.nn.BCEWithLogitsLoss(),
        }


__all__ = ["GRURecipe", "LSTMRecipe"]
