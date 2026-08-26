"""Temporal convolution TrainingRecipeV2 implementation."""

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


def _model(input_features: int, channels: int) -> object:
    import torch

    class TemporalConvClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_features = input_features
            self.network = torch.nn.Sequential(
                torch.nn.Conv1d(1, channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
            )
            self.output = torch.nn.Linear(channels, 1)

        def forward(self, values: object) -> object:
            tensor = cast(Any, values)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(1)
            encoded = self.network(tensor).squeeze(-1)
            return self.output(encoded)

    return TemporalConvClassifier()


class TemporalConvRecipe(TrainingRecipeV2):
    """Define a finite-window temporal model without owning Ray/DDP code."""

    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        import torch

        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("model config must be a mapping")
        return {
            "model": _model(
                int(model_config.get("input_features", 4)),
                int(model_config.get("channels", 8)),
            ),
            "loss": torch.nn.BCEWithLogitsLoss(),
        }

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

        del config
        if not isinstance(batch, Mapping):
            raise ValueError("time-series batch must be columnar")
        windows = torch.stack(
            [batch[name].to(dtype=torch.float32) for name in feature_names], dim=1
        )
        if label_name is None:
            raise ValueError("temporal classifier requires a label")
        targets = batch[label_name].to(dtype=torch.float32).reshape(-1, 1)
        weights = batch.get(weight_name) if weight_name is not None else None
        return windows, targets, weights, int(windows.shape[0])

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

        return OptimizationPlan(
            optimizer=torch.optim.Adam(
                cast(Any, model).parameters(),
                lr=float(config.get("learning_rate", 0.01)),
            ),
            gradient_accumulation_steps=int(config.get("accumulation_steps", 1)),
            max_gradient_norm=float(config.get("max_gradient_norm", 1.0)),
        )

    def metric_plan(self, config: Mapping[str, Any]) -> MetricPlan:
        del config
        return MetricPlan(factories={"accuracy": _accuracy})

    def checkpoint_codec(self) -> object:
        return _CheckpointCodec()


__all__ = ["TemporalConvRecipe"]
