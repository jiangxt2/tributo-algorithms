"""Tabular Autoencoder TrainingRecipeV2 implementation."""

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


def _mse(predictions: object, targets: object) -> object:
    import torch

    return torch.mean((cast(Any, predictions) - cast(Any, targets)) ** 2)


class _CheckpointCodec:
    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


class TabularAutoencoderRecipe(TrainingRecipeV2):
    """Reconstruct dense inputs without requiring a label column."""

    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        import torch

        model = config.get("model", {})
        if not isinstance(model, Mapping):
            raise ValueError("model config must be a mapping")
        input_features = int(model.get("input_features", 4))
        latent_features = int(model.get("latent_features", 2))
        module = torch.nn.Sequential(
            torch.nn.Linear(input_features, latent_features),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_features, input_features),
        )
        return {"model": module, "loss": torch.nn.MSELoss()}

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

        del label_name, config
        if not isinstance(batch, Mapping):
            raise ValueError("Autoencoder batch must be columnar")
        features = torch.stack(
            [batch[name].to(dtype=torch.float32) for name in feature_names], dim=1
        )
        weights = batch.get(weight_name) if weight_name is not None else None
        return features, features, weights, int(features.shape[0])

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
        reconstruction = model(features)
        return TrainingStepResult(reconstruction, loss(reconstruction, targets))

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
        return MetricPlan(factories={"reconstruction_mse": _mse})

    def checkpoint_codec(self) -> object:
        return _CheckpointCodec()


__all__ = ["TabularAutoencoderRecipe"]
