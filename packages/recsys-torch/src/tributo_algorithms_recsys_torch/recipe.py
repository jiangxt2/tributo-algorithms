"""Two-Tower TrainingRecipeV2 implementation."""

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


class _CheckpointCodec:
    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


def _pair_accuracy(predictions: object, targets: object) -> object:
    import torch

    logits = cast(Any, predictions)
    labels = cast(Any, targets)
    predicted = (torch.sigmoid(logits) >= 0.5).to(dtype=labels.dtype)
    return (predicted == labels).to(torch.float32).mean()


def _model(config: Mapping[str, Any]) -> object:
    import torch

    user_count = int(config.get("user_count", 1))
    item_count = int(config.get("item_count", 1))
    embedding_dim = int(config.get("embedding_dim", 16))
    if min(user_count, item_count, embedding_dim) < 1:
        raise ValueError("Two-Tower dimensions must be positive")

    class TwoTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.user_embedding = torch.nn.Embedding(user_count, embedding_dim)
            self.item_embedding = torch.nn.Embedding(item_count, embedding_dim)
            self.user_bias = torch.nn.Embedding(user_count, 1)
            self.item_bias = torch.nn.Embedding(item_count, 1)

        def forward(self, pairs: object) -> object:
            values = cast(Any, pairs).long()
            users = values[:, 0]
            items = values[:, 1]
            score = (self.user_embedding(users) * self.item_embedding(items)).sum(
                dim=1, keepdim=True
            )
            return score + self.user_bias(users) + self.item_bias(items)

    return TwoTower()


class TwoTowerRecipe(TrainingRecipeV2):
    """Train user/item embeddings from labeled interaction pairs."""

    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        import torch

        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("Two-Tower model config must be a mapping")
        return {"model": _model(model_config), "loss": torch.nn.BCEWithLogitsLoss()}

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
        if not isinstance(batch, Mapping) or len(feature_names) != 2:
            raise ValueError("Two-Tower requires user and item ID columns")
        if label_name is None:
            raise ValueError("Two-Tower requires an interaction label")
        pairs = torch.stack(
            [batch[name].to(dtype=torch.long) for name in feature_names], dim=1
        )
        labels = batch[label_name].to(dtype=torch.float32).reshape(-1, 1)
        weights = batch.get(weight_name) if weight_name is not None else None
        return pairs, labels, weights, int(pairs.shape[0])

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
        labels = cast(Any, targets)
        return TrainingStepResult(
            predictions,
            loss(predictions, labels),
            coverage_counts={
                "positive_pairs": int((labels == 1).sum()),
                "negative_pairs": int((labels == 0).sum()),
            },
        )

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
                weight_decay=float(config.get("weight_decay", 0.0)),
            ),
            gradient_accumulation_steps=int(config.get("accumulation_steps", 1)),
            max_gradient_norm=float(config.get("max_gradient_norm", 1.0)),
        )

    def metric_plan(self, config: Mapping[str, Any]) -> MetricPlan:
        del config
        return MetricPlan(factories={"pair_accuracy": _pair_accuracy})

    def checkpoint_codec(self) -> object:
        return _CheckpointCodec()


__all__ = ["TwoTowerRecipe"]
