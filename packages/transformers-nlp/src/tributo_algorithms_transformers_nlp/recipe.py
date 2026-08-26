"""Pre-tokenized Transformer classifier TrainingRecipeV2."""

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


def _transformer_model(
    *, vocab_size: int, sequence_length: int, hidden_size: int, heads: int
) -> object:
    import torch

    class TokenTransformerClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sequence_length = sequence_length
            self.embedding = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 2,
                batch_first=True,
                dropout=0.0,
            )
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, values: object) -> object:
            tokens = cast(Any, values).long()
            if tokens.ndim != 2:
                tokens = tokens.reshape(tokens.shape[0], self.sequence_length)
            mask = tokens.eq(0)
            encoded = self.encoder(self.embedding(tokens), src_key_padding_mask=mask)
            valid = (~mask).to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            return self.output(pooled)

    return TokenTransformerClassifier()


class TokenTransformerRecipe(TrainingRecipeV2):
    """Train a bounded Transformer over explicit pre-tokenized columns."""

    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        import torch

        model = config.get("model", {})
        if not isinstance(model, Mapping):
            raise ValueError("model config must be a mapping")
        return {
            "model": _transformer_model(
                vocab_size=int(model.get("vocab_size", 128)),
                sequence_length=int(model.get("sequence_length", 8)),
                hidden_size=int(model.get("hidden_size", 16)),
                heads=int(model.get("heads", 2)),
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
        if not isinstance(batch, Mapping) or label_name is None:
            raise ValueError(
                "Transformer classification requires token columns and label"
            )
        tokens = torch.stack([batch[name].long() for name in feature_names], dim=1)
        targets = batch[label_name].to(dtype=torch.float32).reshape(-1, 1)
        weights = batch.get(weight_name) if weight_name is not None else None
        return tokens, targets, weights, int(tokens.shape[0])

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
        self, model: object, config: Mapping[str, Any]
    ) -> OptimizationPlan:
        import torch

        return OptimizationPlan(
            optimizer=torch.optim.AdamW(
                cast(Any, model).parameters(),
                lr=float(config.get("learning_rate", 0.001)),
            ),
            gradient_accumulation_steps=int(config.get("accumulation_steps", 1)),
            max_gradient_norm=float(config.get("max_gradient_norm", 1.0)),
        )

    def metric_plan(self, config: Mapping[str, Any]) -> MetricPlan:
        del config
        return MetricPlan(factories={"accuracy": _accuracy})

    def checkpoint_codec(self) -> object:
        return _CheckpointCodec()


__all__ = ["TokenTransformerRecipe"]
