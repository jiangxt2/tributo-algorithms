"""Typed TorchRecipe for pre-tokenized Transformer classification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from tributo.algorithms.api.torch_runtime import (
    TorchLossContribution,
    TorchMetricContribution,
)
from tributo.algorithms.spi import (
    TorchArtifactContext,
    TorchArtifactPlan,
    TorchBatch,
    TorchBatchContext,
    TorchBuildContext,
    TorchMetricPlan,
    TorchModuleSet,
    TorchOptimizationPlan,
    TorchRecipe,
    TorchRuntimeContext,
    TorchStepContext,
    TorchStepResult,
)


def _integer_tensor(value: object, field_name: str) -> Any:
    import torch

    tensor = torch.as_tensor(value)
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise ValueError(f"{field_name} must contain integer values")
    return tensor.to(dtype=torch.int64)


def _model(
    *, vocab_size: int, sequence_length: int, hidden_size: int, heads: int
) -> object:
    import torch

    if (
        vocab_size < 2
        or sequence_length < 1
        or hidden_size < 1
        or heads < 1
        or hidden_size % heads
    ):
        raise ValueError("Transformer dimensions are invalid")

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
            tokens = cast(torch.Tensor, values)
            if tokens.ndim != 2 or tokens.shape[1] != sequence_length:
                raise ValueError(
                    "Transformer input must have shape [batch, sequence_length]"
                )
            if bool((tokens < 0).any()) or bool((tokens >= vocab_size).any()):
                raise ValueError("Transformer token IDs are out of range")
            padding_mask = tokens.eq(0)
            if bool(padding_mask.all(dim=1).any()):
                raise ValueError("Transformer rows must contain a non-padding token")
            encoded = self.encoder(
                self.embedding(tokens), src_key_padding_mask=padding_mask
            )
            valid = (~padding_mask).to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            return self.output(pooled)

    return TokenTransformerClassifier()


def _batch(batch: object, context: TorchBatchContext) -> TorchBatch:
    import torch

    if not isinstance(batch, Mapping):
        raise ValueError("Transformer batch must be columnar")
    feature_names = context.feature_names or tuple(
        name for name in batch if name not in {context.label_name, context.weight_name}
    )
    label_name = context.label_name or ("label" if "label" in batch else None)
    if not feature_names or label_name is None:
        raise ValueError("Transformer requires input_ids and label")
    if context.weight_name is not None:
        raise ValueError("Transformer does not support sample-weight binding")
    if len(feature_names) == 1 and feature_names[0] == "input_ids":
        tokens = _integer_tensor(batch[feature_names[0]], "Transformer token IDs")
    else:
        tokens = torch.stack(
            [
                _integer_tensor(batch[name], "Transformer token IDs")
                for name in feature_names
            ],
            dim=1,
        )
    targets = torch.as_tensor(batch[label_name], dtype=torch.float32).reshape(-1, 1)
    if bool((tokens < 0).any()) or not torch.isfinite(targets).all():
        raise ValueError("Transformer inputs are malformed")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("Transformer labels must be 0 or 1")
    if bool(tokens.eq(0).all(dim=1).any()):
        raise ValueError("Transformer rows must contain a non-padding token")
    rows = int(tokens.shape[0])
    return TorchBatch(
        keyword={"input_ids": tokens},
        targets=targets,
        local_rows=rows,
    )


def _accuracy(predictions: object, targets: object) -> TorchMetricContribution:
    import torch

    logits = cast(torch.Tensor, predictions).reshape(-1)
    labels = cast(torch.Tensor, targets).reshape(-1)
    return TorchMetricContribution(
        float((torch.sigmoid(logits) >= 0.5).eq(labels >= 0.5).sum().item()),
        int(labels.numel()),
    )


class TokenTransformerRecipe(TorchRecipe):
    """Train a Transformer over pre-tokenized integer IDs with derived padding mask."""

    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet:
        import torch

        config = context.runtime.algorithm_config.get("model", {})
        if not isinstance(config, Mapping):
            raise ValueError("Transformer model config must be a mapping")
        return TorchModuleSet(
            {
                "model": _model(
                    vocab_size=int(config.get("vocab_size", 128)),
                    sequence_length=int(config.get("sequence_length", 8)),
                    hidden_size=int(config.get("hidden_size", 16)),
                    heads=int(config.get("heads", 2)),
                ),
                "loss": torch.nn.Identity(),
            }
        )

    def adapt_batch(self, batch: object, context: TorchBatchContext) -> TorchBatch:
        return _batch(batch, context)

    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        del context
        import torch

        predictions = cast(torch.nn.Module, modules["model"])(
            batch.keyword["input_ids"]
        )
        targets = cast(torch.Tensor, batch.targets)
        numerator = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction="sum"
        )
        return TorchStepResult(
            outputs={"output": predictions},
            loss=TorchLossContribution(numerator, batch.local_rows),
            metrics={"accuracy": _accuracy(predictions, targets)},
        )

    def validation_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        return self.training_step(modules, batch, context)

    def configure_optimizers(
        self, modules: TorchModuleSet, context: TorchBuildContext
    ) -> TorchOptimizationPlan:
        import torch

        config = context.runtime.algorithm_config.get("optimizer", {})
        if not isinstance(config, Mapping):
            raise ValueError("Transformer optimizer config must be a mapping")
        return TorchOptimizationPlan(
            optimizer=torch.optim.AdamW(
                cast(torch.nn.Module, modules["model"]).parameters(),
                lr=float(config.get("learning_rate", 0.001)),
                weight_decay=float(config.get("weight_decay", 0.0)),
            ),
            gradient_accumulation_steps=config.get("accumulation_steps", 1),
            max_gradient_norm=config.get("max_gradient_norm", 1.0),
        )

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"accuracy": "sum_count", "train_loss": "sum_count"})

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        config = context.stage.runtime.algorithm_config.get("model", {})
        sequence_length = (
            int(config.get("sequence_length", 8)) if isinstance(config, Mapping) else 8
        )
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {
                    "name": "input_ids",
                    "dtype": "int64",
                    "shape": ("batch", sequence_length),
                },
            ),
            output_signature=(
                {"name": "output", "dtype": "float32", "shape": ("batch", 1)},
            ),
            targets=(
                {
                    "name": "onnx-model",
                    "format": "onnx",
                    "exporter_id": "torch-onnx-v1",
                    "options": {"dynamo": False},
                },
            ),
            roles={"inference": "onnx-model"},
        )


__all__ = ["TokenTransformerRecipe"]
