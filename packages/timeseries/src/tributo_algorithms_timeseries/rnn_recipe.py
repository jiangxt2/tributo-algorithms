"""Typed TorchRecipe implementations for fixed-window LSTM and GRU."""

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


def _config(context: TorchRuntimeContext | TorchBuildContext) -> Mapping[str, Any]:
    runtime = context if isinstance(context, TorchRuntimeContext) else context.runtime
    return cast(Mapping[str, Any], runtime.algorithm_config)


def _batch(batch: object, context: TorchBatchContext) -> TorchBatch:
    import torch

    if not isinstance(batch, Mapping):
        raise ValueError("RNN batch must be columnar")
    features = context.feature_names or tuple(
        name for name in batch if name not in {context.label_name, context.weight_name}
    )
    label_name = context.label_name or ("label" if "label" in batch else None)
    if len(features) < 2 or label_name is None:
        raise ValueError("RNN requires a fixed window and label")
    if context.weight_name is not None:
        raise ValueError("RNN algorithms do not support sample-weight binding")
    sequence = torch.stack(
        [torch.as_tensor(batch[name], dtype=torch.float32) for name in features], dim=1
    ).unsqueeze(-1)
    targets = torch.as_tensor(batch[label_name], dtype=torch.float32).reshape(-1, 1)
    if not torch.isfinite(sequence).all() or not torch.isfinite(targets).all():
        raise ValueError("RNN inputs must be finite")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("RNN labels must be 0 or 1")
    rows = int(sequence.shape[0])
    return TorchBatch(
        keyword={"window": sequence},
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
            sequence = cast(torch.Tensor, values)
            if sequence.ndim == 2 and sequence.shape[1] == input_features:
                sequence = sequence.unsqueeze(-1)
            if (
                sequence.ndim != 3
                or sequence.shape[-1] != 1
                or sequence.shape[1] != input_features
            ):
                raise ValueError("RNN input must have shape [batch, window, 1]")
            encoded, _ = self.recurrent(sequence)
            return self.output(encoded[:, -1, :])

    return RecurrentClassifier()


class _BaseRNNRecipe(TorchRecipe):
    recurrent_kind = "lstm"

    def adapt_batch(self, batch: object, context: TorchBatchContext) -> TorchBatch:
        return _batch(batch, context)

    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        del context
        import torch

        predictions = cast(torch.nn.Module, modules["model"])(batch.keyword["window"])
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

        optimizer_config = _config(context).get("optimizer", {})
        if not isinstance(optimizer_config, Mapping):
            raise ValueError("RNN optimizer config must be a mapping")
        return TorchOptimizationPlan(
            optimizer=torch.optim.Adam(
                cast(torch.nn.Module, modules["model"]).parameters(),
                lr=float(optimizer_config.get("learning_rate", 0.001)),
                weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
            ),
            gradient_accumulation_steps=optimizer_config.get("accumulation_steps", 1),
            max_gradient_norm=optimizer_config.get("max_gradient_norm", 1.0),
        )

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"accuracy": "sum_count", "train_loss": "sum_count"})

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        config = context.stage.runtime.algorithm_config
        model_config = config.get("model", {})
        features = (
            int(model_config.get("input_features", 4))
            if isinstance(model_config, Mapping)
            else 4
        )
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {"name": "window", "dtype": "float32", "shape": ("batch", features)},
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


class LSTMRecipe(_BaseRNNRecipe):
    """Train a fixed-window LSTM binary classifier."""

    recurrent_kind = "lstm"

    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet:
        import torch

        config = _config(context).get("model", {})
        if not isinstance(config, Mapping):
            raise ValueError("LSTM model config must be a mapping")
        return TorchModuleSet(
            {
                "model": _model(config, recurrent_kind="lstm"),
                "loss": torch.nn.Identity(),
            }
        )


class GRURecipe(_BaseRNNRecipe):
    """Train a fixed-window GRU binary classifier."""

    recurrent_kind = "gru"

    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet:
        import torch

        config = _config(context).get("model", {})
        if not isinstance(config, Mapping):
            raise ValueError("GRU model config must be a mapping")
        return TorchModuleSet(
            {"model": _model(config, recurrent_kind="gru"), "loss": torch.nn.Identity()}
        )


__all__ = ["GRURecipe", "LSTMRecipe"]
