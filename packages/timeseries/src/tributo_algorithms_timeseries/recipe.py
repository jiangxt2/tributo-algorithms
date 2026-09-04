"""Typed TorchRecipe implementation for the temporal convolution classifier."""

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


def _window_batch(batch: object, context: TorchBatchContext) -> TorchBatch:
    import torch

    if not isinstance(batch, Mapping):
        raise ValueError("time-series batch must be columnar")
    feature_names = context.feature_names or tuple(
        name for name in batch if name not in {context.label_name, context.weight_name}
    )
    label_name = context.label_name or ("label" if "label" in batch else None)
    if not feature_names or label_name is None:
        raise ValueError("time-series classifier requires ordered features and label")
    if context.weight_name is not None:
        raise ValueError("time-series algorithms do not support sample-weight binding")
    windows = torch.stack(
        [torch.as_tensor(batch[name], dtype=torch.float32) for name in feature_names],
        dim=1,
    )
    targets = torch.as_tensor(batch[label_name], dtype=torch.float32).reshape(-1, 1)
    if not torch.isfinite(windows).all() or not torch.isfinite(targets).all():
        raise ValueError("time-series features and labels must be finite")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("time-series labels must be 0 or 1")
    rows = int(windows.shape[0])
    return TorchBatch(
        keyword={"window": windows},
        targets=targets,
        local_rows=rows,
    )


def _accuracy(predictions: object, targets: object) -> TorchMetricContribution:
    import torch

    logits = cast(torch.Tensor, predictions).reshape(-1)
    labels = cast(torch.Tensor, targets).reshape(-1)
    predicted = torch.sigmoid(logits) >= 0.5
    return TorchMetricContribution(
        float(predicted.eq(labels >= 0.5).sum().item()), int(labels.numel())
    )


def _model(input_features: int, channels: int) -> object:
    import torch

    if input_features < 1 or channels < 1:
        raise ValueError("TCN dimensions must be positive")

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
            tensor = cast(torch.Tensor, values)
            if tensor.ndim != 2 or tensor.shape[1] != input_features:
                raise ValueError("TCN input must have shape [batch, window]")
            encoded = self.network(tensor.unsqueeze(1)).squeeze(-1)
            return self.output(encoded)

    return TemporalConvClassifier()


class TemporalConvRecipe(TorchRecipe):
    """Train a finite-window temporal convolution classifier."""

    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet:
        import torch

        model_config = _config(context).get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("TCN model config must be a mapping")
        return TorchModuleSet(
            {
                "model": _model(
                    int(model_config.get("input_features", 4)),
                    int(model_config.get("channels", 8)),
                ),
                "loss": torch.nn.Identity(),
            }
        )

    def adapt_batch(self, batch: object, context: TorchBatchContext) -> TorchBatch:
        return _window_batch(batch, context)

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
            raise ValueError("TCN optimizer config must be a mapping")
        return TorchOptimizationPlan(
            optimizer=torch.optim.Adam(
                cast(torch.nn.Module, modules["model"]).parameters(),
                lr=float(optimizer_config.get("learning_rate", 0.01)),
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


__all__ = ["TemporalConvRecipe"]
