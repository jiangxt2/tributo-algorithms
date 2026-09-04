"""Typed TorchRecipe implementation for the tabular autoencoder."""

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


def _batch(batch: object, context: TorchBatchContext) -> TorchBatch:
    import torch

    if not isinstance(batch, Mapping):
        raise ValueError("Autoencoder batch must be columnar")
    feature_names = context.feature_names or tuple(batch)
    if not feature_names:
        raise ValueError("Autoencoder requires dense feature columns")
    if context.label_name is not None or context.weight_name is not None:
        raise ValueError("Autoencoder does not accept label or sample-weight bindings")
    features = torch.stack(
        [torch.as_tensor(batch[name], dtype=torch.float32) for name in feature_names],
        dim=1,
    )
    if not torch.isfinite(features).all():
        raise ValueError("Autoencoder features must be finite")
    rows = int(features.shape[0])
    return TorchBatch(
        keyword={"features": features},
        targets=features,
        local_rows=rows,
    )


def _model(config: Mapping[str, Any]) -> object:
    import torch

    input_features = int(config.get("input_features", 4))
    latent_features = int(config.get("latent_features", 2))
    if input_features < 1 or latent_features < 1:
        raise ValueError("Autoencoder dimensions must be positive")
    return torch.nn.Sequential(
        torch.nn.Linear(input_features, latent_features),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_features, input_features),
    )


class TabularAutoencoderRecipe(TorchRecipe):
    """Reconstruct dense inputs with element-normalized squared error."""

    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet:
        import torch

        model_config = context.runtime.algorithm_config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("Autoencoder model config must be a mapping")
        return TorchModuleSet(
            {"model": _model(model_config), "loss": torch.nn.Identity()}
        )

    def adapt_batch(self, batch: object, context: TorchBatchContext) -> TorchBatch:
        return _batch(batch, context)

    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        del context
        import torch

        model = cast(torch.nn.Module, modules["model"])
        features = cast(torch.Tensor, batch.keyword["features"])
        reconstruction = model(features)
        squared_error = (reconstruction - cast(torch.Tensor, batch.targets)) ** 2
        numerator = squared_error.sum()
        element_count = int(squared_error.numel())
        return TorchStepResult(
            outputs={"output": reconstruction},
            loss=TorchLossContribution(numerator, element_count),
            metrics={
                "reconstruction_mse": TorchMetricContribution(
                    float(numerator.detach().item()), element_count
                )
            },
        )

    def validation_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        return self.training_step(modules, batch, context)

    def configure_optimizers(
        self, modules: TorchModuleSet, context: TorchBuildContext
    ) -> TorchOptimizationPlan:
        import torch

        optimizer_config = context.runtime.algorithm_config.get("optimizer", {})
        if not isinstance(optimizer_config, Mapping):
            raise ValueError("Autoencoder optimizer config must be a mapping")
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
        return TorchMetricPlan(
            {"reconstruction_mse": "sum_count", "train_loss": "sum_count"}
        )

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        config = context.stage.runtime.algorithm_config.get("model", {})
        features = (
            int(config.get("input_features", 4)) if isinstance(config, Mapping) else 4
        )
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {"name": "features", "dtype": "float32", "shape": ("batch", features)},
            ),
            output_signature=(
                {
                    "name": "output",
                    "dtype": "float32",
                    "shape": ("batch", features),
                },
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


__all__ = ["TabularAutoencoderRecipe"]
