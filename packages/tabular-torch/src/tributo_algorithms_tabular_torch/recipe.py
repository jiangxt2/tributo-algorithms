"""Typed TorchRecipe implementations for dense DNN and PU learning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from tributo.algorithms.api.torch_runtime import (
    TorchCompositeLossContribution,
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


def _feature_and_label_names(
    batch: Mapping[str, object], context: TorchBatchContext
) -> tuple[tuple[str, ...], str]:
    label_name = context.label_name or ("label" if "label" in batch else None)
    if label_name is None:
        raise ValueError("tabular-torch training requires a label")
    feature_names = context.feature_names or tuple(
        name for name in batch if name not in {label_name, context.weight_name}
    )
    if not feature_names:
        raise ValueError("tabular-torch requires at least one feature")
    return tuple(feature_names), label_name


def _dense_batch(batch: object, context: TorchBatchContext) -> TorchBatch:
    import torch

    if not isinstance(batch, Mapping):
        raise ValueError("tabular-torch batch must be columnar")
    feature_names, label_name = _feature_and_label_names(batch, context)
    if context.weight_name is not None:
        raise ValueError("tabular-torch does not support sample-weight binding")
    try:
        features = torch.stack(
            [
                torch.as_tensor(batch[name], dtype=torch.float32)
                for name in feature_names
            ],
            dim=1,
        )
        labels = torch.as_tensor(batch[label_name], dtype=torch.float32).reshape(-1, 1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("tabular-torch batch columns are malformed") from exc
    if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
        raise ValueError("tabular-torch features and labels must be finite")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("tabular-torch labels must be 0 or 1")
    rows = int(features.shape[0])
    return TorchBatch(
        keyword={"features": features},
        targets=labels,
        local_rows=rows,
        coverage_counts={"train": rows},
    )


def _binary_metric(predictions: object, targets: object) -> TorchMetricContribution:
    import torch

    logits = cast(torch.Tensor, predictions).reshape(-1)
    labels = cast(torch.Tensor, targets).reshape(-1)
    correct = (torch.sigmoid(logits) >= 0.5).eq(labels >= 0.5)
    return TorchMetricContribution(float(correct.sum().item()), int(labels.numel()))


def _model(config: Mapping[str, Any]) -> object:
    import torch

    input_features = int(config.get("input_features", 4))
    hidden_units = tuple(int(value) for value in config.get("hidden_units", (16, 8)))
    if (
        input_features < 1
        or not hidden_units
        or any(value < 1 for value in hidden_units)
    ):
        raise ValueError("DNN dimensions must be positive")
    layers: list[torch.nn.Module] = []
    current = input_features
    for width in hidden_units:
        layers.extend((torch.nn.Linear(current, width), torch.nn.ReLU()))
        current = width
    layers.append(torch.nn.Linear(current, 1))
    return torch.nn.Sequential(*layers)


class _BaseDenseRecipe(TorchRecipe):
    """Shared typed hooks for dense binary classifiers."""

    def adapt_batch(self, batch: object, context: TorchBatchContext) -> TorchBatch:
        return _dense_batch(batch, context)

    def configure_optimizers(
        self, modules: TorchModuleSet, context: TorchBuildContext
    ) -> TorchOptimizationPlan:
        import torch

        config = _config(context)
        optimizer_config = config.get("optimizer", {})
        if not isinstance(optimizer_config, Mapping):
            raise ValueError("optimizer config must be a mapping")
        model = cast(torch.nn.Module, modules["model"])
        return TorchOptimizationPlan(
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=float(optimizer_config.get("learning_rate", 0.001)),
                weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
            ),
            gradient_accumulation_steps=int(
                optimizer_config.get("accumulation_steps", 1)
            ),
            max_gradient_norm=float(optimizer_config.get("max_gradient_norm", 1.0)),
        )

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        config = _config(context.stage.runtime)
        model_config = config.get("model", {})
        input_features = (
            int(model_config.get("input_features", 4))
            if isinstance(model_config, Mapping)
            else 4
        )
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {
                    "name": "features",
                    "dtype": "float32",
                    "shape": ("batch", input_features),
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


class DNNRecipe(_BaseDenseRecipe):
    """Binary dense DNN classification."""

    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet:
        import torch

        config = _config(context)
        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("model config must be a mapping")
        return TorchModuleSet(
            {"model": _model(model_config), "loss": torch.nn.Identity()}
        )

    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        del context
        import torch

        model = cast(torch.nn.Module, modules["model"])
        predictions = model(batch.keyword["features"])
        targets = cast(torch.Tensor, batch.targets)
        numerator = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, targets, reduction="sum"
        )
        return TorchStepResult(
            outputs={"output": predictions},
            loss=TorchLossContribution(numerator, batch.local_rows),
            coverage_counts=dict(batch.coverage_counts),
            metrics={"accuracy": _binary_metric(predictions, targets)},
        )

    def validation_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        return self.training_step(modules, batch, context)

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"accuracy": "sum_count", "train_loss": "sum_count"})


class PURecipe(_BaseDenseRecipe):
    """Positive-unlabeled classifier using the Wheel-owned global reducer."""

    component_schema_id = "tributo.official.tabular_torch.pu-risk-components.v1"

    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet:
        import torch

        config = _config(context)
        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("model config must be a mapping")
        return TorchModuleSet(
            {"model": _model(model_config), "loss": torch.nn.Identity()}
        )

    def configure_optimizers(
        self, modules: TorchModuleSet, context: TorchBuildContext
    ) -> TorchOptimizationPlan:
        plan = super().configure_optimizers(modules, context)
        if plan.gradient_accumulation_steps != 1:
            raise ValueError("PU requires gradient_accumulation_steps=1")
        return plan

    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        del context
        import torch

        model = cast(torch.nn.Module, modules["model"])
        predictions = model(batch.keyword["features"]).reshape(-1)
        labels = cast(torch.Tensor, batch.targets).reshape(-1)
        positive = labels == 1
        unlabeled = labels == 0
        positive_loss = torch.nn.functional.softplus(-predictions[positive]).sum()
        positive_as_negative = torch.nn.functional.softplus(predictions[positive]).sum()
        unlabeled_negative = torch.nn.functional.softplus(predictions[unlabeled]).sum()
        positive_count = int(positive.sum().item())
        unlabeled_count = int(unlabeled.sum().item())
        metrics = {
            "observed_positive_recall": TorchMetricContribution(
                float((torch.sigmoid(predictions[positive]) >= 0.5).sum().item()),
                positive_count,
            )
        }
        return TorchStepResult(
            outputs={"output": predictions.reshape(-1, 1)},
            loss=TorchCompositeLossContribution(
                self.component_schema_id,
                {
                    "positive_loss_sum": positive_loss,
                    "positive_as_negative_sum": positive_as_negative,
                    "unlabeled_negative_sum": unlabeled_negative,
                },
                {
                    "positive_count": positive_count,
                    "unlabeled_count": unlabeled_count,
                },
                evidence={
                    "positive_count": positive_count,
                    "unlabeled_count": unlabeled_count,
                },
            ),
            coverage_counts={
                "train": batch.local_rows,
                "coverage.positive": positive_count,
                "coverage.unlabeled": unlabeled_count,
            },
            metrics=metrics,
        )

    def validation_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        return self.training_step(modules, batch, context)

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan(
            {
                "observed_positive_recall": "sum_count",
                "train_loss": "sum_count",
            }
        )


__all__ = ["DNNRecipe", "PURecipe"]
