"""Typed TorchRecipe implementation for the Two-Tower recommender."""

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

        def forward(self, user_id: object, item_id: object) -> object:
            users = cast(torch.Tensor, user_id).long()
            items = cast(torch.Tensor, item_id).long()
            if users.ndim != 1 or items.ndim != 1 or users.shape != items.shape:
                raise ValueError("Two-Tower IDs must have shape [batch]")
            if bool((users < 0).any()) or bool((users >= user_count).any()):
                raise ValueError("Two-Tower user ID is out of range")
            if bool((items < 0).any()) or bool((items >= item_count).any()):
                raise ValueError("Two-Tower item ID is out of range")
            score = (self.user_embedding(users) * self.item_embedding(items)).sum(
                dim=1, keepdim=True
            )
            return score + self.user_bias(users) + self.item_bias(items)

    return TwoTower()


def _batch(batch: object, context: TorchBatchContext) -> TorchBatch:
    import torch

    if not isinstance(batch, Mapping):
        raise ValueError("Two-Tower batch must be columnar")
    names = context.feature_names or ("user_id", "item_id")
    label_name = context.label_name or ("label" if "label" in batch else None)
    if len(names) != 2 or label_name is None:
        raise ValueError("Two-Tower requires user_id, item_id, and label")
    if context.weight_name is not None:
        raise ValueError("Two-Tower does not support sample-weight binding")
    user_id = torch.as_tensor(batch[names[0]], dtype=torch.int64)
    item_id = torch.as_tensor(batch[names[1]], dtype=torch.int64)
    labels = torch.as_tensor(batch[label_name], dtype=torch.float32).reshape(-1, 1)
    if not torch.isfinite(labels).all() or not bool(
        ((labels == 0) | (labels == 1)).all()
    ):
        raise ValueError("Two-Tower labels must be finite 0 or 1")
    if user_id.ndim != 1 or item_id.ndim != 1 or user_id.shape != item_id.shape:
        raise ValueError("Two-Tower IDs must have shape [batch]")
    rows = int(user_id.shape[0])
    return TorchBatch(
        keyword={"user_id": user_id, "item_id": item_id},
        targets=labels,
        local_rows=rows,
        coverage_counts={
            "train": rows,
            "coverage.positive_pairs": int((labels == 1).sum().item()),
            "coverage.negative_pairs": int((labels == 0).sum().item()),
        },
    )


class TwoTowerRecipe(TorchRecipe):
    """Train user/item embeddings from named interaction ID tensors."""

    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet:
        import torch

        config = context.runtime.algorithm_config.get("model", {})
        if not isinstance(config, Mapping):
            raise ValueError("Two-Tower model config must be a mapping")
        return TorchModuleSet({"model": _model(config), "loss": torch.nn.Identity()})

    def adapt_batch(self, batch: object, context: TorchBatchContext) -> TorchBatch:
        return _batch(batch, context)

    def training_step(
        self, modules: TorchModuleSet, batch: TorchBatch, context: TorchStepContext
    ) -> TorchStepResult:
        del context
        import torch

        model = cast(torch.nn.Module, modules["model"])
        predictions = model(batch.keyword["user_id"], batch.keyword["item_id"])
        labels = cast(torch.Tensor, batch.targets)
        numerator = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, labels, reduction="sum"
        )
        correct = (torch.sigmoid(predictions) >= 0.5).eq(labels >= 0.5)
        return TorchStepResult(
            outputs={"output": predictions},
            loss=TorchLossContribution(numerator, batch.local_rows),
            coverage_counts=dict(batch.coverage_counts),
            metrics={
                "pair_accuracy": TorchMetricContribution(
                    float(correct.sum().item()), batch.local_rows
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

        config = context.runtime.algorithm_config.get("optimizer", {})
        if not isinstance(config, Mapping):
            raise ValueError("Two-Tower optimizer config must be a mapping")
        return TorchOptimizationPlan(
            optimizer=torch.optim.Adam(
                cast(torch.nn.Module, modules["model"]).parameters(),
                lr=float(config.get("learning_rate", 0.01)),
                weight_decay=float(config.get("weight_decay", 0.0)),
            ),
            gradient_accumulation_steps=int(config.get("accumulation_steps", 1)),
            max_gradient_norm=float(config.get("max_gradient_norm", 1.0)),
        )

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan(
            {"pair_accuracy": "sum_count", "train_loss": "sum_count"}
        )

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {"name": "user_id", "dtype": "int64", "shape": ("batch",)},
                {"name": "item_id", "dtype": "int64", "shape": ("batch",)},
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


__all__ = ["TwoTowerRecipe"]
