"""TrainingRecipeV2 implementations for dense DNN and PU learning."""

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


def _binary_accuracy(predictions: object, targets: object) -> object:
    import torch

    logits = cast(Any, predictions)
    labels = cast(Any, targets)
    predicted = (torch.sigmoid(logits) >= 0.5).to(dtype=labels.dtype)
    return (predicted == labels).to(dtype=torch.float32).mean()


def _observed_positive_recall(predictions: object, targets: object) -> object:
    import torch

    logits = cast(Any, predictions)
    labels = cast(Any, targets)
    positive = labels == 1
    if not bool(positive.any()):
        return torch.zeros((), device=logits.device)
    return (torch.sigmoid(logits[positive]) >= 0.5).to(torch.float32).mean()


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


def _dense_batch(
    batch: object,
    *,
    feature_names: tuple[str, ...],
    label_name: str | None,
    weight_name: str | None,
) -> tuple[object, object, object | None, int]:
    import torch

    if not isinstance(batch, Mapping):
        raise ValueError("tabular-torch batch must be columnar")
    if label_name is None:
        raise ValueError("tabular-torch training requires a label")
    features = torch.stack(
        [batch[name].to(dtype=torch.float32) for name in feature_names], dim=1
    )
    labels = batch[label_name].to(dtype=torch.float32).reshape(-1, 1)
    weights = batch.get(weight_name) if weight_name is not None else None
    return features, labels, weights, int(features.shape[0])


class _PULoss:
    """Lazy nnPU/uPU callable retaining the published risk semantics."""

    def __init__(
        self,
        *,
        class_prior: float,
        beta: float,
        gamma: float,
        loss_type: str,
    ) -> None:
        if not 0 < class_prior < 1:
            raise ValueError("PU class_prior must be in (0, 1)")
        if beta < 0 or not 0 <= gamma <= 1 or loss_type not in {"nnpu", "upu"}:
            raise ValueError("PU risk configuration is invalid")
        self.class_prior = class_prior
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.last_local_counts = {"positive": 0, "unlabeled": 0}

    def __call__(self, logits: object, labels: object) -> object:
        import torch
        import torch.distributed as dist

        values = cast(Any, logits).reshape(-1)
        expected = cast(Any, labels).reshape(-1)
        positive = expected == 1
        unlabeled = expected == 0
        self.last_local_counts = {
            "positive": int(positive.sum()),
            "unlabeled": int(unlabeled.sum()),
        }
        if not bool(torch.all(positive | unlabeled)):
            raise ValueError("PU labels must be 1 (positive) or 0 (unlabeled)")
        positive_loss_sum = torch.nn.functional.softplus(-values[positive]).sum()
        positive_as_negative_sum = torch.nn.functional.softplus(values[positive]).sum()
        unlabeled_negative_sum = torch.nn.functional.softplus(values[unlabeled]).sum()
        counts = torch.tensor(
            [int(positive.sum()), int(unlabeled.sum())],
            dtype=values.dtype,
            device=values.device,
        )
        detached_sums = torch.stack(
            (
                positive_loss_sum.detach(),
                positive_as_negative_sum.detach(),
                unlabeled_negative_sum.detach(),
            )
        )
        world_size = 1
        if dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(detached_sums, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        if bool((counts <= 0).any()):
            raise ValueError(
                "each global PU step must contain positive and unlabeled examples"
            )
        scale = float(world_size)
        positive_risk = scale * self.class_prior * positive_loss_sum / counts[0]
        negative_risk = scale * (
            unlabeled_negative_sum / counts[1]
            - self.class_prior * positive_as_negative_sum / counts[0]
        )
        global_negative_risk = (
            detached_sums[2] / counts[1]
            - self.class_prior * detached_sums[1] / counts[0]
        )
        if self.loss_type == "nnpu" and bool(global_negative_risk < -self.beta):
            return -self.gamma * negative_risk
        return positive_risk + negative_risk


class _BaseDenseRecipe(TrainingRecipeV2):
    metric_name = "accuracy"

    def batch_adapter(
        self,
        batch: object,
        *,
        feature_names: tuple[str, ...],
        label_name: str | None,
        weight_name: str | None,
        config: Mapping[str, Any],
    ) -> tuple[object, object, object | None, int]:
        del config
        return _dense_batch(
            batch,
            feature_names=feature_names,
            label_name=label_name,
            weight_name=weight_name,
        )

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
                lr=float(config.get("learning_rate", 0.001)),
                weight_decay=float(config.get("weight_decay", 0.0)),
            ),
            gradient_accumulation_steps=int(config.get("accumulation_steps", 1)),
            max_gradient_norm=float(config.get("max_gradient_norm", 1.0)),
        )

    def checkpoint_codec(self) -> object:
        return _CheckpointCodec()


class DNNRecipe(_BaseDenseRecipe):
    """Binary dense DNN classification without Ray control-plane code."""

    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        import torch

        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            raise ValueError("model config must be a mapping")
        return {"model": _model(model_config), "loss": torch.nn.BCEWithLogitsLoss()}

    def metric_plan(self, config: Mapping[str, Any]) -> MetricPlan:
        del config
        return MetricPlan(factories={"accuracy": _binary_accuracy})


class PURecipe(_BaseDenseRecipe):
    """Neural nnPU/uPU classifier over positive and unlabeled labels."""

    def build_modules(self, config: Mapping[str, Any]) -> Mapping[str, object]:
        import torch

        model_config = config.get("model", {})
        loss_config = config.get("loss", {})
        if not isinstance(model_config, Mapping) or not isinstance(
            loss_config, Mapping
        ):
            raise ValueError("PU model and loss config must be mappings")
        loss: object
        if "class_prior" not in loss_config:
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = _PULoss(
                class_prior=float(loss_config["class_prior"]),
                beta=float(loss_config.get("beta", 0.0)),
                gamma=float(loss_config.get("gamma", 1.0)),
                loss_type=str(loss_config.get("type", "nnpu")),
            )
        return {"model": _model(model_config), "loss": loss}

    def metric_plan(self, config: Mapping[str, Any]) -> MetricPlan:
        del config
        return MetricPlan(
            factories={"observed_positive_recall": _observed_positive_recall}
        )

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
        value = loss(predictions, targets)
        counts = getattr(loss, "last_local_counts", {})
        return TrainingStepResult(predictions, value, coverage_counts=counts)


__all__ = ["DNNRecipe", "PURecipe"]
