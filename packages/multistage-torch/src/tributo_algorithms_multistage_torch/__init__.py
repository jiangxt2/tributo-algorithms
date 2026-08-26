"""Official finite multi-stage PyTorch algorithms with lazy runtime exports."""

from __future__ import annotations

from typing import Any

from tributo_algorithms_multistage_torch.descriptor import DISTILLATION_DESCRIPTOR
from tributo_algorithms_multistage_torch.pretrain_descriptor import (
    PRETRAIN_FINETUNE_DESCRIPTOR,
)

_LAZY_EXPORTS = {
    "DistributedDistillation": (
        "tributo_algorithms_multistage_torch.algorithm",
        "DistributedDistillation",
    ),
    "DistributedPretrainFinetune": (
        "tributo_algorithms_multistage_torch.pretrain",
        "DistributedPretrainFinetune",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = [
    "DISTILLATION_DESCRIPTOR",
    "PRETRAIN_FINETUNE_DESCRIPTOR",
    "DistributedDistillation",
    "DistributedPretrainFinetune",
]
