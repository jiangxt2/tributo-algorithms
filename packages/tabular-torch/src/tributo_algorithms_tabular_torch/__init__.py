"""Official distributed tabular PyTorch algorithms with lazy runtime exports."""

from __future__ import annotations

from typing import Any

from tributo_algorithms_tabular_torch.descriptor import DNN_DESCRIPTOR, PU_DESCRIPTOR

_LAZY_EXPORTS = {
    "DNNRecipe": ("tributo_algorithms_tabular_torch.recipe", "DNNRecipe"),
    "PURecipe": ("tributo_algorithms_tabular_torch.recipe", "PURecipe"),
    "estimate_class_prior": (
        "tributo_algorithms_tabular_torch.priors",
        "estimate_class_prior",
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
    "DNN_DESCRIPTOR",
    "DNNRecipe",
    "PU_DESCRIPTOR",
    "PURecipe",
    "estimate_class_prior",
]
