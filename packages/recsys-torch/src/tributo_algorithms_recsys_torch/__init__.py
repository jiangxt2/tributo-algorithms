"""Official distributed recommendation algorithms with lazy runtime exports."""

from __future__ import annotations

from typing import Any

from tributo_algorithms_recsys_torch.descriptor import TWO_TOWER_DESCRIPTOR
from tributo_algorithms_recsys_torch.jagged_descriptor import JAGGED_DESCRIPTOR

_LAZY_EXPORTS = {
    "DistributedJaggedEmbedding": (
        "tributo_algorithms_recsys_torch.jagged",
        "DistributedJaggedEmbedding",
    ),
    "TwoTowerRecipe": ("tributo_algorithms_recsys_torch.recipe", "TwoTowerRecipe"),
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
    "JAGGED_DESCRIPTOR",
    "TWO_TOWER_DESCRIPTOR",
    "DistributedJaggedEmbedding",
    "TwoTowerRecipe",
]
