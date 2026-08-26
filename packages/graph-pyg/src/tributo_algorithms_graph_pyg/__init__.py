"""Official PyG graph algorithms with lazy runtime exports."""

from __future__ import annotations

from typing import Any

from tributo_algorithms_graph_pyg.descriptor import (
    GRAPHSAGE_DESCRIPTOR,
    RGCN_DESCRIPTOR,
)

_LAZY_EXPORTS = {
    "DistributedGraphSAGE": (
        "tributo_algorithms_graph_pyg.algorithm",
        "DistributedGraphSAGE",
    ),
    "DistributedRGCN": ("tributo_algorithms_graph_pyg.algorithm", "DistributedRGCN"),
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
    "DistributedGraphSAGE",
    "DistributedRGCN",
    "GRAPHSAGE_DESCRIPTOR",
    "RGCN_DESCRIPTOR",
]
