"""Official distributed causal algorithms with lazy runtime exports."""

from __future__ import annotations

from typing import Any

from tributo_algorithms_causal_core.descriptor import (
    ATE_DESCRIPTOR,
    LINEAR_DML_DESCRIPTOR,
    LINEAR_IV_DESCRIPTOR,
)

_LAZY_EXPORTS = {
    "DifferenceInMeansATE": (
        "tributo_algorithms_causal_core.algorithm",
        "DifferenceInMeansATE",
    ),
    "LinearDMLATE": ("tributo_algorithms_causal_core.algorithm", "LinearDMLATE"),
    "LinearIVATE": ("tributo_algorithms_causal_core.algorithm", "LinearIVATE"),
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
    "ATE_DESCRIPTOR",
    "LINEAR_DML_DESCRIPTOR",
    "LINEAR_IV_DESCRIPTOR",
    "DifferenceInMeansATE",
    "LinearDMLATE",
    "LinearIVATE",
]
