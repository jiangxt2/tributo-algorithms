"""Official distributed DoWhy adapters with lazy runtime exports."""

from __future__ import annotations

from typing import Any

from tributo_algorithms_causal_dowhy.descriptor import DOWHY_DESCRIPTOR
from tributo_algorithms_causal_dowhy.gcm_descriptor import GCM_DESCRIPTOR

_LAZY_EXPORTS = {
    "DistributedDoWhyRefutation": (
        "tributo_algorithms_causal_dowhy.algorithm",
        "DistributedDoWhyRefutation",
    ),
    "DistributedGCMRootCause": (
        "tributo_algorithms_causal_dowhy.gcm",
        "DistributedGCMRootCause",
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
    "DOWHY_DESCRIPTOR",
    "GCM_DESCRIPTOR",
    "DistributedDoWhyRefutation",
    "DistributedGCMRootCause",
]
