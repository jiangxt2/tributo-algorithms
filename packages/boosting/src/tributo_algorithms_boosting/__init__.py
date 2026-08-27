"""Official distributed boosting algorithms.

The package root deliberately imports only the lightweight descriptor. Runtime
implementations are lazy so Descriptor-only catalog discovery does not import
Ray Train or XGBoost implementation modules.
"""

from __future__ import annotations

from typing import Any

from tributo_algorithms_boosting.descriptor import XGBOOST_DESCRIPTOR
from tributo_algorithms_boosting.lightgbm_descriptor import LIGHTGBM_DESCRIPTOR

_LAZY_EXPORTS = {
    "DistributedXGBoost": (
        "tributo_algorithms_boosting.algorithm",
        "DistributedXGBoost",
    ),
    "XGBoostStageResult": ("tributo_algorithms_boosting.stages", "XGBoostStageResult"),
    "XGBoostStageRunner": ("tributo_algorithms_boosting.stages", "XGBoostStageRunner"),
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
    "DistributedXGBoost",
    "LIGHTGBM_DESCRIPTOR",
    "XGBOOST_DESCRIPTOR",
    "XGBoostStageResult",
    "XGBoostStageRunner",
]
