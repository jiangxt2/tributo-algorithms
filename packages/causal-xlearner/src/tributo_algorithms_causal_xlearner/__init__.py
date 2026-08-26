"""Official distributed X-Learner with lazy runtime exports."""

from __future__ import annotations

from typing import Any

from tributo_algorithms_causal_xlearner.descriptor import X_LEARNER_DESCRIPTOR

_LAZY_EXPORTS = {
    "DistributedXLearner": (
        "tributo_algorithms_causal_xlearner.algorithm",
        "DistributedXLearner",
    ),
    "XLearnerModel": ("tributo_algorithms_causal_xlearner.model", "XLearnerModel"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = ["DistributedXLearner", "XLearnerModel", "X_LEARNER_DESCRIPTOR"]
