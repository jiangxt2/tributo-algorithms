"""Official distributed doubly robust estimators with lazy runtime exports."""

from __future__ import annotations

from typing import Any

from tributo_algorithms_causal_dr.descriptor import DR_DESCRIPTOR


def __getattr__(name: str) -> Any:
    if name != "DistributedDRLearner":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(
        importlib.import_module("tributo_algorithms_causal_dr.algorithm"), name
    )
    globals()[name] = value
    return value


__all__ = ["DR_DESCRIPTOR", "DistributedDRLearner"]
