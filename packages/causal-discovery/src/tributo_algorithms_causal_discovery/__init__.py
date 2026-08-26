"""Official distributed causal discovery algorithms with lazy runtime exports."""

from __future__ import annotations

from typing import Any

from tributo_algorithms_causal_discovery.descriptor import PC_DISCOVERY_DESCRIPTOR


def __getattr__(name: str) -> Any:
    if name != "DistributedPCStability":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(
        importlib.import_module("tributo_algorithms_causal_discovery.algorithm"),
        name,
    )
    globals()[name] = value
    return value


__all__ = ["PC_DISCOVERY_DESCRIPTOR", "DistributedPCStability"]
