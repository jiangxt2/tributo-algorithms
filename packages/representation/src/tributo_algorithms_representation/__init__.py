"""Official self-supervised representation algorithms with lazy exports."""

from typing import Any

from tributo_algorithms_representation.descriptor import TABULAR_AUTOENCODER_DESCRIPTOR


def __getattr__(name: str) -> Any:
    if name != "TabularAutoencoderRecipe":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(
        importlib.import_module("tributo_algorithms_representation.recipe"), name
    )
    globals()[name] = value
    return value


__all__ = ["TABULAR_AUTOENCODER_DESCRIPTOR", "TabularAutoencoderRecipe"]
