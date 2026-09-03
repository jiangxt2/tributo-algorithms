"""Official distributed Transformer NLP algorithms with lazy exports."""

from typing import Any

from tributo_algorithms_transformers_nlp.descriptor import TOKEN_TRANSFORMER_DESCRIPTOR


def __getattr__(name: str) -> Any:
    if name != "TokenTransformerRecipe":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(
        importlib.import_module("tributo_algorithms_transformers_nlp.recipe"), name
    )
    globals()[name] = value
    return value


__all__ = ["TOKEN_TRANSFORMER_DESCRIPTOR", "TokenTransformerRecipe"]
