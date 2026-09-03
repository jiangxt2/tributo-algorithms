"""Official distributed time-series algorithms with lazy runtime exports."""

from typing import Any

from tributo_algorithms_timeseries.descriptor import TEMPORAL_CONV_DESCRIPTOR
from tributo_algorithms_timeseries.rnn_descriptor import GRU_DESCRIPTOR, LSTM_DESCRIPTOR

_LAZY_EXPORTS = {
    "TemporalConvRecipe": (
        "tributo_algorithms_timeseries.recipe",
        "TemporalConvRecipe",
    ),
    "LSTMRecipe": ("tributo_algorithms_timeseries.rnn_recipe", "LSTMRecipe"),
    "GRURecipe": ("tributo_algorithms_timeseries.rnn_recipe", "GRURecipe"),
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
    "GRU_DESCRIPTOR",
    "GRURecipe",
    "LSTM_DESCRIPTOR",
    "LSTMRecipe",
    "TEMPORAL_CONV_DESCRIPTOR",
    "TemporalConvRecipe",
]
