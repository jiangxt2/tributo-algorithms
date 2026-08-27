"""Tests for fixed-window LSTM and GRU recipes."""

from __future__ import annotations

import torch
from tributo_algorithms_timeseries.rnn_descriptor import GRU_DESCRIPTOR, LSTM_DESCRIPTOR
from tributo_algorithms_timeseries.rnn_recipe import GRURecipe, LSTMRecipe

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "lag_0": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "lag_1": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "lag_2": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "lag_3": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "label": torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }


def test_lstm_recipe_produces_fixed_window_logits() -> None:
    recipe = LSTMRecipe()
    modules = recipe.build_modules({"model": {"input_features": 4, "hidden_size": 4}})
    features, targets, weights, rows = recipe.batch_adapter(
        _batch(),
        feature_names=("lag_0", "lag_1", "lag_2", "lag_3"),
        label_name="label",
        weight_name=None,
        config={},
    )
    result = recipe.training_step(modules, features, targets, weights, {})
    assert tuple(result.predictions.shape) == (rows, 1)


def test_gru_descriptor_and_recipe_are_registered() -> None:
    assert LSTM_DESCRIPTOR.name == "lstm_classifier"
    assert GRU_DESCRIPTOR.name == "gru_classifier"
    assert (
        GRURecipe().build_modules({"model": {"input_features": 4}})["model"] is not None
    )
