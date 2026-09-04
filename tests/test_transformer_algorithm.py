"""Tests for the pre-tokenized Transformer TorchRecipe."""

from __future__ import annotations

import hashlib

import pytest
import torch
from tributo.algorithms import (
    DistributionStrategy,
    TorchBatchContext,
    TorchBuildContext,
    TorchRuntimeContext,
    TorchStageContext,
    TorchStageRunIdentity,
    TorchStepContext,
)
from tributo_algorithms_transformers_nlp import TOKEN_TRANSFORMER_DESCRIPTOR
from tributo_algorithms_transformers_nlp.recipe import TokenTransformerRecipe


def _context(optimizer: dict[str, object] | None = None) -> TorchBuildContext:
    policy = TOKEN_TRANSFORMER_DESCRIPTOR.registration.distribution_spec.policy
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        "transformer",
        "transformer",
        hashlib.sha256(b"transformer").hexdigest(),
        policy.digest,
        policy.execution_plan.digest,
    )
    algorithm_config: dict[str, object] = {
        "model": {
            "vocab_size": 32,
            "sequence_length": 4,
            "hidden_size": 8,
            "heads": 2,
        }
    }
    if optimizer is not None:
        algorithm_config["optimizer"] = optimizer
    runtime = TorchRuntimeContext(
        algorithm_config,
        "transformer",
        1,
        policy.digest,
        policy.execution_plan.digest,
        identity,
        input_binding_digest="1" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    return TorchBuildContext(runtime, stage)


def test_transformer_descriptor_uses_torch_runtime() -> None:
    distribution = TOKEN_TRANSFORMER_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.RAY_TRAIN_TORCH


def test_transformer_derives_padding_mask_from_input_ids() -> None:
    recipe = TokenTransformerRecipe()
    build = _context()
    batch_context = TorchBatchContext(
        build.stage,
        ("token_0", "token_1", "token_2", "token_3"),
        label_name="label",
    )
    modules = recipe.build_modules(build)
    adapted = recipe.adapt_batch(
        {
            "token_0": torch.tensor([1, 4]),
            "token_1": torch.tensor([2, 5]),
            "token_2": torch.tensor([3, 0]),
            "token_3": torch.tensor([0, 0]),
            "label": torch.tensor([1.0, 0.0]),
        },
        batch_context,
    )
    result = recipe.training_step(modules, adapted, TorchStepContext(build.stage, 0, 0))
    assert adapted.keyword["input_ids"].dtype == torch.int64
    assert result.outputs["output"].shape == (2, 1)
    assert result.loss.normalizer == 2


def test_transformer_rejects_all_padding_rows() -> None:
    recipe = TokenTransformerRecipe()
    build = _context()
    with pytest.raises(ValueError, match="non-padding"):
        recipe.adapt_batch(
            {
                "token_0": torch.tensor([0]),
                "token_1": torch.tensor([0]),
                "token_2": torch.tensor([0]),
                "token_3": torch.tensor([0]),
                "label": torch.tensor([1.0]),
            },
            TorchBatchContext(
                build.stage,
                ("token_0", "token_1", "token_2", "token_3"),
                label_name="label",
            ),
        )


@pytest.mark.parametrize("invalid_tokens", [torch.tensor([1.5]), torch.tensor([True])])
def test_transformer_rejects_non_integer_tokens(invalid_tokens: torch.Tensor) -> None:
    recipe = TokenTransformerRecipe()
    build = _context()
    with pytest.raises(ValueError, match="integer values"):
        recipe.adapt_batch(
            {
                "token_0": invalid_tokens,
                "token_1": torch.tensor([2]),
                "token_2": torch.tensor([3]),
                "token_3": torch.tensor([0]),
                "label": torch.tensor([1.0]),
            },
            TorchBatchContext(
                build.stage,
                ("token_0", "token_1", "token_2", "token_3"),
                label_name="label",
            ),
        )


def test_transformer_rejects_fractional_accumulation() -> None:
    recipe = TokenTransformerRecipe()
    build = _context({"accumulation_steps": 1.5})
    modules = recipe.build_modules(build)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        recipe.configure_optimizers(modules, build)
