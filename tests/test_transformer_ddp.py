"""Real CPU DDP regression for an exhausted Transformer rank."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tributo_algorithms_transformers_nlp.recipe import _model


def _batch(rank: int, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = ((2, 1, 3) if rank == 0 else (1, 0, 0))[index]
    tokens = torch.tensor([1 + index + 7 * rank, 2, 3, 0]).repeat(rows, 1)
    return tokens, torch.full((rows, 1), float(rank))


def _model_factory() -> torch.nn.Module:
    return cast(
        torch.nn.Module,
        _model(vocab_size=32, sequence_length=4, hidden_size=8, heads=2),
    )


def _ddp_worker(rank: int, directory: str) -> None:
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    reports = []
    try:
        for accumulation in (1, 2):
            torch.manual_seed(1234)
            raw = _model_factory()
            initial = {
                name: value.detach().clone() for name, value in raw.state_dict().items()
            }
            model = DistributedDataParallel(raw)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            local_rows = empty_steps = optimizer_steps = 0
            for index in range(3):
                values, labels = _batch(rank, index)
                local_rows += len(values)
                empty_steps += int(len(values) == 0)
                boundary = (index + 1) % accumulation == 0 or index == 2
                with nullcontext() if boundary else model.no_sync():
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        model(values), labels, reduction="sum"
                    ).backward()
                if boundary:
                    denominator = torch.tensor(local_rows, dtype=torch.float64)
                    dist.all_reduce(denominator)
                    assert denominator.item() > 0
                    for name, parameter in raw.named_parameters():
                        assert parameter.grad is not None, name
                        assert torch.isfinite(parameter.grad).all(), name
                        parameter.grad.mul_(2.0 / denominator.item())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    local_rows = 0

            reference = _model_factory()
            reference.load_state_dict(initial)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
            for start in range(0, 3, accumulation):
                batches = [
                    _batch(owner, index)
                    for index in range(start, min(3, start + accumulation))
                    for owner in (0, 1)
                ]
                torch.nn.functional.binary_cross_entropy_with_logits(
                    reference(torch.cat([values for values, _ in batches])),
                    torch.cat([labels for _, labels in batches]),
                    reduction="mean",
                ).backward()
                reference_optimizer.step()
                reference_optimizer.zero_grad(set_to_none=True)
            max_error = 0.0
            for name, value in raw.state_dict().items():
                expected = reference.state_dict()[name]
                torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)
                max_error = max(max_error, float((value - expected).abs().max()))
            state = torch.cat(
                [value.detach().reshape(-1) for value in raw.parameters()]
            )
            states = [torch.empty_like(state) for _ in range(2)]
            dist.all_gather(states, state)
            torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
            assert empty_steps == (0 if rank == 0 else 2)
            assert optimizer_steps == (3 if accumulation == 1 else 2)
            reports.append(
                {
                    "accumulation": accumulation,
                    "empty_steps": empty_steps,
                    "optimizer_steps": optimizer_steps,
                    "max_baseline_error": max_error,
                }
            )
    finally:
        dist.destroy_process_group()
    (root / f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "pid": os.getpid(), "runs": reports})
    )


def test_transformer_empty_rank_matches_unsharded_baseline(tmp_path: Path) -> None:
    assert dist.is_available() and dist.is_gloo_available()
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=_ddp_worker, args=(rank, str(tmp_path)))
        for rank in (0, 1)
    ]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 60
        for worker in workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        assert all(not worker.is_alive() for worker in workers), "DDP timed out"
        assert [worker.exitcode for worker in workers] == [0, 0]
        reports = [
            json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)
        ]
        assert [report["rank"] for report in reports] == [0, 1]
        assert [report["pid"] for report in reports] == [
            worker.pid for worker in workers
        ]
        assert len({report["pid"] for report in reports}) == 2
        for report in reports:
            assert [run["accumulation"] for run in report["runs"]] == [1, 2]
        print(json.dumps({"ddp_workers": reports}), flush=True)
    finally:
        for worker in workers:
            if worker.pid is not None and worker.is_alive():
                worker.kill()
        for worker in workers:
            if worker.pid is not None:
                worker.join(timeout=5)
                assert not worker.is_alive(), "DDP child was not reaped"
