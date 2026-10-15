"""Ray Train data configuration that preserves every input row.

Ray's stock :class:`ray.train.DataConfig` uses ``streaming_split(...,
equal=True)``.  That is convenient for synchronous data-parallel algorithms, but
it intentionally drops a remainder when the row count is not divisible by the
worker count.  Tributo's execution receipt promises complete input coverage, so
official algorithms use this configuration to request the lossless split.
"""

from __future__ import annotations

from typing import Any

from ray.train import DataConfig


class CompleteCoverageDataConfig(DataConfig):
    """Split Train datasets without dropping a remainder row."""

    def configure(
        self,
        datasets: dict[str, Any],
        world_size: int,
        worker_handles: list[Any] | None,
        worker_node_ids: list[str] | None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del worker_handles, kwargs
        # Ray 2.59.0 compatibility: retain its execution-option resolution
        # and locality setup, changing only streaming_split's ``equal``.
        output: list[dict[str, Any]] = [{} for _ in range(world_size)]
        for dataset_name, dataset in datasets.items():
            if dataset.name is None:
                dataset.set_name(dataset_name)

        if self._datasets_to_split == "all":
            datasets_to_split = set(datasets.keys())
        else:
            datasets_to_split = set(self._datasets_to_split)
        locality_hints = worker_node_ids if self._enable_shard_locality else None
        for name, dataset in datasets.items():
            dataset = dataset.copy(dataset)
            dataset.context.execution_options = self._resolve_execution_options(name)
            if name in datasets_to_split:
                for index, split in enumerate(
                    dataset.streaming_split(
                        world_size,
                        equal=False,
                        locality_hints=locality_hints,
                    )
                ):
                    output[index][name] = split
            else:
                for index in range(world_size):
                    output[index][name] = dataset.iterator()
        return output


__all__ = ["CompleteCoverageDataConfig"]
