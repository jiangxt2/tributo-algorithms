"""Reusable public XGBoost stage runner for official multi-stage algorithms."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, cast

import ray
from tributo.algorithms.api import AlgorithmExecutionError, WorkerResources

from tributo_algorithms_boosting.algorithm import _EvidenceCollector, _train_loop
from tributo_algorithms_boosting.data_config import CompleteCoverageDataConfig


@dataclass(frozen=True)
class XGBoostStageResult:
    """One completed distributed Booster stage and its bounded evidence."""

    name: str
    checkpoint: object
    booster_raw: bytes
    evidence: Mapping[str, object]
    row_count: int


class XGBoostStageRunner:
    """Run named Ray Train/XGBoost stages through one audited implementation."""

    def __init__(
        self,
        *,
        worker_count: int,
        resources_per_worker: WorkerResources,
        storage_path: str,
        input_binding_digest: str,
    ) -> None:
        self.worker_count = worker_count
        self.resources = resources_per_worker
        self.storage_path = storage_path
        self.input_binding_digest = input_binding_digest

    def fit(
        self,
        name: str,
        dataset: object,
        *,
        feature_names: tuple[str, ...],
        label_name: str,
        params: Mapping[str, object],
        num_boost_round: int,
    ) -> XGBoostStageResult:
        import xgboost
        from ray.train import RunConfig, ScalingConfig
        from ray.train.xgboost import XGBoostTrainer

        dataset_value = cast(Any, dataset)
        rows = int(dataset_value.count())
        if rows < self.worker_count:
            raise AlgorithmExecutionError(
                f"XGBoost stage {name!r} has insufficient rows for every Worker"
            )
        try:
            block_count = dataset_value.num_blocks()
        except NotImplementedError:
            dataset_value = dataset_value.materialize()
            block_count = dataset_value.num_blocks()
        if block_count < self.worker_count:
            dataset_value = dataset_value.repartition(
                self.worker_count,
                strict=True,
                shuffle=False,
            )
        collector_type = ray.remote(_EvidenceCollector).options(num_cpus=0)
        collector: Any = collector_type.remote()
        stage_digest = hashlib.sha256(
            f"{self.input_binding_digest}:{name}".encode("utf-8")
        ).hexdigest()
        try:
            trainer = XGBoostTrainer(
                train_loop_per_worker=_train_loop,
                train_loop_config={
                    "feature_names": list(feature_names),
                    "label_name": label_name,
                    "params": dict(params),
                    "num_boost_round": num_boost_round,
                    "evidence_actor": collector,
                    "binding_digest": stage_digest,
                },
                scaling_config=ScalingConfig(
                    num_workers=self.worker_count,
                    use_gpu=self.resources.num_gpus > 0,
                    placement_strategy="SPREAD",
                    resources_per_worker={
                        "CPU": self.resources.num_cpus,
                        "GPU": self.resources.num_gpus,
                        **dict(self.resources.custom),
                    },
                ),
                datasets={"train": dataset_value},
                dataset_config=CompleteCoverageDataConfig(datasets_to_split=["train"]),
                run_config=RunConfig(
                    name=f"tributo-xgboost-stage-{name}",
                    storage_path=self.storage_path,
                ),
            )
            result = trainer.fit()
            checkpoint = result.checkpoint
            if checkpoint is None:
                raise AlgorithmExecutionError(
                    f"XGBoost stage {name!r} did not publish a Checkpoint"
                )
            workers = ray.get(collector.snapshot.remote())
            digests = {str(item.get("model_state_digest")) for item in workers}
            if len(workers) != self.worker_count or len(digests) != 1:
                raise AlgorithmExecutionError(
                    f"XGBoost stage {name!r} evidence is incomplete"
                )
            with checkpoint.as_directory() as directory:
                booster = xgboost.Booster()
                booster.load_model(f"{directory}/model.ubj")
                booster_raw = bytes(booster.save_raw(raw_format="ubj"))
            evidence: Mapping[str, object] = {
                "workers": workers,
                "state": {
                    "coordination": "framework_native",
                    "synchronized": True,
                    "bounded": True,
                    "global_model_digest": next(iter(digests)),
                    "details": {
                        "framework": "xgboost",
                        "collective": "rabit",
                        "stage": name,
                    },
                },
                "input_complete": True,
                "expected_training_rows": rows,
            }
            return XGBoostStageResult(
                name=name,
                checkpoint=checkpoint,
                booster_raw=booster_raw,
                evidence=evidence,
                row_count=rows,
            )
        finally:
            ray.kill(collector, no_restart=True)


__all__ = ["XGBoostStageResult", "XGBoostStageRunner"]
