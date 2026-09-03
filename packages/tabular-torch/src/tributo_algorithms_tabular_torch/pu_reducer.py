"""Wheel-owned global nnPU/uPU loss reduction."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tributo.algorithms.api.torch_runtime import (
    TorchCompositeGlobalState,
    TorchGlobalLossContext,
    TorchGlobalLossReduction,
    TorchMetricContribution,
)


@dataclass(frozen=True)
class PURiskReducerPlan:
    """Immutable PU risk parameters shared by the Recipe and reducer."""

    mode: str = "nnpu"
    class_prior: float = 0.5
    beta: float = 0.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"nnpu", "upu"}:
            raise ValueError("PU risk mode must be nnpu or upu")
        if not 0.0 < self.class_prior < 1.0 or not math.isfinite(self.class_prior):
            raise ValueError("PU class_prior must be finite and in (0, 1)")
        if self.beta < 0.0 or not math.isfinite(self.beta):
            raise ValueError("PU beta must be finite and non-negative")
        if not 0.0 <= self.gamma <= 1.0 or not math.isfinite(self.gamma):
            raise ValueError("PU gamma must be finite and in [0, 1]")


class PUGlobalLossReducer:
    """Reduce PU components without performing collectives or I/O."""

    api_version = 1
    reducer_id = "tributo.official.tabular_torch.pu-risk"
    component_schema_id = "tributo.official.tabular_torch.pu-risk-components.v1"
    code_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def reduce(
        self,
        config: Mapping[str, object],
        global_state: TorchCompositeGlobalState,
        context: TorchGlobalLossContext,
    ) -> TorchGlobalLossReduction:
        del context
        loss_config = config.get("loss", {})
        if not isinstance(loss_config, Mapping):
            return TorchGlobalLossReduction(
                "rejected", failure_code="pu.invalid_config"
            )
        try:
            prior = float(loss_config["class_prior"])
            beta = float(loss_config.get("beta", 0.0))
            gamma = float(loss_config.get("gamma", 1.0))
            mode = str(loss_config.get("type", "nnpu"))
            positive_count = float(global_state.normalizers["positive_count"])
            unlabeled_count = float(global_state.normalizers["unlabeled_count"])
            positive_loss = float(global_state.components["positive_loss_sum"])
            positive_as_negative = float(
                global_state.components["positive_as_negative_sum"]
            )
            unlabeled_negative = float(
                global_state.components["unlabeled_negative_sum"]
            )
        except (KeyError, TypeError, ValueError):
            return TorchGlobalLossReduction(
                "rejected", failure_code="pu.invalid_components"
            )
        try:
            risk_plan = PURiskReducerPlan(
                mode=mode,
                class_prior=prior,
                beta=beta,
                gamma=gamma,
            )
        except ValueError:
            return TorchGlobalLossReduction(
                "rejected", failure_code="pu.invalid_config"
            )
        if positive_count <= 0 or unlabeled_count <= 0:
            return TorchGlobalLossReduction("rejected", failure_code="pu.empty_group")
        negative_risk = (
            unlabeled_negative / unlabeled_count
            - prior * positive_as_negative / positive_count
        )
        correction = risk_plan.mode == "nnpu" and negative_risk < -risk_plan.beta
        if correction:
            coefficients = {
                "positive_loss_sum": 0.0,
                "positive_as_negative_sum": risk_plan.gamma
                * risk_plan.class_prior
                / positive_count,
                "unlabeled_negative_sum": -risk_plan.gamma / unlabeled_count,
            }
            branch = "nnpu_correction"
            risk = -risk_plan.gamma * negative_risk
        else:
            coefficients = {
                "positive_loss_sum": risk_plan.class_prior / positive_count,
                "positive_as_negative_sum": -risk_plan.class_prior / positive_count,
                "unlabeled_negative_sum": 1.0 / unlabeled_count,
            }
            branch = "upu" if mode == "upu" else "nnpu_normal"
            risk = (
                risk_plan.class_prior * positive_loss / positive_count + negative_risk
            )
        return TorchGlobalLossReduction(
            "accepted",
            coefficients=coefficients,
            branch=branch,
            evidence={
                "mode": mode,
                "class_prior": risk_plan.class_prior,
                "beta": risk_plan.beta,
                "gamma": risk_plan.gamma,
            },
            metrics={"train_loss": TorchMetricContribution(risk, 1.0)},
        )


__all__ = ["PUGlobalLossReducer", "PURiskReducerPlan"]
