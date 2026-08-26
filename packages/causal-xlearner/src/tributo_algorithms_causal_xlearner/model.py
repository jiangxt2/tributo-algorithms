"""Fixed-composition X-Learner model and prediction semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np

STAGES = ("mu0", "mu1", "tau0", "tau1", "propensity")
FORMULA = "cate=propensity*tau0+(1-propensity)*tau1"
QUADRANT_CODES = {
    "low_response_low_uplift": 0,
    "low_response_high_uplift": 1,
    "high_response_low_uplift": 2,
    "high_response_high_uplift": 3,
}
STAGE_OBJECTIVES = {
    "mu0": "reg:squarederror",
    "mu1": "reg:squarederror",
    "tau0": "reg:squarederror",
    "tau1": "reg:squarederror",
    "propensity": "binary:logistic",
}


@dataclass(frozen=True)
class XLearnerPrediction:
    mu0: np.ndarray
    mu1: np.ndarray
    tau0: np.ndarray
    tau1: np.ndarray
    propensity: np.ndarray
    cate: np.ndarray
    quadrant: np.ndarray


class XLearnerModel:
    def __init__(
        self,
        boosters: Mapping[str, object],
        *,
        feature_names: tuple[str, ...],
        response_threshold: float,
        propensity_clip: tuple[float, float],
    ) -> None:
        if set(boosters) != set(STAGES):
            raise ValueError("X-Learner requires exactly five Booster stages")
        self.boosters = dict(boosters)
        self.feature_names = feature_names
        self.response_threshold = response_threshold
        self.propensity_clip = propensity_clip

    @classmethod
    def from_raw(
        cls,
        raw: Mapping[str, bytes],
        **kwargs: Any,
    ) -> XLearnerModel:
        import xgboost

        boosters = {}
        for stage in STAGES:
            booster = xgboost.Booster()
            booster.load_model(bytearray(raw[stage]))
            boosters[stage] = booster
        return cls(boosters, **kwargs)

    def predict(self, features: object) -> XLearnerPrediction:
        import xgboost

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("X-Learner input shape does not match feature_names")
        matrix = xgboost.DMatrix(values)
        predictions = {
            stage: np.asarray(cast(Any, self.boosters[stage]).predict(matrix))
            for stage in STAGES
        }
        propensity = np.clip(
            predictions["propensity"], self.propensity_clip[0], self.propensity_clip[1]
        )
        cate = (
            propensity * predictions["tau0"] + (1.0 - propensity) * predictions["tau1"]
        )
        response = np.maximum(predictions["mu0"], predictions["mu1"])
        high_response = response >= self.response_threshold
        high_uplift = cate >= 0.0
        quadrant = np.where(
            high_response,
            np.where(
                high_uplift, "high_response_high_uplift", "high_response_low_uplift"
            ),
            np.where(
                high_uplift, "low_response_high_uplift", "low_response_low_uplift"
            ),
        )
        return XLearnerPrediction(
            mu0=predictions["mu0"],
            mu1=predictions["mu1"],
            tau0=predictions["tau0"],
            tau1=predictions["tau1"],
            propensity=propensity,
            cate=cate,
            quadrant=quadrant,
        )


__all__ = [
    "FORMULA",
    "QUADRANT_CODES",
    "STAGES",
    "STAGE_OBJECTIVES",
    "XLearnerModel",
    "XLearnerPrediction",
]
