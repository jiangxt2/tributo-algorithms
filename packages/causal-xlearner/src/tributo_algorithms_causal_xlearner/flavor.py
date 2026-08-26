"""Safe fixed-composition X-Learner Bundle flavor."""

from __future__ import annotations

import json
from typing import Any, ClassVar, cast

import numpy as np
from tributo.exceptions import ModelLoadError, UnsupportedArtifactFormat
from tributo.exporting.models import ResolvedArtifact
from tributo.exporting.runtime import SECURITY_MODE_SAFE, BundleModel

from tributo_algorithms_causal_xlearner.model import (
    FORMULA,
    QUADRANT_CODES,
    STAGE_OBJECTIVES,
    STAGES,
    XLearnerModel,
)


def _objective(booster: object) -> str:
    config = json.loads(cast(Any, booster).save_config())
    return str(config["learner"]["objective"]["name"])


class XLearnerFlavor:
    api_version: ClassVar[int] = 1
    flavor_id: ClassVar[str] = "official-x-learner-v1"
    supported_formats: ClassVar[tuple[str, ...]] = ("x-learner",)
    batch_supported: ClassVar[bool] = True
    serveable: ClassVar[bool] = False
    security_mode: ClassVar[str] = SECURITY_MODE_SAFE
    signature_required: ClassVar[bool] = True
    required_dependencies: ClassVar[tuple[str, ...]] = ("xgboost",)

    def load(
        self,
        artifact: ResolvedArtifact,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        del role, unsafe
        if architecture_id not in (None, "x_learner"):
            raise UnsupportedArtifactFormat(
                "official-x-learner-v1 requires x_learner architecture"
            )
        try:
            import xgboost

            metadata = json.loads(artifact.entrypoint_path.read_text(encoding="utf-8"))
            if metadata.get("api_version") != 1 or metadata.get("formula") != FORMULA:
                raise ValueError("metadata contract")
            if metadata.get("quadrant_codes") != QUADRANT_CODES:
                raise ValueError("quadrant contract")
            components = metadata.get("components")
            if not isinstance(components, dict) or set(components) != set(STAGES):
                raise ValueError("component set")
            boosters = {}
            for stage in STAGES:
                booster = xgboost.Booster()
                booster.load_model(str(artifact.path_for(components[stage])))
                if _objective(booster) != STAGE_OBJECTIVES[stage]:
                    raise ValueError(f"component objective {stage}")
                boosters[stage] = booster
            model = XLearnerModel(
                boosters,
                feature_names=tuple(metadata["feature_names"]),
                response_threshold=float(metadata["response_threshold"]),
                propensity_clip=tuple(metadata["propensity_clip"]),
            )
        except Exception as exc:
            raise ModelLoadError(
                f"official X-Learner artifact is invalid ({type(exc).__name__})"
            ) from None
        return _XLearnerBundleModel(model)


class _XLearnerBundleModel:
    input_names = ("float_input",)
    output_names = ("mu0", "mu1", "tau0", "tau1", "propensity", "cate", "quadrant")
    input_dtypes = ("float32",)
    output_dtypes = (
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "float32",
        "int64",
    )

    def __init__(self, model: XLearnerModel) -> None:
        self.model = model

    @property
    def input_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return ((None, len(self.model.feature_names)),)

    @property
    def output_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return tuple((None,) for _ in self.output_names)

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = self.model.predict(inputs["float_input"])
        outputs = {
            name: np.asarray(getattr(result, name), dtype=np.float32)
            for name in self.output_names
            if name != "quadrant"
        }
        outputs["quadrant"] = np.asarray(
            [QUADRANT_CODES[str(value)] for value in result.quadrant],
            dtype=np.int64,
        )
        return outputs


__all__ = ["XLearnerFlavor"]
