"""Safe CatBoost native Bundle model flavor."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np
from tributo.exceptions import ModelLoadError
from tributo.exporting.models import ResolvedArtifact
from tributo.exporting.runtime import SECURITY_MODE_SAFE, BundleModel


class CatBoostNativeFlavor:
    """Load a CatBoost binary model without pickle or remote code."""

    api_version: ClassVar[int] = 1
    flavor_id: ClassVar[str] = "catboost-native-v1"
    supported_formats: ClassVar[tuple[str, ...]] = ("catboost",)
    batch_supported: ClassVar[bool] = True
    serveable: ClassVar[bool] = True
    security_mode: ClassVar[str] = SECURITY_MODE_SAFE
    signature_required: ClassVar[bool] = True
    required_dependencies: ClassVar[tuple[str, ...]] = ("catboost",)
    operations: ClassVar[tuple[str, ...]] = ("prediction.batch", "prediction.online")
    conditional_operations: ClassVar[tuple[str, ...]] = ()

    def load(
        self,
        artifact: ResolvedArtifact,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        del role, unsafe, architecture_id
        try:
            from catboost import CatBoost
        except ImportError as exc:
            raise ModelLoadError("catboost is required for catboost-native-v1") from exc
        try:
            raw_metadata = json.loads(
                artifact.path_for("config.json").read_text(encoding="utf-8")
            )
            if not isinstance(raw_metadata, dict):
                raise ValueError("CatBoost native metadata is not an object")
            raw_feature_names = raw_metadata.get("feature_names")
            raw_cat_features = raw_metadata.get("cat_features", [])
            raw_classes = raw_metadata.get("classes", [])
            raw_task = raw_metadata.get("task")
            raw_class_count = raw_metadata.get("class_count", 2)
            if not isinstance(raw_feature_names, list) or not all(
                isinstance(name, str) and name for name in raw_feature_names
            ):
                raise ValueError("CatBoost native feature metadata is invalid")
            if not isinstance(raw_cat_features, list) or any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in raw_cat_features
            ):
                raise ValueError("CatBoost native categorical metadata is invalid")
            if not isinstance(raw_classes, list):
                raise ValueError("CatBoost native class metadata is invalid")
            if (
                not isinstance(raw_task, str)
                or not isinstance(raw_class_count, int)
                or isinstance(raw_class_count, bool)
            ):
                raise ValueError("CatBoost native task metadata is invalid")
            model = CatBoost()
            model.load_model(str(artifact.entrypoint_path))
        except Exception as exc:
            raise ModelLoadError(
                "CatBoost native artifact could not be loaded"
            ) from exc
        return _CatBoostRuntimeModel(
            model,
            tuple(str(name) for name in raw_feature_names),
            raw_task,
            raw_class_count,
            tuple(raw_cat_features),
            tuple(raw_classes),
        )


class _CatBoostRuntimeModel:
    def __init__(
        self,
        model: Any,
        feature_names: tuple[str, ...],
        task: str,
        class_count: int,
        cat_features: tuple[int, ...],
        classes: tuple[object, ...],
    ) -> None:
        if not feature_names or task not in {"classification", "regression"}:
            raise ModelLoadError("CatBoost native metadata is incomplete")
        if any(index < 0 or index >= len(feature_names) for index in cat_features):
            raise ModelLoadError("CatBoost native categorical metadata is invalid")
        if len(set(cat_features)) != len(cat_features):
            raise ModelLoadError("CatBoost native categorical metadata is duplicated")
        if task == "classification" and class_count < 2:
            raise ModelLoadError("CatBoost native class metadata is invalid")
        if task == "classification" and len(classes) != class_count:
            raise ModelLoadError("CatBoost native class metadata is incomplete")
        try:
            if len(set(classes)) != len(classes):
                raise ModelLoadError("CatBoost native class metadata is duplicated")
        except TypeError as exc:
            raise ModelLoadError(
                "CatBoost native class metadata is not scalar"
            ) from exc
        self._model = model
        self._feature_names = feature_names
        self._task = task
        self._class_count = class_count
        self._cat_features = cat_features
        self._classes = classes

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("float_input",)

    @property
    def output_names(self) -> tuple[str, ...]:
        return (
            ("label", "probabilities")
            if self._task == "classification"
            else ("variable",)
        )

    @property
    def input_dtypes(self) -> tuple[str, ...]:
        return ("object" if self._cat_features else "float32",)

    @property
    def output_dtypes(self) -> tuple[str, ...]:
        return ("int64", "float32") if self._task == "classification" else ("float32",)

    @property
    def input_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return ((None, len(self._feature_names)),)

    @property
    def output_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return (
            ((None,), (None, self._class_count))
            if self._task == "classification"
            else ((None, 1),)
        )

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(inputs) != {"float_input"}:
            raise ModelLoadError("CatBoost model requires exactly float_input")
        values = np.asarray(inputs["float_input"])
        if values.ndim != 2 or values.shape[1] != len(self._feature_names):
            raise ModelLoadError("CatBoost input shape does not match feature metadata")
        if self._task == "classification":
            raw_labels = np.asarray(
                self._model.predict(values, prediction_type="Class")
            ).reshape(-1)
            class_indices = {value: index for index, value in enumerate(self._classes)}
            try:
                labels = np.asarray(
                    [
                        class_indices[
                            value.item() if isinstance(value, np.generic) else value
                        ]
                        for value in raw_labels
                    ],
                    dtype=np.int64,
                )
            except (KeyError, TypeError) as exc:
                raise ModelLoadError(
                    "CatBoost prediction contains an unknown class"
                ) from exc
            return {
                "label": labels,
                "probabilities": np.asarray(
                    self._model.predict(values, prediction_type="Probability"),
                    dtype=np.float32,
                ),
            }
        return {
            "variable": np.asarray(
                self._model.predict(values, prediction_type="RawFormulaVal"),
                dtype=np.float32,
            ).reshape(-1, 1)
        }


__all__ = ["CatBoostNativeFlavor"]
