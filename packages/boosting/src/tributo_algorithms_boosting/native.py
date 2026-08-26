"""Official XGBoost exporters, validator, and safe native Bundle flavor."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, cast

from pydantic import BaseModel, ConfigDict, Field
from tributo.exceptions import ModelLoadError, UnsupportedArtifactFormat
from tributo.exporting.errors import sanitize_error_message
from tributo.exporting.models import (
    ArtifactDraft,
    DraftFile,
    ExportContext,
    ExportSource,
    FailureInfo,
    PlannedTarget,
    ProducerInfo,
    ResolvedArtifact,
    SupportRequest,
    SupportResult,
    ValidationResult,
    ValidatorBinding,
)
from tributo.exporting.runtime import SECURITY_MODE_SAFE, BundleModel


class XGBoostONNXOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    opset: int = Field(default=15, ge=12, le=21)


class XGBoostNativeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


class XGBoostValidationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_samples: int = Field(default=2, ge=1, le=128)


@dataclass(frozen=True)
class _NativeExplanation:
    values: Any
    data: Any
    base_values: Any
    model_outputs: Any


class _NativeTreeExplainer:
    def __init__(self, model: _XGBoostModel, feature_names: tuple[str, ...]) -> None:
        self.model = model
        self.feature_names = feature_names

    def __call__(
        self,
        values: object,
        *,
        check_additivity: bool = False,
    ) -> _NativeExplanation:
        del check_additivity
        import numpy as np

        data = np.asarray(values, dtype=np.float32)
        if data.ndim != 2 or (
            self.feature_names and len(self.feature_names) != data.shape[1]
        ):
            raise ValueError("XGBoost attribution input shape is invalid")
        feature_types = tuple(self.model.booster.feature_types or ())
        matrix = self.model.xgboost.DMatrix(
            data,
            feature_names=list(self.feature_names) or None,
            feature_types=list(feature_types) or None,
        )
        contributions = np.asarray(
            self.model.booster.predict(
                matrix,
                pred_contribs=True,
                approx_contribs=False,
                strict_shape=True,
            )
        )
        outputs = np.asarray(
            self.model.booster.predict(
                matrix,
                output_margin=True,
                strict_shape=True,
            )
        )
        expected = (data.shape[0], outputs.shape[1], data.shape[1] + 1)
        if outputs.ndim != 2 or contributions.shape != expected:
            raise ValueError("XGBoost attribution output shape is invalid")
        return _NativeExplanation(
            values=np.transpose(contributions[:, :, :-1], (0, 2, 1)),
            data=data,
            base_values=contributions[:, :, -1],
            model_outputs=outputs,
        )


class OfficialXGBoostONNXExporter:
    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "official-xgboost-onnx-v1"
    priority: ClassVar[int] = 100
    output_format: ClassVar[str] = "onnx"
    output_flavor_id: ClassVar[str] = "onnx-runtime-v1"
    source_kinds: ClassVar[tuple[str, ...]] = ("xgboost_result",)
    options_model: ClassVar[type[BaseModel]] = XGBoostONNXOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
        ValidatorBinding(validator_id="onnx-runtime-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        objective = str(request.source_metadata.get("objective", ""))
        supported = request.source_kind == "xgboost_result" and (
            objective.startswith(("binary:", "multi:"))
            or objective == "reg:squarederror"
        )
        return SupportResult(
            supported=supported,
            code="OK" if supported else "UNSUPPORTED_XGBOOST_SOURCE",
            reason=(
                "numeric XGBoost source is supported"
                if supported
                else "official XGBoost ONNX requires a supported numeric objective"
            ),
        )

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        del upstream
        import onnxmltools
        import xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        booster = cast(xgboost.Booster, source.model_object)
        feature_names = tuple(
            str(name) for name in source.feature_schema.get("feature_names", ())
        )
        if not feature_names:
            feature_names = tuple(
                f"f{index}" for index in range(int(booster.num_features()))
            )
        converted = onnxmltools.convert_xgboost(
            booster,
            initial_types=[
                ("float_input", FloatTensorType([None, len(feature_names)]))
            ],
            target_opset=int(target.typed_options.get("opset", 15)),
        )
        converted.graph.name = "tributo-official-xgboost"
        path = context.artifact_dir / "model.onnx"
        path.write_bytes(converted.SerializeToString())
        return ArtifactDraft(
            name=target.target.name,
            format="onnx",
            flavor_id="onnx-runtime-v1",
            files=(DraftFile(relative_path="model.onnx", role="model"),),
            entrypoint="model.onnx",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={
                    "xgboost": xgboost.__version__,
                    "onnxmltools": getattr(onnxmltools, "__version__", "unknown"),
                },
                effective_options=dict(target.typed_options),
            ),
        )


class OfficialXGBoostUBJExporter:
    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "official-xgboost-ubj-v1"
    priority: ClassVar[int] = 100
    output_format: ClassVar[str] = "ubj"
    output_flavor_id: ClassVar[str] = "official-xgboost-native-v1"
    source_kinds: ClassVar[tuple[str, ...]] = ("xgboost_result",)
    options_model: ClassVar[type[BaseModel]] = XGBoostNativeOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
        ValidatorBinding(
            validator_id="official-xgboost-native-runtime-v1", required=True
        ),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        supported = request.source_kind == "xgboost_result"
        return SupportResult(
            supported=supported,
            code="OK" if supported else "UNSUPPORTED_SOURCE_KIND",
            reason=(
                "XGBoost Booster source is supported"
                if supported
                else "official UBJ exporter requires xgboost_result"
            ),
        )

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        del upstream
        import xgboost

        booster = cast(xgboost.Booster, source.model_object)
        (context.artifact_dir / "model.ubj").write_bytes(
            bytes(booster.save_raw(raw_format="ubj"))
        )
        files = [DraftFile(relative_path="model.ubj", role="model")]
        feature_names = tuple(
            booster.feature_names or source.feature_schema.get("feature_names", ())
        )
        if feature_names:
            (context.artifact_dir / "feature_names.json").write_text(
                json.dumps(feature_names), encoding="utf-8"
            )
            files.append(DraftFile(relative_path="feature_names.json", role="config"))
        return ArtifactDraft(
            name=target.target.name,
            format="ubj",
            flavor_id="official-xgboost-native-v1",
            files=tuple(files),
            entrypoint="model.ubj",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={"xgboost": xgboost.__version__},
                effective_options=dict(target.typed_options),
            ),
        )


class OfficialXGBoostNativeValidator:
    api_version: ClassVar[int] = 1
    validator_id: ClassVar[str] = "official-xgboost-native-runtime-v1"
    options_model: ClassVar[type[BaseModel]] = XGBoostValidationOptions

    def validate(
        self,
        source: ExportSource,
        artifact: ResolvedArtifact,
        upstream: Mapping[str, ResolvedArtifact],
        options: BaseModel,
    ) -> ValidationResult:
        del upstream
        try:
            import numpy as np
            import xgboost

            source_booster = source.model_object
            if not isinstance(source_booster, xgboost.Booster):
                raise TypeError("official XGBoost validation requires a Booster")
            started = time.perf_counter()
            loaded = xgboost.Booster()
            loaded.load_model(str(artifact.entrypoint_path))
            features = int(loaded.num_features())
            if features != int(source_booster.num_features()) or features < 1:
                raise ValueError("reloaded Booster feature count drifted")
            values = np.zeros(
                (int(getattr(options, "num_samples", 2)), features),
                dtype=np.float32,
            )
            source_margin = source_booster.predict(
                xgboost.DMatrix(values), output_margin=True, strict_shape=True
            )
            loaded_margin = loaded.predict(
                xgboost.DMatrix(values), output_margin=True, strict_shape=True
            )
            if not np.allclose(source_margin, loaded_margin, rtol=1e-6, atol=1e-7):
                raise ValueError("reloaded Booster predictions drifted")
            return ValidationResult(
                validator_id=self.validator_id,
                status="passed",
                metrics={
                    "feature_count": float(features),
                    "load_seconds": round(time.perf_counter() - started, 6),
                },
            )
        except Exception as exc:
            return ValidationResult(
                validator_id=self.validator_id,
                status="failed",
                failure=FailureInfo(
                    code=type(exc).__name__,
                    category="validation",
                    message=sanitize_error_message(str(exc))[:4096],
                ),
            )


class OfficialXGBoostNativeFlavor:
    api_version: ClassVar[int] = 1
    flavor_id: ClassVar[str] = "official-xgboost-native-v1"
    supported_formats: ClassVar[tuple[str, ...]] = ("ubj",)
    batch_supported: ClassVar[bool] = True
    serveable: ClassVar[bool] = True
    security_mode: ClassVar[str] = SECURITY_MODE_SAFE
    signature_required: ClassVar[bool] = True
    required_dependencies: ClassVar[tuple[str, ...]] = ("xgboost",)
    operations: ClassVar[tuple[str, ...]] = (
        "prediction.batch",
        "prediction.online",
    )
    conditional_operations: ClassVar[tuple[str, ...]] = ()

    def load(
        self,
        artifact: ResolvedArtifact,
        *,
        role: str,
        unsafe: bool = False,
        architecture_id: str | None = None,
    ) -> BundleModel:
        del role, unsafe
        if architecture_id not in (None, "xgboost"):
            raise UnsupportedArtifactFormat(
                f"official XGBoost flavor cannot load {architecture_id!r}"
            )
        try:
            import xgboost
        except ImportError as exc:
            raise ModelLoadError("official XGBoost flavor requires xgboost") from exc
        booster = xgboost.Booster()
        try:
            booster.load_model(str(artifact.entrypoint_path))
        except Exception as exc:
            raise ModelLoadError(
                f"failed to load official XGBoost model ({type(exc).__name__})"
            ) from None
        sidecar = artifact.path_for("feature_names.json")
        if sidecar.is_file():
            raw_names = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(raw_names, list) or not all(
                isinstance(name, str) for name in raw_names
            ):
                raise ModelLoadError("XGBoost feature-name sidecar is invalid")
            if booster.feature_names and tuple(booster.feature_names) != tuple(
                raw_names
            ):
                raise ModelLoadError("XGBoost feature-name sidecar drifted")
            booster.feature_names = list(raw_names)
        return cast(BundleModel, _XGBoostModel(booster, xgboost))


class _XGBoostModel:
    def __init__(self, booster: Any, xgboost: Any) -> None:
        self.booster = booster
        self.xgboost = xgboost
        config = json.loads(booster.save_config())["learner"]
        self.objective = str(config["objective"]["name"])
        self.features = int(booster.num_features())
        self.classification = self.objective.startswith(("binary:", "multi:"))
        self.classes = max(2, int(config["learner_model_param"]["num_class"]))

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("float_input",)

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("label", "probabilities") if self.classification else ("prediction",)

    @property
    def input_dtypes(self) -> tuple[str, ...]:
        return ("float32",)

    @property
    def output_dtypes(self) -> tuple[str, ...]:
        return ("int64", "float32") if self.classification else ("float32",)

    @property
    def input_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return ((None, self.features),)

    @property
    def output_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        return ((None,), (None, self.classes)) if self.classification else ((None, 1),)

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        matrix = self.xgboost.DMatrix(
            np.asarray(inputs["float_input"], dtype=np.float32)
        )
        values = np.asarray(self.booster.predict(matrix), dtype=np.float32)
        if not self.classification:
            return {"prediction": values.reshape(-1, 1)}
        if self.objective.startswith("multi:"):
            probabilities = values.reshape(-1, self.classes)
        else:
            positive = values.reshape(-1)
            probabilities = np.column_stack((1.0 - positive, positive)).astype(
                np.float32, copy=False
            )
        return {
            "label": probabilities.argmax(axis=1).astype(np.int64),
            "probabilities": probabilities,
        }

    @property
    def native_attribution_id(self) -> str | None:
        config = json.loads(self.booster.save_config())["learner"]
        booster_kind = str(config["gradient_booster"]["name"])
        supported_objective = self.objective.startswith(("binary:", "multi:")) or (
            self.objective in {"reg:squarederror", "reg:logistic"}
        )
        return (
            "xgboost-tree-shap-v1"
            if booster_kind in {"gbtree", "dart"} and supported_objective
            else None
        )

    @property
    def native_model_object(self) -> Any:
        return self.booster

    @property
    def native_feature_names(self) -> tuple[str, ...]:
        return tuple(self.booster.feature_names or ())

    @property
    def native_objective(self) -> str | None:
        return self.objective

    def native_attribution_support(self, request: Any) -> Any:
        from tributo.explainability.protocols import SupportDecision

        if self.native_attribution_id is None:
            return SupportDecision(
                supported=False,
                reason="XGBoost Booster does not support exact TreeSHAP",
                backend="tree",
                exactness="exact",
            )
        if request.output_target not in {"model_output", "raw", "raw_margin"}:
            return SupportDecision(
                supported=False,
                reason="official native XGBoost attribution supports raw outputs",
                backend="tree",
                exactness="exact",
            )
        if request.output_selection == "predicted" and not self.classification:
            return SupportDecision(
                supported=False,
                reason="predicted output selection requires classification",
                backend="tree",
                exactness="exact",
            )
        return SupportDecision(supported=True, backend="tree", exactness="exact")

    def prepare_native_attribution(
        self,
        request: Any,
        *,
        feature_names: tuple[str, ...],
        reference_data: Any,
    ) -> Any:
        from tributo.explainability.protocols import PreparedExplainer

        del reference_data
        decision = self.native_attribution_support(request)
        if not decision.supported:
            raise ValueError(decision.reason)
        names = feature_names or self.native_feature_names
        if names and len(names) != self.features:
            raise ValueError("XGBoost feature names do not match model width")
        return PreparedExplainer(
            backend="tree",
            exactness="exact",
            explain=_NativeTreeExplainer(self, names),
            feature_names=names,
        )


__all__ = [
    "OfficialXGBoostNativeFlavor",
    "OfficialXGBoostNativeValidator",
    "OfficialXGBoostONNXExporter",
    "OfficialXGBoostUBJExporter",
]
