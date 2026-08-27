"""CatBoost native model exporter plugin."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict
from tributo.exporting.models import (
    ArtifactDraft,
    DraftFile,
    ExportContext,
    ExportSource,
    PlannedTarget,
    ProducerInfo,
    ResolvedArtifact,
    SupportRequest,
    SupportResult,
    ValidatorBinding,
)

from tributo_algorithms_catboost.algorithm import CatBoostModel


class CatBoostExportOptions(BaseModel):
    """No optional fields are exposed until native parity is validated."""

    model_config = ConfigDict(extra="forbid")


class CatBoostNativeExporter:
    """Write the CatBoost binary model and its typed metadata."""

    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "catboost-native-v1"
    priority: ClassVar[int] = 100
    output_format: ClassVar[str] = "catboost"
    output_flavor_id: ClassVar[str] = "catboost-native-v1"
    source_kinds: ClassVar[tuple[str, ...]] = ("catboost_result",)
    options_model: ClassVar[type[BaseModel]] = CatBoostExportOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        if request.source_kind != "catboost_result":
            return SupportResult(
                supported=False,
                code="UNSUPPORTED_SOURCE_KIND",
                reason="CatBoost exporter requires source_kind='catboost_result'",
            )
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        del upstream
        import catboost

        model = source.model_object
        if not isinstance(model, CatBoostModel):
            raise TypeError("CatBoost native exporter requires CatBoostModel")
        model_path = context.artifact_dir / "model.cbm"
        config_path = context.artifact_dir / "config.json"
        cast(Any, model.model).save_model(str(model_path), format="cbm")
        raw_classes = getattr(model.model, "classes_", ())
        classes = model.classes or tuple(
            raw_classes.tolist() if hasattr(raw_classes, "tolist") else raw_classes
        )
        class_count = len(classes) or 2
        config_path.write_text(
            json.dumps(
                {
                    "cat_features": list(model.cat_features),
                    "classes": list(classes),
                    "feature_names": list(model.feature_names),
                    "task": model.task,
                    "class_count": class_count,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return ArtifactDraft(
            name=target.target.name,
            format=self.output_format,
            flavor_id=self.output_flavor_id,
            files=(
                DraftFile(relative_path="model.cbm", role="model"),
                DraftFile(relative_path="config.json", role="config"),
            ),
            entrypoint="model.cbm",
            producer=ProducerInfo(
                exporter_id=self.exporter_id,
                framework_versions={"catboost": catboost.__version__},
            ),
        )


__all__ = ["CatBoostNativeExporter"]
