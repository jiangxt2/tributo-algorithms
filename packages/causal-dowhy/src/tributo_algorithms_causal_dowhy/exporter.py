"""DoWhy GCM report exporter."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from tributo.exporting.models import SupportRequest, SupportResult, ValidatorBinding
from tributo.training.exporters.causal_report import CausalReportExporter


class GCMReportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GCMReportExporter(CausalReportExporter):
    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "official-causal-gcm-report-v1"
    output_format: ClassVar[str] = "json"
    output_flavor_id: ClassVar[str] = "report"
    artifact_kind: ClassVar[str] = "report"
    priority: ClassVar[int] = 90
    source_kinds: ClassVar[tuple[str, ...]] = ("causal_gcm", "prebuilt_onnx")
    options_model: ClassVar[type[BaseModel]] = GCMReportOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        supported = request.source_kind in cls.source_kinds
        return SupportResult(
            supported=supported,
            code="OK" if supported else "UNSUPPORTED_SOURCE_KIND",
            reason=(
                "causal GCM report source is supported"
                if supported
                else "GCM report exporter requires causal_gcm source"
            ),
        )


__all__ = ["GCMReportExporter"]
