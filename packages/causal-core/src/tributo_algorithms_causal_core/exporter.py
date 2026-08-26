"""Causal report Bundle exporter owned by the official causal Wheel."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from tributo.exporting.models import (
    SupportRequest,
    SupportResult,
    ValidatorBinding,
)
from tributo.training.exporters.causal_report import CausalReportExporter


class CausalReportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OfficialCausalReportExporter(CausalReportExporter):
    """Export versioned causal study metadata as a Bundle report role."""

    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "official-causal-report-v1"
    output_format: ClassVar[str] = "json"
    output_flavor_id: ClassVar[str] = "report"
    artifact_kind: ClassVar[str] = "report"
    priority: ClassVar[int] = 90
    source_kinds: ClassVar[tuple[str, ...]] = ("prebuilt_onnx",)
    options_model: ClassVar[type[BaseModel]] = CausalReportOptions
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(
            supported=request.source_kind == "prebuilt_onnx",
            code=(
                "OK"
                if request.source_kind == "prebuilt_onnx"
                else "UNSUPPORTED_SOURCE_KIND"
            ),
            reason=(
                "causal report source is supported"
                if request.source_kind == "prebuilt_onnx"
                else "causal report requires a prebuilt causal ONNX source"
            ),
        )


__all__ = ["OfficialCausalReportExporter"]
