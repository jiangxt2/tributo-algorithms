"""X-Learner composite and causal-report Bundle exporters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ResolvedAlgorithmPlan,
)
from tributo.exporting.models import (
    ArtifactDraft,
    BundleOutputConfig,
    CheckpointField,
    DraftFile,
    ExportCheckpointV1,
    ExportContext,
    ExportSource,
    ExportTarget,
    PlannedTarget,
    ProducerInfo,
    ResolvedArtifact,
    SupportRequest,
    SupportResult,
    ValidatorBinding,
)
from tributo.exporting.service import BundleExportService
from tributo.training.exporters.causal_report import CausalReportExporter

from tributo_algorithms_causal_xlearner.algorithm import XLearnerResult
from tributo_algorithms_causal_xlearner.model import FORMULA, QUADRANT_CODES, STAGES


class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid")


class XLearnerExporter:
    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "official-x-learner-v1"
    output_format: ClassVar[str] = "x-learner"
    output_flavor_id: ClassVar[str] = "official-x-learner-v1"
    priority: ClassVar[int] = 90
    source_kinds: ClassVar[tuple[str, ...]] = ("official_x_learner",)
    options_model: ClassVar[type[BaseModel]] = _Options
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        supported = request.source_kind == "official_x_learner"
        return SupportResult(
            supported=supported,
            code="OK" if supported else "UNSUPPORTED_SOURCE_KIND",
            reason="official X-Learner source" if supported else "wrong source kind",
        )

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        del upstream
        result = source.model_object
        if not isinstance(result, XLearnerResult):
            raise TypeError("official X-Learner exporter requires XLearnerResult")
        files = []
        for stage in STAGES:
            name = f"{stage}.ubj"
            (context.artifact_dir / name).write_bytes(result.booster_raw[stage])
            files.append(DraftFile(relative_path=name, role="model"))
        metadata = {
            "api_version": 1,
            "feature_names": list(result.feature_names),
            "response_threshold": result.response_threshold,
            "propensity_clip": list(result.propensity_clip),
            "components": {stage: f"{stage}.ubj" for stage in STAGES},
            "formula": FORMULA,
            "quadrant_codes": QUADRANT_CODES,
            "composition_digest": result.composition_digest,
        }
        (context.artifact_dir / "x_learner.json").write_text(
            json.dumps(metadata, sort_keys=True), encoding="utf-8"
        )
        files.append(DraftFile(relative_path="x_learner.json", role="config"))
        return ArtifactDraft(
            name=target.target.name,
            format=self.output_format,
            flavor_id=self.output_flavor_id,
            files=tuple(files),
            entrypoint="x_learner.json",
            producer=ProducerInfo(exporter_id=self.exporter_id),
        )


class XLearnerONNXExporter:
    """Export the five XGBoost stages and CATE formula as one ONNX graph."""

    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "official-x-learner-onnx-v1"
    output_format: ClassVar[str] = "onnx"
    output_flavor_id: ClassVar[str] = "onnx-runtime-v1"
    priority: ClassVar[int] = 90
    source_kinds: ClassVar[tuple[str, ...]] = ("official_x_learner",)
    options_model: ClassVar[type[BaseModel]] = _Options
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = (
        ValidatorBinding(validator_id="structure-v1", required=True),
        ValidatorBinding(validator_id="onnx-runtime-v1", required=True),
    )
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        supported = request.source_kind == "official_x_learner"
        return SupportResult(
            supported=supported,
            code="OK" if supported else "UNSUPPORTED_SOURCE_KIND",
            reason="official X-Learner ONNX graph"
            if supported
            else "wrong source kind",
        )

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        del upstream
        import copy

        import onnx
        import onnxmltools
        import xgboost
        from onnx import TensorProto, helper
        from onnxmltools.convert.common.data_types import FloatTensorType

        result = source.model_object
        if not isinstance(result, XLearnerResult):
            raise TypeError("official X-Learner ONNX exporter requires XLearnerResult")
        converted = []
        for stage in STAGES:
            booster = xgboost.Booster()
            booster.load_model(bytearray(result.booster_raw[stage]))
            converted.append(
                onnxmltools.convert_xgboost(
                    booster,
                    initial_types=[
                        (
                            "float_input",
                            FloatTensorType([None, len(result.feature_names)]),
                        )
                    ],
                    target_opset=15,
                )
            )
        nodes = []
        for stage, model in zip(STAGES, converted, strict=True):
            if len(model.graph.node) != 1:
                raise AlgorithmExecutionError(
                    "X-Learner ONNX conversion produced an unexpected graph"
                )
            node = copy.deepcopy(model.graph.node[0])
            node.name = stage
            node.output[0] = stage
            nodes.append(node)

        def constant(name: str, value: float) -> Any:
            return helper.make_node(
                "Constant",
                [],
                [name],
                name=name,
                value=helper.make_tensor(
                    f"{name}_value", TensorProto.FLOAT, [], [value]
                ),
            )

        nodes.extend(
            [
                constant("clip_min", result.propensity_clip[0]),
                constant("clip_max", result.propensity_clip[1]),
                helper.make_node(
                    "Cast",
                    ["propensity"],
                    ["propensity_float"],
                    name="propensity_float",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Clip",
                    ["propensity_float", "clip_min", "clip_max"],
                    ["propensity_clipped"],
                    name="propensity_clip",
                ),
                constant("one", 1.0),
                helper.make_node(
                    "Sub", ["one", "propensity_clipped"], ["one_minus_propensity"]
                ),
                helper.make_node(
                    "Mul", ["propensity_clipped", "tau0"], ["treated_effect"]
                ),
                helper.make_node(
                    "Mul", ["one_minus_propensity", "tau1"], ["control_effect"]
                ),
                helper.make_node("Add", ["treated_effect", "control_effect"], ["cate"]),
            ]
        )
        graph = helper.make_graph(
            nodes,
            "tributo_x_learner",
            [
                helper.make_tensor_value_info(
                    "float_input",
                    TensorProto.FLOAT,
                    [None, len(result.feature_names)],
                )
            ],
            [helper.make_tensor_value_info("cate", TensorProto.FLOAT, [None, 1])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_operatorsetid("", 15),
                helper.make_operatorsetid("ai.onnx.ml", 3),
            ],
            producer_name="tributo-algorithms-causal-xlearner",
        )
        onnx.checker.check_model(model)
        (context.artifact_dir / "model.onnx").write_bytes(model.SerializeToString())
        return ArtifactDraft(
            name=target.target.name,
            format=self.output_format,
            flavor_id=self.output_flavor_id,
            files=(DraftFile(relative_path="model.onnx", role="model"),),
            entrypoint="model.onnx",
            producer=ProducerInfo(exporter_id=self.exporter_id),
        )


class XLearnerReportExporter(CausalReportExporter):
    api_version: ClassVar[int] = 2
    exporter_id: ClassVar[str] = "official-x-learner-report-v1"
    output_format: ClassVar[str] = "json"
    output_flavor_id: ClassVar[str] = "report"
    artifact_kind: ClassVar[str] = "report"
    priority: ClassVar[int] = 90
    source_kinds: ClassVar[tuple[str, ...]] = ("official_x_learner",)
    options_model: ClassVar[type[BaseModel]] = _Options
    validator_bindings: ClassVar[tuple[ValidatorBinding, ...]] = ()
    mutates_source: ClassVar[bool] = False
    upstream_requirements: ClassVar[tuple[Any, ...]] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        supported = request.source_kind == "official_x_learner"
        return SupportResult(
            supported=supported,
            code="OK" if supported else "UNSUPPORTED_SOURCE_KIND",
            reason="official X-Learner report" if supported else "wrong source kind",
        )

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: Mapping[str, ResolvedArtifact],
        target: PlannedTarget,
    ) -> ArtifactDraft:
        # Keep the report target self-contained if a caller constructs an
        # ExportSource without copying the study metadata.  The canonical
        # X-Learner path already supplies this field; this fallback preserves
        # the report contract for direct/plugin callers.
        if source.metadata.get("causal_study") is None:
            result = source.model_object
            if not isinstance(result, XLearnerResult):
                raise TypeError("X-Learner report requires XLearnerResult")
            source = source.model_copy(
                update={
                    "metadata": {
                        **source.metadata,
                        "causal_study": {
                            "api_version": 1,
                            "kind": "causal_meta_learner",
                            "method": "distributed_x_learner",
                            "effect": dict(result.metrics),
                            "component_stages": list(STAGES),
                            "composition_digest": result.composition_digest,
                            "formula": FORMULA,
                        },
                    }
                }
            )
        return super().export(context, source, upstream, target)


def export_result(
    *, result: object, checkpoint: object, plan: ResolvedAlgorithmPlan, run_id: str
) -> AlgorithmExecutionResult:
    del checkpoint
    if not isinstance(result, XLearnerResult):
        raise AlgorithmExecutionError("X-Learner export requires XLearnerResult")
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError("X-Learner output.bundle_uri is required")
    report = {
        "api_version": 1,
        "kind": "causal_meta_learner",
        "method": "distributed_x_learner",
        "effect": dict(result.metrics),
        "component_stages": list(STAGES),
        "composition_digest": result.composition_digest,
        "formula": FORMULA,
    }
    source = ExportSource(
        source_kind="official_x_learner",
        model_object=result,
        architecture_id="x_learner",
        feature_schema={"feature_names": list(result.feature_names)},
        metadata={"causal_study": report, "framework": "xgboost"},
        source_fingerprint=result.composition_digest,
        checkpoint_contract=ExportCheckpointV1(
            trainer_type="x_learner",
            architecture_id="x_learner",
            input_schema=(
                CheckpointField(
                    name="float_input",
                    dtype="float32",
                    shape=("batch", len(result.feature_names)),
                ),
            ),
            output_schema=tuple(
                CheckpointField(
                    name=name,
                    dtype="int64" if name == "quadrant" else "float32",
                    shape=("batch",),
                )
                for name in (
                    "mu0",
                    "mu1",
                    "tau0",
                    "tau1",
                    "propensity",
                    "cate",
                    "quadrant",
                )
            ),
            task_type="causal_effect_estimation",
            framework="xgboost",
            framework_version="2.1+",
        ),
    )
    bundle = BundleExportService().export_bundle(
        source,
        BundleOutputConfig(
            bundle_uri=str(output["bundle_uri"]),
            request_id=run_id,
            run_id=run_id,
            targets=[
                ExportTarget(
                    name="x-learner-onnx",
                    format="onnx",
                    exporter_id="official-x-learner-onnx-v1",
                ),
                ExportTarget(
                    name="x-learner-model",
                    format="x-learner",
                    exporter_id="official-x-learner-v1",
                ),
                ExportTarget(
                    name="causal-report",
                    format="json",
                    exporter_id="official-x-learner-report-v1",
                ),
            ],
            roles={"inference": "x-learner-model", "report": "causal-report"},
        ),
    )
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics=dict(result.metrics),
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
            "composition_digest": result.composition_digest,
        },
    )


__all__ = [
    "XLearnerExporter",
    "XLearnerONNXExporter",
    "XLearnerReportExporter",
    "export_result",
]
