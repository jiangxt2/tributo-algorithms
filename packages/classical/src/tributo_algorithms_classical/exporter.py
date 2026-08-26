"""ONNX Bundle exporter shared by official sklearn algorithms."""

from __future__ import annotations

import json
from collections.abc import Mapping

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
)
from tributo.util.annotations import PublicAPI

from tributo_algorithms_classical.models import SklearnModel


@PublicAPI(stability="alpha")
def export_sklearn_model(
    *,
    model: SklearnModel,
    plan: ResolvedAlgorithmPlan,
    run_id: str | None = None,
) -> AlgorithmExecutionResult:
    """Export a fitted sklearn model through Tributo's existing Bundle service."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn import __version__ as sklearn_version

    if not isinstance(model, SklearnModel):
        raise AlgorithmExecutionError("classical exporter received an invalid model")
    try:
        converted = convert_sklearn(
            model.estimator,
            initial_types=[
                ("float_input", FloatTensorType([None, len(model.feature_names)]))
            ],
            options=(
                {id(model.estimator): {"zipmap": False}}
                if model.task == "classification"
                else None
            ),
            target_opset=18,
        )
        converted.graph.name = f"tributo-{plan.resolution.algorithm}"
        converted.doc_string = ""
        for node in converted.graph.node:
            node.doc_string = ""
        for name, value in sorted(
            {
                "feature_names": json.dumps(model.feature_names),
                "distribution_spec_digest": plan.runtime.distribution_digest or "",
            }.items()
        ):
            item = converted.metadata_props.add()
            item.key = name
            item.value = value
        payload = converted.SerializeToString()
    except Exception as exc:
        raise AlgorithmExecutionError(
            f"classical ONNX export failed: {type(exc).__name__}"
        ) from exc
    artifact = ArtifactDraft.from_payload(
        name="model",
        kind="model",
        format="application/onnx",
        payload=payload,
    )
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping):
        raise AlgorithmConfigurationError("classical fit requires output config")
    bundle_uri = output.get("bundle_uri")
    if not isinstance(bundle_uri, str) or not bundle_uri:
        raise AlgorithmConfigurationError("output.bundle_uri must be non-empty")
    from tributo.exporting.models import (
        BundleOutputConfig,
        CheckpointField,
        ExportCheckpointV1,
        ExportSource,
        ExportTarget,
    )
    from tributo.exporting.service import BundleExportService

    class_count = len(getattr(model.estimator, "classes_", ()))
    if model.task == "classification" and class_count < 2:
        raise AlgorithmExecutionError(
            "classification estimator must expose at least two classes"
        )

    source = ExportSource(
        source_kind="prebuilt_onnx",
        model_object=payload,
        feature_schema={"feature_names": list(model.feature_names)},
        metadata={
            "framework": "sklearn",
            "framework_versions": {"scikit-learn": sklearn_version},
            "task_type": model.task,
            "producer_distribution": "tributo-algorithms-classical",
        },
        source_fingerprint=artifact.sha256,
        checkpoint_contract=ExportCheckpointV1(
            trainer_type="sklearn",
            architecture_id=plan.resolution.algorithm,
            input_schema=(
                CheckpointField(
                    name="float_input",
                    dtype="float32",
                    shape=("batch", len(model.feature_names)),
                ),
            ),
            output_schema=(
                CheckpointField(
                    name=("label" if model.task == "classification" else "variable"),
                    dtype=("int64" if model.task == "classification" else "float32"),
                    shape=("batch", 1)
                    if model.task != "classification"
                    else ("batch",),
                ),
                *(
                    (
                        CheckpointField(
                            name="probabilities",
                            dtype="float32",
                            shape=("batch", class_count),
                        ),
                    )
                    if model.task == "classification"
                    else ()
                ),
            ),
            task_type=model.task,
            framework="sklearn",
            framework_version=sklearn_version,
        ),
    )
    bundle = BundleExportService().export_bundle(
        source,
        BundleOutputConfig(
            bundle_uri=bundle_uri,
            request_id=run_id,
            run_id=run_id,
            targets=[
                ExportTarget(
                    name="onnx-model",
                    format="onnx",
                    exporter_id="prebuilt-onnx-v1",
                )
            ],
            roles={"inference": "onnx-model"},
        ),
    )
    return AlgorithmExecutionResult(
        status="succeeded",
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
        },
        artifacts=(artifact,),
    )


__all__ = ["export_sklearn_model"]
