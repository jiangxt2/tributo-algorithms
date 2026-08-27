"""Typed ONNX Bundle exporters for PCA and KMeans."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ArtifactDraft,
    ResolvedAlgorithmPlan,
)

from tributo_algorithms_classical.unsupervised_models import (
    IsolationForestModel,
    KMeansModel,
    PCAModel,
)


def _onnx_metadata(
    model: Any,
    plan: ResolvedAlgorithmPlan,
    graph_name: str,
) -> bytes:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    feature_count = len(model.feature_names)
    input_info = helper.make_tensor_value_info(
        "float_input", TensorProto.FLOAT, [None, feature_count]
    )
    initializers = [
        numpy_helper.from_array(
            np.asarray(model.mean, dtype=np.float32),
            name="mean",
        ),
    ]
    nodes = [
        helper.make_node("Sub", ["float_input", "mean"], ["centered"]),
        helper.make_node(
            "MatMul",
            ["centered", "components"],
            ["variable"],
        ),
    ]
    initializers.append(
        numpy_helper.from_array(
            np.asarray(model.components, dtype=np.float32).T,
            name="components",
        )
    )
    output_info = helper.make_tensor_value_info(
        "variable",
        TensorProto.FLOAT,
        [None, int(model.components.shape[0])],
    )
    graph = helper.make_graph(
        nodes,
        graph_name,
        [input_info],
        [output_info],
        initializer=initializers,
    )
    model_proto = helper.make_model(
        graph,
        producer_name="tributo-algorithms-classical",
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model_proto.metadata_props.add(
        key="feature_names", value=json.dumps(model.feature_names)
    )
    model_proto.metadata_props.add(
        key="distribution_spec_digest",
        value=plan.runtime.distribution_digest or "",
    )
    model_proto.doc_string = ""
    for node in model_proto.graph.node:
        node.doc_string = ""
    onnx.checker.check_model(model_proto)
    return bytes(model_proto.SerializeToString())


def _kmeans_onnx(model: KMeansModel, plan: ResolvedAlgorithmPlan) -> bytes:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    feature_count = len(model.feature_names)
    input_info = helper.make_tensor_value_info(
        "float_input", TensorProto.FLOAT, [None, feature_count]
    )
    labels = helper.make_tensor_value_info("label", TensorProto.INT64, [None])
    distance = helper.make_tensor_value_info("distance", TensorProto.FLOAT, [None])
    axes_batch = numpy_helper.from_array(
        np.asarray([1], dtype=np.int64), name="axes_batch"
    )
    axes_feature = numpy_helper.from_array(
        np.asarray([2], dtype=np.int64), name="axes_feature"
    )
    centers = numpy_helper.from_array(
        np.asarray(model.centers, dtype=np.float32), name="centers"
    )
    nodes = [
        helper.make_node("Unsqueeze", ["float_input", "axes_batch"], ["input_3d"]),
        helper.make_node("Unsqueeze", ["centers", "axes_zero"], ["centers_3d"]),
        helper.make_node("Sub", ["input_3d", "centers_3d"], ["delta"]),
        helper.make_node("Mul", ["delta", "delta"], ["squared"]),
        helper.make_node(
            "ReduceSum", ["squared", "axes_feature"], ["distances"], keepdims=0
        ),
        helper.make_node("ArgMin", ["distances"], ["label"], axis=1, keepdims=0),
        helper.make_node(
            "ReduceMin", ["distances", "axes_cluster"], ["distance"], keepdims=0
        ),
    ]
    initializers = [
        centers,
        axes_batch,
        axes_feature,
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="axes_zero"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="axes_cluster"),
    ]
    graph = helper.make_graph(
        nodes,
        "tributo-kmeans",
        [input_info],
        [labels, distance],
        initializer=initializers,
    )
    model_proto = helper.make_model(
        graph,
        producer_name="tributo-algorithms-classical",
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    model_proto.metadata_props.add(
        key="feature_names", value=json.dumps(model.feature_names)
    )
    model_proto.metadata_props.add(
        key="distribution_spec_digest",
        value=plan.runtime.distribution_digest or "",
    )
    model_proto.metadata_props.add(key="variant", value=model.variant)
    onnx.checker.check_model(model_proto)
    return bytes(model_proto.SerializeToString())


def _bundle_result(
    *,
    artifact: ArtifactDraft,
    model: Any,
    plan: ResolvedAlgorithmPlan,
    run_id: str | None,
    payload: bytes,
    task_type: str,
    output_schema: Sequence[Any],
    output_values: Mapping[str, object],
) -> AlgorithmExecutionResult:
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping):
        raise AlgorithmConfigurationError("unsupervised fit requires output config")
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

    source = ExportSource(
        source_kind="prebuilt_onnx",
        model_object=payload,
        feature_schema={"feature_names": list(model.feature_names)},
        metadata={
            "framework": "onnx",
            "framework_versions": {"onnx": "18"},
            "task_type": task_type,
            "producer_distribution": "tributo-algorithms-classical",
        },
        source_fingerprint=artifact.sha256,
        checkpoint_contract=ExportCheckpointV1(
            trainer_type=task_type,
            architecture_id=plan.resolution.algorithm,
            input_schema=(
                CheckpointField(
                    name="float_input",
                    dtype="float32",
                    shape=("batch", len(model.feature_names)),
                ),
            ),
            output_schema=tuple(output_schema),
            preprocessing={"type": "none"},
            task_type=task_type,
            framework="onnx",
            framework_version="18",
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
            **dict(output_values),
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
        },
        artifacts=(artifact,),
    )


def export_pca_model(
    *,
    model: PCAModel,
    plan: ResolvedAlgorithmPlan,
    run_id: str | None = None,
) -> AlgorithmExecutionResult:
    """Export PCA as a typed ONNX transform Bundle."""
    if not isinstance(model, PCAModel):
        raise AlgorithmExecutionError("PCA exporter received an invalid model")
    payload = _onnx_metadata(model, plan, "tributo-pca")
    artifact = ArtifactDraft.from_payload(
        name="model",
        kind="model",
        format="application/onnx",
        payload=payload,
    )
    from tributo.exporting.models import CheckpointField

    return _bundle_result(
        artifact=artifact,
        model=model,
        plan=plan,
        run_id=run_id,
        payload=payload,
        task_type="feature_transform",
        output_schema=(
            CheckpointField(
                name="variable",
                dtype="float32",
                shape=("batch", int(model.components.shape[0])),
            ),
        ),
        output_values={
            "n_components": int(model.components.shape[0]),
            "explained_variance": model.explained_variance.tolist(),
            "explained_variance_ratio": model.explained_variance_ratio.tolist(),
        },
    )


def export_kmeans_model(
    *,
    model: KMeansModel,
    plan: ResolvedAlgorithmPlan,
    run_id: str | None = None,
) -> AlgorithmExecutionResult:
    """Export KMeans labels and distances as a typed ONNX Bundle."""
    if not isinstance(model, KMeansModel):
        raise AlgorithmExecutionError("KMeans exporter received an invalid model")
    payload = _kmeans_onnx(model, plan)
    artifact = ArtifactDraft.from_payload(
        name="model",
        kind="model",
        format="application/onnx",
        payload=payload,
    )
    from tributo.exporting.models import CheckpointField

    return _bundle_result(
        artifact=artifact,
        model=model,
        plan=plan,
        run_id=run_id,
        payload=payload,
        task_type="clustering",
        output_schema=(
            CheckpointField(name="label", dtype="int64", shape=("batch",)),
            CheckpointField(name="distance", dtype="float32", shape=("batch",)),
        ),
        output_values={
            "n_clusters": int(model.centers.shape[0]),
            "variant": model.variant,
        },
    )


def export_isolation_forest_model(
    *,
    model: IsolationForestModel,
    plan: ResolvedAlgorithmPlan,
    run_id: str | None = None,
) -> AlgorithmExecutionResult:
    """Export Isolation Forest raw scores and the explicit score threshold."""
    if not isinstance(model, IsolationForestModel):
        raise AlgorithmExecutionError(
            "Isolation Forest exporter received an invalid model"
        )
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from tributo.exporting.models import CheckpointField

    try:
        converted = convert_sklearn(
            model.estimator,
            initial_types=[
                ("float_input", FloatTensorType([None, len(model.feature_names)]))
            ],
            target_opset={"": 18, "ai.onnx.ml": 3},
        )
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

        score_output = next(
            output for output in converted.graph.output if output.name == "scores"
        )
        del converted.graph.output[:]
        threshold_output = helper.make_tensor_value_info(
            "threshold", TensorProto.FLOAT, [None, 1]
        )
        converted.graph.output.extend([score_output, threshold_output])
        converted.graph.initializer.extend(
            [
                numpy_helper.from_array(
                    np.asarray(0, dtype=np.int64), name="threshold_batch_index"
                ),
                numpy_helper.from_array(
                    np.asarray([0], dtype=np.int64), name="threshold_axes_zero"
                ),
                numpy_helper.from_array(
                    np.asarray([1], dtype=np.int64), name="threshold_one"
                ),
                numpy_helper.from_array(
                    np.asarray(
                        [[float(cast(Any, model.estimator).offset_)]],
                        dtype=np.float32,
                    ),
                    name="threshold_value",
                ),
            ]
        )
        converted.graph.node.extend(
            [
                helper.make_node("Shape", ["float_input"], ["threshold_input_shape"]),
                helper.make_node(
                    "Gather",
                    ["threshold_input_shape", "threshold_batch_index"],
                    ["threshold_batch_size"],
                    axis=0,
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["threshold_batch_size", "threshold_axes_zero"],
                    ["threshold_batch_dim"],
                ),
                helper.make_node(
                    "Concat",
                    ["threshold_batch_dim", "threshold_one"],
                    ["threshold_shape"],
                    axis=0,
                ),
                helper.make_node(
                    "Expand", ["threshold_value", "threshold_shape"], ["threshold"]
                ),
            ]
        )
        converted.graph.name = "tributo-isolation-forest"
        converted.doc_string = ""
        for node in converted.graph.node:
            node.doc_string = ""
        payload = converted.SerializeToString()
    except Exception as exc:
        raise AlgorithmExecutionError(
            f"Isolation Forest ONNX export failed: {type(exc).__name__}"
        ) from exc
    artifact = ArtifactDraft.from_payload(
        name="model",
        kind="model",
        format="application/onnx",
        payload=payload,
    )
    return _bundle_result(
        artifact=artifact,
        model=model,
        plan=plan,
        run_id=run_id,
        payload=payload,
        task_type="anomaly_detection",
        output_schema=(
            CheckpointField(name="scores", dtype="float32", shape=("batch", 1)),
            CheckpointField(name="threshold", dtype="float32", shape=("batch", 1)),
        ),
        output_values={
            "n_estimators": len(cast(Any, model.estimator).estimators_),
            "threshold": float(cast(Any, model.estimator).offset_),
        },
    )


__all__ = [
    "export_isolation_forest_model",
    "export_kmeans_model",
    "export_pca_model",
]
