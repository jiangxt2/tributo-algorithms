"""Tests for official distributed causal core algorithms."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from sklearn import __version__ as sklearn_version
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    DistributionStrategy,
    InputBinding,
    ResolvedAlgorithmPlan,
)
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.models import ExportSource
from tributo.exporting.runtime import BundleModelLoader
from tributo.exporting.service import BundleExportService
from tributo_algorithms_causal_core import (
    ATE_DESCRIPTOR,
    LINEAR_DML_DESCRIPTOR,
    LINEAR_IV_DESCRIPTOR,
    DifferenceInMeansATE,
    LinearDMLATE,
    LinearIVATE,
)
from tributo_algorithms_causal_core.algorithm import CausalATEModel, export_model
from tributo_algorithms_causal_core.contracts import TreatmentCoverageValidator

_CAUSAL_ALGORITHM_NAMES = (
    ATE_DESCRIPTOR.registration.spec.name,
    LINEAR_DML_DESCRIPTOR.registration.spec.name,
    LINEAR_IV_DESCRIPTOR.registration.spec.name,
)


def _algorithm() -> DifferenceInMeansATE:
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            primary_input_binding=InputBinding(
                name="train",
                resolver_id="tributo.fake_tabular",
                reference="causal",
                feature_names=("treatment", "x0"),
                label_name="outcome",
            ),
            algorithm_config={
                "treatment_col": "treatment",
                "policy_cost": 0.5,
                "cross_fit_folds": 1,
            },
        ),
    )
    return DifferenceInMeansATE(plan)


def _causal_model(feature_count: int) -> CausalATEModel:
    return CausalATEModel(
        method="distributed_linear_dml",
        treatment="treatment",
        outcome="outcome",
        feature_names=("treatment",)
        + tuple(f"x{index}" for index in range(feature_count - 1)),
        treated_count=4,
        control_count=4,
        treated_mean=3.0,
        control_mean=1.0,
        effect=2.0,
        standard_error=0.5,
        confidence_interval=(1.0, 3.0),
        policy_cost=0.5,
        treat_all_policy=True,
    )


def _export_plan(*, algorithm: str, bundle_uri: Path | None) -> ResolvedAlgorithmPlan:
    output = {} if bundle_uri is None else {"bundle_uri": str(bundle_uri)}
    return cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            algorithm_config={"output": output},
            resolution=SimpleNamespace(algorithm=algorithm),
        ),
    )


def test_ate_descriptor_uses_exact_map_reduce() -> None:
    distribution = ATE_DESCRIPTOR.registration.distribution_spec
    assert distribution is not None
    assert distribution.strategy is DistributionStrategy.RAY_MAP_REDUCE
    assert ATE_DESCRIPTOR.registration.contract_bindings is not None
    assert (
        LINEAR_DML_DESCRIPTOR.registration.distribution_spec.strategy
        is DistributionStrategy.RAY_MAP_REDUCE
    )
    assert (
        LINEAR_IV_DESCRIPTOR.registration.distribution_spec.strategy
        is DistributionStrategy.RAY_MAP_REDUCE
    )


def test_difference_in_means_merges_exact_sufficient_statistics() -> None:
    algorithm = _algorithm()
    first = algorithm.map_partition(
        (
            {
                "treatment": [0, 1, 0, 1],
                "outcome": [1.0, 3.0, 2.0, 4.0],
            },
        ),
        AlgorithmExecutionContext(inputs={}),
    )
    second = algorithm.map_partition(
        (
            {
                "treatment": [0, 1, 0, 1],
                "outcome": [2.0, 4.0, 3.0, 5.0],
            },
        ),
        AlgorithmExecutionContext(inputs={}),
    )
    merged = algorithm.merge_states(first, second)
    assert all(isinstance(value, np.ndarray) for value in merged.values())
    model = algorithm.finalize_model(merged)

    assert model.treated_count == 4
    assert model.control_count == 4
    assert model.effect == 2.0
    assert model.treat_all_policy is True
    assert algorithm.coverage_counts(first) == {"treated": 2, "control": 2}


def test_treatment_coverage_contract_partitions_rows() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {
                "input_rows": {
                    "train": 4,
                    "coverage.treated": 2,
                    "coverage.control": 2,
                }
            },
            {
                "input_rows": {
                    "train": 4,
                    "coverage.treated": 2,
                    "coverage.control": 2,
                }
            },
        ],
    }
    assert TreatmentCoverageValidator().validate(value) == value


def test_linear_dml_partials_out_distributed_confounders() -> None:
    base = _algorithm().plan
    algorithm = LinearDMLATE(base)
    treatment = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
    confounder = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.float64)
    outcome = 1.0 + 2.0 * treatment + 1.5 * confounder
    states = []
    for start in (0, 4):
        states.append(
            algorithm.map_partition(
                (
                    {
                        "treatment": treatment[start : start + 4],
                        "x0": confounder[start : start + 4],
                        "outcome": outcome[start : start + 4],
                    },
                ),
                AlgorithmExecutionContext(inputs={}),
            )
        )
    model = algorithm.finalize_model(algorithm.merge_states(*states))
    assert model.method == "distributed_linear_dml"
    assert model.effect == pytest.approx(2.0)


def test_linear_dml_supports_deterministic_cross_fit_state() -> None:
    base = _algorithm().plan
    plan = SimpleNamespace(
        primary_input_binding=base.primary_input_binding,
        algorithm_config={
            "treatment_col": "treatment",
            "cross_fit_folds": 2,
        },
    )
    algorithm = LinearDMLATE(plan)
    treatment = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
    confounder = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.float64)
    outcome = 1.0 + 2.0 * treatment + 1.5 * confounder
    states = [
        algorithm.map_partition(
            (
                {
                    "treatment": treatment[start : start + 4],
                    "x0": confounder[start : start + 4],
                    "outcome": outcome[start : start + 4],
                },
            ),
            AlgorithmExecutionContext(inputs={}),
        )
        for start in (0, 4)
    ]
    merged = algorithm.merge_states(*states)
    assert merged["fold_xtx"].shape == (2, 2, 2)
    model = algorithm.finalize_model(merged)
    assert model.effect == pytest.approx(2.0)
    assert model.diagnostics["cross_fit_folds"] == 2


def test_linear_iv_recovers_synthetic_truth() -> None:
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            primary_input_binding=InputBinding(
                name="train",
                resolver_id="tributo.fake_tabular",
                reference="iv",
                feature_names=("treatment", "instrument", "x0"),
                label_name="outcome",
            ),
            algorithm_config={
                "treatment_col": "treatment",
                "instrument_col": "instrument",
                "policy_cost": 0.5,
            },
        ),
    )
    algorithm = LinearIVATE(plan)
    instrument = np.asarray([0, 1] * 8, dtype=np.float64)
    treatment = instrument.copy()
    confounder = np.asarray([0, 0, 1, 1, 2, 2, 3, 3] * 2, dtype=np.float64)
    outcome = 1.0 + 2.0 * treatment + 1.5 * confounder
    states = []
    for start in (0, 8):
        states.append(
            algorithm.map_partition(
                (
                    {
                        "treatment": treatment[start : start + 8],
                        "instrument": instrument[start : start + 8],
                        "x0": confounder[start : start + 8],
                        "outcome": outcome[start : start + 8],
                    },
                ),
                AlgorithmExecutionContext(inputs={}),
            )
        )
    model = algorithm.finalize_model(algorithm.merge_states(*states))
    assert model.method == "distributed_linear_2sls"
    assert model.effect == pytest.approx(2.0)
    assert model.diagnostics["first_stage_correlation"] == pytest.approx(1.0)


@pytest.mark.parametrize("algorithm", _CAUSAL_ALGORITHM_NAMES)
@pytest.mark.parametrize("feature_count", (1, 2, 10))
def test_causal_export_builds_typed_checkpoint_contract(
    algorithm: str,
    feature_count: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: ExportSource | None = None

    def capture_export(
        service: BundleExportService,
        source: ExportSource,
        config: object,
    ) -> SimpleNamespace:
        del service, config
        nonlocal captured
        captured = source
        return SimpleNamespace(
            bundle_id="captured-bundle",
            canonical_uri=str(tmp_path / "captured-bundle"),
            execution_id="captured-execution",
            manifest_sha256="0" * 64,
        )

    monkeypatch.setattr(BundleExportService, "export_bundle", capture_export)
    model = _causal_model(feature_count)
    export_model(
        model=model,
        plan=_export_plan(
            algorithm=algorithm,
            bundle_uri=tmp_path / "bundles",
        ),
        run_id=f"contract-{algorithm}-{feature_count}",
    )

    assert captured is not None
    assert captured.architecture_id == algorithm
    assert captured.metadata["framework"] == "sklearn"
    assert captured.metadata["framework_version"] == sklearn_version
    assert captured.metadata["framework_versions"] == {"scikit-learn": sklearn_version}
    contract = captured.checkpoint_contract
    assert contract is not None
    assert contract.trainer_type == "causal_ate"
    assert contract.architecture_id == algorithm
    assert contract.preprocessing == {"type": "none"}
    assert contract.task_type == "causal_effect_estimation"
    assert contract.framework == "sklearn"
    assert contract.framework_version == sklearn_version
    assert contract.input_schema[0].model_dump() == {
        "name": "float_input",
        "dtype": "float32",
        "shape": ("batch", feature_count),
    }
    assert contract.output_schema[0].model_dump() == {
        "name": "variable",
        "dtype": "float32",
        "shape": ("batch", 1),
    }


@pytest.mark.parametrize("algorithm", _CAUSAL_ALGORITHM_NAMES)
@pytest.mark.parametrize("feature_count", (1, 2, 10))
def test_causal_export_bundle_round_trip(
    algorithm: str, feature_count: int, tmp_path: Path
) -> None:
    model = _causal_model(feature_count)
    result = export_model(
        model=model,
        plan=_export_plan(
            algorithm=algorithm,
            bundle_uri=tmp_path / "bundles",
        ),
        run_id=f"round-trip-{algorithm}-{feature_count}",
    )
    bundle_uri = cast(str, result.outputs["bundle_uri"])

    manifest = BundleReader().read_manifest(bundle_uri)
    assert manifest.roles == {
        "inference": "effect-model",
        "report": "causal-report",
    }
    assert manifest.source_info.framework == "sklearn"
    assert manifest.source_info.framework_version == sklearn_version
    assert manifest.source_info.architecture_id == algorithm
    assert manifest.source_info.task_type == "causal_effect_estimation"
    assert [field.model_dump() for field in manifest.input_signature.input_fields] == [
        {
            "name": "float_input",
            "dtype": "float32",
            "shape": ("batch", feature_count),
        }
    ]
    assert [
        field.model_dump() for field in manifest.output_signature.output_fields
    ] == [
        {
            "name": "variable",
            "dtype": "float32",
            "shape": ("batch", 1),
        }
    ]

    with BundleReader().open_artifact(bundle_uri, role="report") as artifact:
        report = json.loads(artifact.entrypoint_path.read_text(encoding="utf-8"))
    assert report["kind"] == "causal_report"
    assert report["exporter_id"] == "official-causal-report-v1"
    assert report["study"] == model.report()

    inputs = np.arange(3 * feature_count, dtype=np.float32).reshape(3, feature_count)
    with BundleModelLoader().open(
        bundle_uri,
        role="inference",
        unsafe=False,
    ) as runtime:
        outputs = runtime.predict({"float_input": inputs})

    assert tuple(outputs) == ("variable",)
    assert outputs["variable"].dtype == np.dtype("float32")
    assert outputs["variable"].shape == (3, 1)
    np.testing.assert_allclose(
        outputs["variable"],
        np.full((3, 1), model.effect, dtype=np.float32),
    )


def test_causal_export_rejects_invalid_model(tmp_path: Path) -> None:
    with pytest.raises(AlgorithmExecutionError, match="invalid model"):
        export_model(
            model=cast(CausalATEModel, object()),
            plan=_export_plan(
                algorithm=_CAUSAL_ALGORITHM_NAMES[0],
                bundle_uri=tmp_path / "bundles",
            ),
        )


def test_causal_export_requires_bundle_uri(tmp_path: Path) -> None:
    with pytest.raises(AlgorithmConfigurationError, match="bundle_uri"):
        export_model(
            model=_causal_model(2),
            plan=_export_plan(
                algorithm=_CAUSAL_ALGORITHM_NAMES[0],
                bundle_uri=None,
            ),
        )
