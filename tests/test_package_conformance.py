"""Official Wheels consume Tributo's public algorithm Conformance Testkit."""

from __future__ import annotations

import subprocess
import sys


def test_every_official_entry_point_passes_descriptor_only_conformance() -> None:
    script = r"""
import importlib.metadata
import json
import sys

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from tributo.algorithms.conformance import validate_installed_algorithm_package

heavy = {
    "catboost",
    "causallearn",
    "dowhy",
    "lightgbm",
    "torch",
    "torch_geometric",
    "transformers",
    "xgboost",
}
before = heavy & set(sys.modules)
reports = []
expected = {
    "xgboost.framework_native", "lightgbm.framework_native", "catboost.parallel_ensemble",
    "difference_in_means_ate", "linear_dml_ate", "linear_iv_ate", "pc_stability_discovery",
    "dowhy_linear_refutation", "gcm_root_cause", "doubly_robust_ate", "x_learner.framework_native",
    "extra_trees.joblib", "extra_trees.native", "linear_regression.iterative", "random_forest.joblib",
    "random_forest.native", "logistic_regression.iterative", "multinomial_nb", "pca.map_reduce",
    "kmeans.iterative", "kmeans_minibatch.iterative", "sgd_classifier.iterative", "sgd_regressor.iterative",
    "isolation_forest.parallel_ensemble", "graphsage_node_classifier", "rgcn_node_classifier",
    "pretrain_finetune_classifier", "teacher_student_distillation", "jagged_embedding_recommender",
    "two_tower_recommender", "tabular_autoencoder", "dnn", "pu", "temporal_conv_classifier",
    "lstm_classifier", "gru_classifier", "token_transformer_classifier",
}
for entry_point in importlib.metadata.entry_points(group="tributo.algorithms"):
    distribution = getattr(entry_point, "dist", None)
    name = distribution.metadata["Name"] if distribution is not None else ""
    if name.lower().replace("_", "-") != "tributo-algorithms":
        continue
    descriptor = entry_point.load()
    core_specifier = SpecifierSet(descriptor.tributo_version_spec)
    if core_specifier != SpecifierSet(">=1.1"):
        raise AssertionError("official descriptor changed the Core minimum or added an upper bound")
    if core_specifier.contains("1.0.0", prereleases=True):
        raise AssertionError("official descriptor admitted Core below the minimum")
    if not core_specifier.contains("1.2.0.dev0", prereleases=True):
        raise AssertionError("official descriptor rejected the selected Core development version")
    core_dependencies = tuple(
        dependency
        for value in descriptor.registration.environment.dependencies
        if (dependency := Requirement(value)).name == "tributo"
    )
    if not core_dependencies or any(
        dependency.specifier != core_specifier for dependency in core_dependencies
    ):
        raise AssertionError("official EnvironmentSpec disagrees with Core compatibility")
    report = validate_installed_algorithm_package(
        descriptor,
        entry_point_name=entry_point.name,
    )
    if report.implementation_loaded:
        raise AssertionError("Conformance loaded an implementation module")
    if len(report.contract_ids) != 4:
        raise AssertionError("official descriptor omitted executable contracts")
    reports.append(report)
after = heavy & set(sys.modules)
if after != before:
    raise AssertionError(f"descriptor discovery imported heavy frameworks: {sorted(after - before)}")
print(json.dumps({"count": len(reports)}))
if {report.entry_point_name for report in reports} != expected:
    raise AssertionError("official algorithm Entry Point names drifted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"count": 37' in result.stdout
