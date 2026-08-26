"""Official Wheels consume Tributo's public algorithm Conformance Testkit."""

from __future__ import annotations

import subprocess
import sys


def test_every_official_entry_point_passes_descriptor_only_conformance() -> None:
    script = r"""
import importlib.metadata
import json
import sys

from tributo.algorithms.conformance import validate_installed_algorithm_package

heavy = {"causallearn", "dowhy", "torch", "torch_geometric", "transformers", "xgboost"}
before = heavy & set(sys.modules)
reports = []
for entry_point in importlib.metadata.entry_points(group="tributo.algorithms"):
    distribution = getattr(entry_point, "dist", None)
    name = distribution.metadata["Name"] if distribution is not None else ""
    if not name.startswith("tributo-algorithms-"):
        continue
    report = validate_installed_algorithm_package(
        entry_point.load(),
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
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"count": 27' in result.stdout
