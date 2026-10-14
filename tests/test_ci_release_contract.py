"""Static contracts for the aggregate package and its CI/release flow."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[1]
_RESOLVER = _ROOT / "scripts" / "resolve_release_package.py"
_VERIFIER = _ROOT / "scripts" / "verify_installed_distribution.py"


def _resolver_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_package_resolver", _RESOLVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verifier_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "installed_distribution_verifier", _VERIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_tag_resolves_the_single_distribution() -> None:
    resolver = _resolver_module()

    release = resolver.resolve_release_package(_ROOT, "tributo-algorithms-v0.1.0")

    assert release.distribution == "tributo-algorithms"
    assert release.version == "0.1.0"


@pytest.mark.parametrize(
    "tag, message",
    [
        ("tributo-algorithms-0.1.0", "must match"),
        ("../tributo-algorithms-v0.1.0", "must match"),
        ("unknown-v0.1.0", "must match"),
        ("tributo-algorithms-v0.2.0", "does not match project version"),
    ],
)
def test_release_tag_rejects_unsafe_or_inconsistent_values(
    tag: str,
    message: str,
) -> None:
    resolver = _resolver_module()

    with pytest.raises(ValueError, match=message):
        resolver.resolve_release_package(_ROOT, tag)


def test_release_resolver_writes_github_step_outputs(tmp_path: Path) -> None:
    github_output = tmp_path / "github-output"

    result = subprocess.run(
        [
            sys.executable,
            str(_RESOLVER),
            "--root",
            str(_ROOT),
            "--tag",
            "tributo-algorithms-v0.1.0",
            "--github-output",
            str(github_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        "distribution=tributo-algorithms",
        "version=0.1.0",
    ]


def test_root_metadata_declares_one_distribution_and_all_domain_extras() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())

    assert project["project"]["name"] == "tributo-algorithms"
    assert project["project"]["dependencies"] == ["tributo>=1.1,<1.2"]
    entry_points = project["project"]["entry-points"]
    assert len(entry_points["tributo.algorithms"]) == 37
    assert len(entry_points["tributo.exporters"]) == 9
    assert len(entry_points["tributo.validators"]) == 1
    assert len(entry_points["tributo.model_flavors"]) == 3

    extras = project["project"]["optional-dependencies"]
    assert set(extras) == {
        "all",
        "boosting",
        "catboost",
        "causal-core",
        "causal-discovery",
        "causal-dowhy",
        "causal-dr",
        "causal-xlearner",
        "classical",
        "graph-pyg",
        "multistage-torch",
        "recsys-torch",
        "representation",
        "tabular-torch",
        "timeseries",
        "transformers-nlp",
    }
    assert all(
        not dependency.startswith("tributo-algorithms-")
        for dependencies in extras.values()
        for dependency in dependencies
    )
    all_dependencies = {Requirement(value).name for value in extras["all"]}
    assert all(
        {Requirement(value).name for value in dependencies} <= all_dependencies
        for extra, dependencies in extras.items()
        if extra != "all"
    )
    assert "torch>=2.5,<2.14" in extras["graph-pyg"]
    assert "torch>=2.5,<2.14" in extras["all"]
    assert not list((_ROOT / "packages").glob("*/pyproject.toml"))


def test_ci_and_release_workflows_cover_single_artifact_flow() -> None:
    ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release = (_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert 'python-version: ["3.12", "3.13"]' in ci
    assert "uv sync --locked --extra all --group dev" in ci
    assert "uv build --no-sources" in ci
    assert "--distribution tributo-algorithms" in ci
    assert "--expected-entry-points 37" in ci
    assert "--expected-exporters 9" in ci
    assert "--expected-validators 1" in ci
    assert "--expected-model-flavors 3" in ci
    assert "--expected-extras all boosting" in ci
    assert "branches: [master]" in ci
    assert "path: Tributo" not in ci
    assert "TRIBUTO_CORE_REF" not in ci

    assert '"tributo-algorithms-v*"' in release
    assert "repository-url: https://test.pypi.org/legacy/" in release
    assert "--extra-index-url" not in release
    assert "--default-index https://pypi.org/simple" in release
    assert "--default-index https://test.pypi.org/simple" in release
    assert '--no-deps --reinstall-package "${DISTRIBUTION}"' in release
    assert "--expected-exporters 9" in release
    assert "--expected-extras all boosting" in release
    assert "needs: [build, verify-testpypi]" in release
    assert "sha256sum --check SHA256SUMS" in release
    assert release.count("uses: pypa/gh-action-pypi-publish@release/v1") == 2
    assert "id-token: write" in release


def test_source_free_verifier_exercises_scalar_binding_contract() -> None:
    verifier = _verifier_module()

    verifier._verify_scalar_single_column_binding()
