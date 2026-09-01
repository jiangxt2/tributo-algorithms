"""Static contracts for reproducible CI and release source layout."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

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


def test_release_tag_resolves_exact_workspace_package() -> None:
    resolver = _resolver_module()

    release = resolver.resolve_release_package(_ROOT, "classical-v0.1.0")

    assert release.directory == "classical"
    assert release.distribution == "tributo-algorithms-classical"
    assert release.version == "0.1.0"


@pytest.mark.parametrize(
    "tag, message",
    [
        ("classical-0.1.0", "must match"),
        ("../classical-v0.1.0", "must match"),
        ("unknown-v0.1.0", "unknown package"),
        ("classical-v0.2.0", "does not match package version"),
    ],
)
def test_release_tag_rejects_unsafe_or_inconsistent_values(
    tag: str,
    message: str,
) -> None:
    resolver = _resolver_module()

    with pytest.raises(ValueError, match=message):
        resolver.resolve_release_package(_ROOT, tag)


def test_release_resolver_writes_github_environment(tmp_path: Path) -> None:
    github_env = tmp_path / "github-env"

    result = subprocess.run(
        [
            sys.executable,
            str(_RESOLVER),
            "--root",
            str(_ROOT),
            "--tag",
            "graph-pyg-v0.1.0",
            "--github-env",
            str(github_env),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert github_env.read_text(encoding="utf-8").splitlines() == [
        "PACKAGE_DIRECTORY=graph-pyg",
        "DISTRIBUTION=tributo-algorithms-graph-pyg",
        "PACKAGE_VERSION=0.1.0",
    ]


def test_workflows_use_the_portable_nested_checkout_layout() -> None:
    ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release = (_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    for workflow in (ci, release):
        assert "path: workspace/tributo-algorithms" in workflow
        assert "path: Tributo" in workflow
        assert "working-directory: workspace/tributo-algorithms" in workflow
        assert "ref: ${{ env.TRIBUTO_CORE_REF }}" in workflow
        assert "uv lock --check" in workflow
        assert "--no-sources" in workflow
        assert "verify_installed_distribution.py" in workflow
        assert "--require-scalar-single-column-binding" in workflow
        assert "6ef79841261d88c98bab420193afbe58ca6baa28" in workflow
    assert "branches: [master]" in ci
    assert "branches: [main]" not in ci
    assert "resolve_release_package.py" in release
    assert "packages-dir: workspace/tributo-algorithms/dist/" in release


def test_source_free_verifier_exercises_scalar_binding_contract() -> None:
    verifier = _verifier_module()

    verifier._verify_scalar_single_column_binding()
