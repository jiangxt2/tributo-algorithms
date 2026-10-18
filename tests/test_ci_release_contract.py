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
from packaging.specifiers import SpecifierSet

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
    assert project["project"]["dependencies"] == ["tributo>=1.2"]
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


@pytest.mark.parametrize(
    ("version", "accepted"),
    [("1.0.0", False), ("1.1.0", False), ("1.2.0.dev0", False), ("1.2.0", True)],
)
def test_core_dependencies_keep_only_the_minimum_version(
    version: str, accepted: bool
) -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    dependencies = [
        *project["dependencies"],
        *(
            value
            for values in project["optional-dependencies"].values()
            for value in values
        ),
    ]
    core_requirements = [
        requirement
        for value in dependencies
        if (requirement := Requirement(value)).name == "tributo"
    ]
    assert core_requirements
    for requirement in core_requirements:
        assert requirement.specifier == SpecifierSet(">=1.2")
        assert requirement.specifier.contains(version, prereleases=True) is accepted


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


def test_xgboost_single_node_smoke_profile_matches_the_lock() -> None:
    profile_path = _ROOT / "tests" / "jobs" / "xgboost-single-node-smoke.env"
    profile = {
        key: value
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }
    locked_packages = tomllib.loads((_ROOT / "uv.lock").read_text())["package"]
    locked_versions = {
        name: next(
            package["version"] for package in locked_packages if package["name"] == name
        )
        for name in ("ray", "tributo")
    }

    assert profile["PYTHON_VERSION"] == "3.12"
    assert profile["TRIBUTO_CORE_VERSION"] == locked_versions["tributo"] == "1.2.0"
    assert profile["RAY_VERSION"] == locked_versions["ray"] == "2.59.0"
    assert profile["RAY_IMAGE"].startswith("rayproject/ray:2.59.0-py312@sha256:")
    image_digest = profile["RAY_IMAGE"].split("@sha256:", maxsplit=1)[1]
    assert len(image_digest) == 64
    assert all(character in "0123456789abcdef" for character in image_digest)


def test_xgboost_single_node_smoke_is_a_separate_ci_tier() -> None:
    ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    dockerfile = (
        _ROOT / "tests" / "jobs" / "Dockerfile.xgboost-single-node-smoke"
    ).read_text(encoding="utf-8")
    runner = (_ROOT / "tests" / "jobs" / "run_xgboost_single_node_smoke.sh").read_text(
        encoding="utf-8"
    )
    payload = (_ROOT / "tests" / "jobs" / "xgboost_single_node_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "xgboost-single-node-smoke:" in ci
    assert "needs: validate" in ci
    assert 'python-version: "3.12"' in ci
    assert "uv export" in ci
    assert "Dockerfile.xgboost-single-node-smoke" in runner
    assert 'source "$ROOT_DIR/tests/jobs/xgboost-single-node-smoke.env"' in runner
    assert '--build-arg "RAY_IMAGE=$RAY_IMAGE"' in runner
    assert "ARG RAY_IMAGE" in dockerfile and "FROM " in dockerfile
    assert "--require-hashes" in dockerfile
    assert "ray job submit" in runner
    assert "trap cleanup EXIT" in runner
    assert "trap 'exit 130' INT" in runner
    assert "trap 'exit 143' TERM" in runner
    assert "--expected-tributo-version" in runner
    assert "--expected-ray-version" in runner
    assert "--expected-ray-image" in runner
    assert "RUNTIME VERSIONS:" in payload
    assert "ReadOptions(target_parallelism=4)" in payload
    assert 'ray.init(address="auto")' in payload
    assert "len({worker.node_id for worker in receipt.workers}) == 1" in payload
    assert '"ray_image": args.expected_ray_image' in payload
    assert "native_onnx_prediction_parity" in payload
    assert "xgboost_worker_gate.py" not in ci


def test_xlearner_single_node_smoke_profile_matches_the_lock() -> None:
    profile_path = _ROOT / "tests" / "jobs" / "xlearner-single-node-smoke.env"
    profile = {
        key: value
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }
    locked_packages = tomllib.loads((_ROOT / "uv.lock").read_text())["package"]
    locked_versions = {
        name: next(
            package["version"] for package in locked_packages if package["name"] == name
        )
        for name in ("ray", "tributo")
    }
    xgboost_profile_path = _ROOT / "tests" / "jobs" / "xgboost-single-node-smoke.env"
    xgboost_profile = {
        key: value
        for line in xgboost_profile_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }

    assert profile["PYTHON_VERSION"] == "3.12"
    assert profile["TRIBUTO_CORE_VERSION"] == locked_versions["tributo"] == "1.2.0"
    assert profile["RAY_VERSION"] == locked_versions["ray"] == "2.59.0"
    assert profile["RAY_IMAGE"] == xgboost_profile["RAY_IMAGE"]
    assert profile["RAY_LOCAL_TAG"] == "rayproject/ray:2.59.0-py312"


def test_xlearner_single_node_smoke_covers_batch_effect_evaluation() -> None:
    ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    dockerfile = (
        _ROOT / "tests" / "jobs" / "Dockerfile.xlearner-single-node-smoke"
    ).read_text(encoding="utf-8")
    runner = (_ROOT / "tests" / "jobs" / "run_xlearner_single_node_smoke.sh").read_text(
        encoding="utf-8"
    )
    payload = (_ROOT / "tests" / "jobs" / "xlearner_single_node_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "xlearner-single-node-smoke:" in ci
    assert ci.count("  xlearner-single-node-smoke:") == 1
    assert "needs: validate" in ci
    assert 'python-version: "3.12"' in ci
    assert "--extra causal-xlearner" in ci
    assert "Dockerfile.xlearner-single-node-smoke" in runner
    assert 'source "$ROOT_DIR/tests/jobs/xlearner-single-node-smoke.env"' in runner
    assert '--build-arg "RAY_IMAGE=$RAY_BUILD_IMAGE"' in runner
    assert "ARG RAY_IMAGE" in dockerfile and "FROM " in dockerfile
    assert "--require-hashes" in dockerfile
    assert "ray job submit" in runner
    assert "trap cleanup EXIT" in runner
    assert "trap 'exit 130' INT" in runner
    assert "trap 'exit 143' TERM" in runner
    assert 'docker_cli container rm --force "$CONTAINER_NAME"' in runner
    assert 'docker_cli image rm "$IMAGE_TAG"' in runner
    assert "/tmp/ray/session_latest/logs" in runner
    assert "ray-session-logs-copy.log" in runner
    assert runner.index("/tmp/ray/session_latest/logs") < runner.index(
        'docker_cli container rm --force "$CONTAINER_NAME"'
    )
    assert "dangling-images-before.log" in runner
    assert "docker-system-df-before.log" in runner
    assert "docker_cli system df" in runner
    assert "RAY_REPO_DIGESTS" in runner
    assert "EXPECTED_RAY_DIGEST" in runner
    assert "docker system prune" not in runner
    assert "BundleModelLoader" in payload
    assert 'use_case="batch"' in payload
    assert "evaluate_cate_predictions" in payload
    assert "ray.nodes()" in payload
    assert "ACIC_SOURCE_URL" in payload
    assert "_file_sha256" in payload
    assert "mu0" in payload and "mu1" in payload
    assert "--acic-sha256" in runner
    assert '"sample_id", "treatment", "outcome"' in payload
    assert "fit metrics do not cover all training rows" in payload
    assert '"fit_metrics": result["fit_metrics"]' in payload
    assert "uv build --no-sources" in readme
    assert "uv export" in readme
    assert "--extra causal-xlearner" in readme


def test_source_free_verifier_exercises_scalar_binding_contract() -> None:
    verifier = _verifier_module()

    verifier._verify_scalar_single_column_binding()
