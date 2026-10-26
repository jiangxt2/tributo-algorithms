"""Check that the manual gate rejects incomplete or false distributed evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pytorch_e2e_job", Path(__file__).parent / "jobs/pytorch_e2e_job.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_JOB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_JOB)


@pytest.fixture
def cpu_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_pytorch_cpu_issues",
        Path(__file__).parent / "jobs/run_pytorch_cpu_issues.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_core_runtime_gate_mounts_its_version_fixture_inputs() -> None:
    compose = (
        Path(__file__).parent / "jobs/docker-compose.pytorch-e2e.yml"
    ).read_text()
    for name in (
        "tests",
        "tools",
        "scripts",
        "ci",
        "pyproject.toml",
        "uv.lock",
        "src/tributo/_common/runtime_versions.json",
    ):
        assert (
            f"${{TRIBUTO_CORE_DIR}}/{name}:/opt/tributo-core-tests/{name}:ro" in compose
        )


def test_resume_can_select_only_the_unfinished_core_case(
    cpu_runner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--core-dir",
            "/core",
            "--evidence",
            "/new",
            "--reuse-core-build",
            "/passed",
            "--cases",
            "core_runtime_profile",
        ],
    )
    args = cpu_runner.parse_arguments()
    assert args.reuse_core_build == Path("/passed")
    assert args.cases == ["core_runtime_profile"]
    assert cpu_runner.gate_command(Path("/algorithms"), tuple(args.cases))[2:] == [
        "--cases",
        "core_runtime_profile",
    ]


@pytest.mark.parametrize("failure", [None, "inputs", "profile", "cache", "timeout"])
def test_cache_restore_reuses_validation_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_runner: Any,
    failure: str | None,
) -> None:
    import hashlib

    @dataclass
    class Config:
        image: str
        external_wheelhouse: None = None

    expected = {"requirements_sha256": "original", "image": "passed:local"}

    def canonical(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True).encode()

    manifest = {
        **expected,
        "installed_distributions": {},
        "image_digest": "a" * 64,
        "manifest_sha256": hashlib.sha256(canonical(expected)).hexdigest(),
    }
    previous = tmp_path / "previous"
    (previous / "core-image").mkdir(parents=True)
    (previous / "core-image/manifest.json").write_text(json.dumps(manifest))
    (previous / "core-image/image-profile.json").write_text(
        json.dumps({"image_digest": "b" * 64 if failure == "profile" else "a" * 64})
    )
    evidence = tmp_path / "new"
    evidence.mkdir()
    commands = []

    def run(command: list[str], **kwargs: Any) -> None:
        commands.append(command)
        assert kwargs["timeout"] == 60
        assert not kwargs["env"]
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 60)
        kwargs["stdout"].write(
            "#1 [stage-0 1/1] RUN check\n"
            + ("#1 DONE 0.1s\n" if failure == "cache" else "#1 CACHED\n")
        )

    monkeypatch.setattr(cpu_runner.subprocess, "run", run)
    builder = SimpleNamespace(
        load_config=lambda *a, **kw: Config("passed:local"),
        _CommandLog=lambda: SimpleNamespace(command=lambda command: None),
        _host_dependency_export=lambda *a, **kw: "export",
        _manifest_core=lambda *a, **kw: {
            **expected,
            "requirements_sha256": "changed" if failure == "inputs" else "original",
        },
        wheel_records=lambda *a: [],
        canonical_json=canonical,
        sha256_bytes=lambda value: hashlib.sha256(value).hexdigest(),
        prepared_wheelhouse=lambda *a: nullcontext(tmp_path),
        prepared_requirements=lambda *a: nullcontext(tmp_path),
        prepared_project_wheelhouse=lambda **kw: nullcontext(tmp_path),
        build_command=lambda config, **kw: ["docker", "build", config.image, "."],
    )
    # No build_image/_validate_image is provided: passing reuse must never call them.
    arguments = {
        "core": tmp_path,
        "previous": previous,
        "evidence": evidence,
        "image": "fresh:local",
        "docker_env": {},
    }
    if failure:
        with pytest.raises(
            subprocess.TimeoutExpired if failure == "timeout" else ValueError
        ):
            cpu_runner.restore_core_image(builder, **arguments)
        assert not (evidence / "reused-core-validation.json").exists()
        assert len(commands) == (0 if failure in {"inputs", "profile"} else 1)
    else:
        assert cpu_runner.restore_core_image(builder, **arguments) == "a" * 64
        assert len(commands) == 1 and "--progress=plain" in commands[0]
        assert (
            json.loads((evidence / "reused-core-validation.json").read_text())[
                "validation_rerun"
            ]
            is False
        )


@pytest.mark.parametrize("actual", ["sha256:" + "a" * 64, "sha256:" + "b" * 64])
def test_cache_restore_requires_the_exact_passed_digest(
    cpu_runner: Any, actual: str
) -> None:
    if actual.endswith("a" * 64):
        cpu_runner.require_image_digest(actual, "a" * 64)
    else:
        with pytest.raises(ValueError, match="differs"):
            cpu_runner.require_image_digest(actual, "a" * 64)


def test_cpu_runner_case_selection_reaches_actual_job_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "run_pytorch_cpu_issues",
        Path(__file__).parent / "jobs/run_pytorch_cpu_issues.py",
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cases = runner.CPU_CASES
    assert len(cases) == 7 and set(cases) <= set(_JOB.DEFAULT_CASES)
    assert not set(cases) & set(_JOB.PERFORMANCE_CASES)
    command = runner.gate_command(Path("/candidate"), cases)
    assert command[:2] == ["bash", "/candidate/tests/jobs/run_pytorch_e2e.sh"]
    selected: list[str] = []

    def orchestrate(values: list[str]) -> int:
        selected.extend(values)
        return 0

    monkeypatch.setattr(_JOB, "orchestrate", orchestrate)
    monkeypatch.setattr(sys, "argv", ["job", "--orchestrate", *command[2:]])
    assert _JOB.main() == 0
    assert selected == list(cases)


def test_cpu_build_download_settings_preserve_hashed_install_and_defaults() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_pytorch_cpu_issues",
        Path(__file__).parent / "jobs/run_pytorch_cpu_issues.py",
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original = ["docker", "buildx", "build", "--load", "/candidate"]
    result = runner.with_download_settings(original)
    assert original == ["docker", "buildx", "build", "--load", "/candidate"]
    assert result == [
        *original[:-1],
        "--build-arg",
        "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
        "--build-arg",
        "PIP_DEFAULT_TIMEOUT=600",
        "/candidate",
    ]
    dockerfile = (Path(__file__).parent / "jobs/Dockerfile.pytorch-e2e").read_text()
    assert "ARG PIP_INDEX_URL=https://pypi.org/simple" in dockerfile
    assert "ARG PIP_DEFAULT_TIMEOUT=15" in dockerfile
    assert 'PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT}"' in dockerfile
    assert "pip install --require-hashes" in dockerfile


def _receipt() -> dict[str, Any]:
    return {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {
                "worker_id": "a",
                "rank": 0,
                "node_id": "n1",
                "input_rows": {"train": 4},
                "model_state_digest": "a" * 64,
            },
            {
                "worker_id": "b",
                "rank": 1,
                "node_id": "n2",
                "input_rows": {"train": 3},
                "model_state_digest": "a" * 64,
            },
        ],
    }


def test_receipt_accepts_exact_uneven_coverage() -> None:
    _JOB.validate_receipt(_receipt(), 7, 2)


@pytest.mark.parametrize(
    "field,value", [("input_complete", False), ("distributed", False)]
)
def test_receipt_rejects_incomplete_success(field: str, value: bool) -> None:
    receipt = _receipt()
    receipt[field] = value
    with pytest.raises(AssertionError):
        _JOB.validate_receipt(receipt, 7, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_id", "a"),
        ("rank", 0),
        ("node_id", "n1"),
        ("model_state_digest", "b" * 64),
        ("input_rows", {"train": 4}),
    ],
)
def test_receipt_rejects_duplicate_missing_or_inconsistent_evidence(
    field: str, value: object
) -> None:
    receipt = deepcopy(_receipt())
    receipt["workers"][1][field] = value
    with pytest.raises(AssertionError):
        _JOB.validate_receipt(receipt, 7, 2)


def test_matrix_covers_thirteen_algorithms_once() -> None:
    assert len(_JOB.ALGORITHMS) == len(set(_JOB.ALGORITHMS)) == 13
    assert len(_JOB.DEFAULT_CASES) == len(set(_JOB.DEFAULT_CASES))
    assert len(_JOB.PERFORMANCE_CASES) == 8


@pytest.mark.parametrize("failure", [None, "pip", "wheel", "duplicate"])
def test_package_gate_checks_every_node(failure: str | None) -> None:
    nodes: list[dict[str, Any]] = [
        {"node_id": str(i), "pip_check": {"returncode": 0}, "unsupported_wheels": []}
        for i in range(3)
    ]
    if failure == "pip":
        nodes[1]["pip_check"]["returncode"] = 1
    elif failure == "wheel":
        nodes[2]["unsupported_wheels"] = ["invalid-platform-wheel"]
    elif failure == "duplicate":
        nodes[1]["node_id"] = nodes[0]["node_id"]
    if failure:
        with pytest.raises(AssertionError):
            _JOB.validate_package_nodes(nodes)
    else:
        _JOB.validate_package_nodes(nodes)


def test_early_docker_failure_cannot_report_success(tmp_path: Path) -> None:
    """Use no Docker daemon: fail the first snapshot and exercise the EXIT trap."""
    tools = tmp_path / "bin"
    tools.mkdir()
    scripts = {
        "git": '#!/bin/sh\nprintf "%s\\n" "$2"\n',
        "docker": "#!/bin/sh\nexit 44\n",
        "trash": '#!/usr/bin/env python3\nimport os, sys\nfrom pathlib import Path\np=Path(sys.argv[1]); os.rename(p, p.with_name(p.name + ".trashed"))\n',
    }
    for name, content in scripts.items():
        target = tools / name
        target.write_text(content)
        target.chmod(0o755)
    evidence = tmp_path / "evidence"
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["bash", str(root / "tests/jobs/run_pytorch_e2e.sh")],
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "TMPDIR": str(tmp_path),
            "TRIBUTO_CORE_DIR": str(root),
            "TRIBUTO_E2E_EVIDENCE": str(evidence),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    cleanup = json.loads((evidence / "cleanup.json").read_text())
    assert cleanup["test_exit"] == 44
    assert cleanup["cleanup_exit"] != 0


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_gate_cancellation_waits_for_the_shell_cleanup_trap(
    tmp_path: Path, signum: int
) -> None:
    runner = Path(__file__).parent / "jobs/run_pytorch_cpu_issues.py"
    marker, ready = tmp_path / "cleaned", tmp_path / "ready"
    shell = tmp_path / "child.sh"
    shell.write_text(
        'trap \'printf cleaned > "$1"\' EXIT\nprintf "%s" "$$" > "$2"\nsleep 60\n'
    )
    code = (
        "import importlib.util,os,sys; from pathlib import Path; "
        "s=importlib.util.spec_from_file_location('runner',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "raise SystemExit(m.run_gate(['bash',sys.argv[2],sys.argv[3],sys.argv[4]],"
        "cwd=Path.cwd(),env=dict(os.environ)))"
    )
    # The child shell and its foreground sleep must receive TERM together.
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(runner), str(shell), str(marker), str(ready)]
    )
    try:
        deadline = time.monotonic() + 5
        while (
            not ready.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert ready.exists(), "managed child did not start"
        process.send_signal(signum)
        assert process.wait(timeout=5) == 128 + signum
        assert marker.read_text() == "cleaned"
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        if ready.exists():
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_gate_returns_child_failure_and_restores_signal_handlers(
    cpu_runner: Any,
) -> None:
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    assert (
        cpu_runner.run_gate(
            [sys.executable, "-c", "raise SystemExit(7)"],
            cwd=Path.cwd(),
            env=dict(os.environ),
        )
        == 7
    )
    assert handlers == {s: signal.getsignal(s) for s in handlers}


def test_gate_cleanup_timeout_is_bounded_and_cannot_report_success(
    tmp_path: Path,
) -> None:
    shell, ready = tmp_path / "child.sh", tmp_path / "ready"
    shell.write_text('trap "" TERM\nprintf "%s" "$$" > "$1"\nsleep 60\n')
    runner = Path(__file__).parent / "jobs/run_pytorch_cpu_issues.py"
    code = (
        "import importlib.util,os,sys; from pathlib import Path; "
        "s=importlib.util.spec_from_file_location('runner',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "raise SystemExit(m.run_gate(['bash',sys.argv[2],sys.argv[3]],"
        "cwd=Path.cwd(),env=dict(os.environ),cleanup_timeout=0.1))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(runner), str(shell), str(ready)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while (
            not ready.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert ready.exists()
        process.terminate()
        _, error = process.communicate(timeout=5)
        assert process.returncode != 0 and "cleanup exceeded its timeout" in error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if ready.exists():
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("failure", [None, "rows", "no_empty_step", "unbound", "stale"])
def test_reused_transformer_training_runs_the_current_coverage_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    report = {
        "algorithm": "token_transformer_classifier",
        "outputs": {},
        "reuse_fingerprint": "current",
        "receipt": _receipt(),
    }
    for worker, rows in zip(report["receipt"]["workers"], [4097, 4096], strict=True):
        worker["input_rows"]["train"] = rows
        worker["batch_count"] = 33
    if failure == "rows":
        report["receipt"]["workers"][1]["input_rows"]["train"] -= 1
    elif failure == "no_empty_step":
        report["receipt"]["workers"][1]["batch_count"] = 32
    elif failure == "unbound":
        report.pop("reuse_fingerprint")
    elif failure == "stale":
        report["reuse_fingerprint"] = "old"
    monkeypatch.setattr(_JOB, "EVIDENCE", tmp_path)
    monkeypatch.setattr(_JOB, "ROOT", tmp_path / "work")
    monkeypatch.setattr(_JOB, "training_reuse_fingerprint", lambda: "current")
    verified = []
    monkeypatch.setattr(
        _JOB, "verify_bundle", lambda *args: verified.append(True) or {}
    )

    def no_training(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("training must not run while checking reuse")

    monkeypatch.setattr(_JOB, "execute", no_training)
    (tmp_path / "token_transformer_classifier-training.json").write_text(
        json.dumps(report)
    )
    if failure:
        with pytest.raises((AssertionError, ValueError)):
            _JOB.run_algorithm("token_transformer_classifier")
        assert not verified
    else:
        result = _JOB.run_algorithm("token_transformer_classifier")
        assert result["reused_training"]
        assert result["empty_batch_alignment"]["zero_steps_by_rank"] == [0, 1]
        assert verified == [True]


def test_training_reuse_fingerprint_binds_dependencies_and_executable_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("locked requirements\n")
    monkeypatch.setattr(_JOB, "RUNTIME_REQUIREMENTS", requirements)
    versions = [SimpleNamespace(metadata={"Name": "torch"}, version="2.13.0+cpu")]
    monkeypatch.setattr(_JOB.importlib.metadata, "distributions", lambda: versions)
    original = _JOB.training_reuse_fingerprint()
    requirements.write_text("different requirements\n")
    assert _JOB.training_reuse_fingerprint() != original
    requirements.write_text("locked requirements\n")
    versions[0].version = "different"
    assert _JOB.training_reuse_fingerprint() != original
    versions[0].version = "2.13.0+cpu"
    source = "def fixture():\n    'old docs'\n    return 8193\n"
    edited = "def fixture():\n    'new docs'\n    # comment\n    return (8193)\n"
    assert _JOB.semantic_source(source) == _JOB.semantic_source(edited)
    assert _JOB.semantic_source(source) != _JOB.semantic_source(
        source.replace("8193", "8195")
    )
    original_source = _JOB.inspect.getsource
    monkeypatch.setattr(
        _JOB.inspect, "getsource", lambda fn: original_source(fn) + "\n# comment\n"
    )
    assert _JOB.training_reuse_fingerprint() == original
    monkeypatch.setattr(
        _JOB.inspect,
        "getsource",
        lambda fn: (
            original_source(fn).replace('"seed": 42', '"seed": 43')
            if fn is _JOB.fixture
            else original_source(fn)
        ),
    )
    assert _JOB.training_reuse_fingerprint() != original


@pytest.mark.parametrize(
    "case", [*_JOB.ALGORITHMS, "multi_epoch", "token_transformer_large_batch"]
)
def test_fresh_and_reused_training_share_the_same_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    rows = (
        31
        if case in {"graphsage_node_classifier", "rgcn_node_classifier"}
        else 8192
        if case == "pu"
        else 8193
        if case == "token_transformer_classifier"
        else 8195
    )
    receipt = _receipt()
    for worker, size in zip(
        receipt["workers"], [rows // 2, rows - rows // 2], strict=True
    ):
        worker["input_rows"]["train"] = size
        worker["batch_count"] = 33
    calls = []

    def fixture(
        algorithm: str, root: Path, **kwargs: Any
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        root.mkdir(parents=True)
        assert kwargs["epochs"] == (3 if case == "multi_epoch" else 1)
        assert kwargs["batch_size"] == (
            8192 if case == "token_transformer_large_batch" else 128
        )
        return "request", {}, {}

    def execute(*args: Any) -> Any:
        calls.append("fit")
        return SimpleNamespace(
            execution=SimpleNamespace(status="succeeded", metrics={}, outputs={}),
            execution_receipt=SimpleNamespace(to_dict=lambda: receipt),
        )

    monkeypatch.setattr(_JOB, "ROOT", tmp_path / "work")
    monkeypatch.setattr(_JOB, "EVIDENCE", tmp_path)
    monkeypatch.setattr(_JOB, "fixture", fixture)
    monkeypatch.setattr(_JOB, "execute", execute)
    monkeypatch.setattr(_JOB, "training_reuse_fingerprint", lambda: "current")
    monkeypatch.setattr(
        _JOB, "verify_bundle", lambda *args: calls.append("bundle") or {}
    )
    fresh = _JOB.run_algorithm(case)
    reused = _JOB.run_algorithm(case)
    assert calls == ["fit", "bundle", "bundle"]
    assert fresh["receipt"] == reused["receipt"] and reused["reused_training"]


@pytest.mark.parametrize(
    "mismatch", [False, True, "tagless", "pull", "mirror_mismatch"]
)
def test_pinned_image_repairs_a_stale_local_alias(
    tmp_path: Path, mismatch: bool | str
) -> None:
    script = (Path(__file__).parent / "jobs/run_pytorch_e2e.sh").read_text()
    function = re.search(
        r"^ensure_image\(\) \{.*?^\}", script, re.MULTILINE | re.DOTALL
    )
    assert function is not None
    fake = tmp_path / "docker"
    state = tmp_path / "state.json"
    digest = "sha256:" + "a" * 64
    pinned = "example:fixed@" + digest
    mirror_tag = "docker.m.daocloud.io/example:fixed"
    mirror = mirror_tag + "@" + digest
    actual = "sha256:wrong" if mismatch is True else "sha256:expected"
    state.write_text(json.dumps({"alias": "sha256:stale", "tags": []}))
    fake.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
        f"p=Path({str(state)!r}); s=json.loads(p.read_text()); a=sys.argv[1:]\n"
        "if a[0]=='pull':\n"
        f" assert a[1]=={mirror_tag!r}; s['pulled']=True; p.write_text(json.dumps(s)); sys.exit(0)\n"
        "if a[0]=='tag':\n"
        f" s['alias']={actual!r}; s['tags'].append(a[2]); p.write_text(json.dumps(s)); sys.exit(0)\n"
        f"assert a[:2]==['image','inspect'] and a[-1] in [{pinned!r},{mirror!r},{mirror_tag!r},'owned:local']\n"
        f"if '--format' not in a and a[-1]=={pinned!r} and {mismatch in {'pull', 'mirror_mismatch'}!r}: sys.exit(1)\n"
        "if '--format' not in a: print('[]')\n"
        f"elif 'RepoDigests' in a[3]: print('example@{digest}')\n"
        f"elif 'RepoTags' in a[3]: print({'' if mismatch == 'tagless' else 'persistent:base'!r})\n"
        f"else: print(s['alias'] if a[-1]=='owned:local' else 'sha256:wrong-mirror' if a[-1]=={mirror_tag!r} and {mismatch == 'mirror_mismatch'!r} else 'sha256:expected')\n"
    )
    fake.chmod(0o755)
    command = (
        f'set -euo pipefail\nDOCKER=("{fake}")\nTRIBUTO_E2E_EVIDENCE="$1"\n'
        + function.group()
        + '\nensure_image "$2" owned:local\n'
    )
    result = subprocess.run(
        ["bash", "-c", command, "verify", str(tmp_path), pinned],
        capture_output=True,
        text=True,
    )
    assert (result.returncode != 0) == (
        mismatch is True or mismatch in {"tagless", "mirror_mismatch"}
    ), result.stderr
    assert json.loads(state.read_text())["tags"] == (
        [] if mismatch in {"tagless", "mirror_mismatch"} else ["owned:local"]
    )
    assert bool(json.loads(state.read_text()).get("pulled")) == (
        mismatch in {"pull", "mirror_mismatch"}
    )
    assert 'ensure_image "$RAY_IMAGE" "$RAY_BASE_LOCAL"' in script
