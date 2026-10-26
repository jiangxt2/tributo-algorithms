"""Build one Core CPU candidate and run the focused fixed-Wheel issue Gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_TIMEOUT = "600"
CPU_CASES = (
    "package",
    "transformer_unit",
    "transformer_ddp_alignment",
    "core_runtime_profile",
    "token_transformer_classifier",
    "dnn",
    "graphsage_node_classifier",
)


def with_download_settings(command: list[str]) -> list[str]:
    """Scope download overrides to this test build and preserve its context."""
    return [
        *command[:-1],
        "--build-arg",
        f"PIP_INDEX_URL={PIP_INDEX_URL}",
        "--build-arg",
        f"PIP_DEFAULT_TIMEOUT={PIP_TIMEOUT}",
        command[-1],
    ]


def gate_command(algorithms: Path, cases: tuple[str, ...]) -> list[str]:
    """Pass case selection through the shell entry to the actual job parser."""
    return [
        "bash",
        str(algorithms / "tests/jobs/run_pytorch_e2e.sh"),
        "--cases",
        *cases,
    ]


def run_gate(
    command: list[str], *, cwd: Path, env: dict[str, str], cleanup_timeout: float = 45
) -> int:
    """Let the shell's cleanup trap finish before cleaning its parent image."""
    child: subprocess.Popen[bytes] | None = None
    cancelled: int | None = None
    requested_at = 0.0

    def signal_child(signum: int) -> None:
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal cancelled, requested_at
        if cancelled is None:
            cancelled = _signum
            requested_at = time.monotonic()
            signal_child(signal.SIGTERM)

    handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in handlers:
        signal.signal(signum, stop)
    try:
        child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
        if cancelled is not None:
            signal_child(signal.SIGTERM)
        while True:
            try:
                result = child.wait(timeout=0.1)
                return 128 + cancelled if cancelled is not None else result
            except subprocess.TimeoutExpired:
                if (
                    cancelled is not None
                    and time.monotonic() - requested_at > cleanup_timeout
                ):
                    signal_child(signal.SIGKILL)
                    child.wait()
                    raise RuntimeError(
                        "Gate cleanup exceeded its timeout; inspect the owned project"
                    ) from None
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def require_cached_steps(output: str) -> None:
    """Refuse reuse evidence if any Dockerfile RUN had to execute again."""
    steps = re.findall(r"^(#\d+) \[.*\] RUN ", output, re.MULTILINE)
    if not steps or any(f"{step} CACHED" not in output.splitlines() for step in steps):
        raise ValueError("Core recovery did not fully reuse cached RUN steps")


def require_image_digest(actual: str, expected: str) -> None:
    if actual != f"sha256:{expected}":
        raise ValueError("Recovered Core image differs from the passed image")


def restore_core_image(
    builder: Any,
    *,
    core: Path,
    previous: Path,
    evidence: Path,
    image: str,
    docker_env: dict[str, str],
) -> str:
    """Restore the sealed build only; reuse its successful validation record."""
    manifest = json.loads((previous / "core-image/manifest.json").read_text())
    profile = json.loads((previous / "core-image/image-profile.json").read_text())
    if profile["image_digest"] != manifest["image_digest"]:
        raise ValueError("Passed Core manifest and profile disagree")
    config = builder.load_config(previous / "base-image-config.json", root=core)
    log = builder._CommandLog()
    exported = builder._host_dependency_export(config, root=core, log=log)
    expected = builder._manifest_core(
        config,
        root=core,
        dependency_export=exported,
        distributions=manifest["installed_distributions"],
        external_wheels=builder.wheel_records(config.external_wheelhouse),
    )
    if (
        any(manifest.get(key) != value for key, value in expected.items())
        or builder.sha256_bytes(builder.canonical_json(expected))
        != manifest["manifest_sha256"]
    ):
        raise ValueError("Current Core inputs differ from the passed manifest")
    with (
        builder.prepared_wheelhouse(config.external_wheelhouse) as external,
        builder.prepared_requirements(exported) as requirements,
        builder.prepared_project_wheelhouse(root=core, log=log) as wheels,
    ):
        command = builder.build_command(
            replace(config, image=image),
            root=core,
            wheelhouse_context=external,
            requirements_context=requirements,
            project_wheelhouse_context=wheels,
            manifest_sha256=manifest["manifest_sha256"],
        )
        command.insert(-1, "--progress=plain")
        log.command(command)
        output_path = evidence / "cache-restore.log"
        with output_path.open("w") as output:
            subprocess.run(
                command,
                cwd=core,
                env=docker_env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=60,
            )
        require_cached_steps(output_path.read_text())
    (evidence / "reused-core-validation.json").write_text(
        json.dumps(
            {
                "source": str(previous / "core-image"),
                "expected_image_digest": manifest["image_digest"],
                "manifest_sha256": manifest["manifest_sha256"],
                "validation_rerun": False,
                "all_run_steps_cached": True,
            },
            indent=2,
        )
        + "\n"
    )
    return str(manifest["image_digest"])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reuse-core-build", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CPU_CASES, default=CPU_CASES)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    core = args.core_dir.resolve()
    algorithms = Path(__file__).resolve().parents[2]
    evidence = args.evidence.resolve()
    if evidence.exists():
        raise ValueError("Evidence directory must be new")
    for root in (core, algorithms):
        actual = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if actual != str(root):
            raise ValueError("Expected an explicit repository worktree")
    evidence.mkdir(parents=True)
    docker_env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }

    def docker(
        command: list[str], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *command],
            env=docker_env,
            capture_output=True,
            text=True,
            check=check,
        )

    def snapshot(prefix: str) -> None:
        for name, command in (
            (
                "images",
                ["image", "ls", "--all", "--no-trunc", "--format", "{{json .}}"],
            ),
            (
                "dangling",
                ["image", "ls", "--filter", "dangling=true", "--no-trunc", "--quiet"],
            ),
            ("docker-df", ["system", "df"]),
        ):
            (evidence / f"{prefix}-{name}.txt").write_text(docker(command).stdout)

    snapshot("before")
    baseline = set(
        docker(["image", "ls", "--all", "--no-trunc", "--quiet"]).stdout.split()
    )
    image = f"tributo-cpu-issues-{int(time.time())}-{os.getpid()}:local"
    if docker(["image", "inspect", image], check=False).returncode == 0:
        raise ValueError("Fresh candidate image tag already exists")
    owned: dict[str, dict[str, Any]] = {}
    sys.path.insert(0, str(core))
    sys.path.insert(0, str(core / "src"))
    from tools import build_tributo_image as builder

    original_log = builder._CommandLog
    original_build_command = builder.build_command

    def build_with_download_settings(*args: Any, **kwargs: Any) -> list[str]:
        return with_download_settings(original_build_command(*args, **kwargs))

    def record_images(values: Any) -> None:
        if not isinstance(values, list):
            return
        for value in values:
            if isinstance(value, dict) and image in (value.get("RepoTags") or []):
                owned[value["Id"]] = {
                    "created": value["Created"],
                    "labels": value.get("Config", {}).get("Labels") or {},
                    "tag_observed": image,
                }
                (evidence / "owned-images.json").write_text(json.dumps(owned, indent=2))

    class TrackingLog(builder._CommandLog):
        def command(self, command: Any) -> None:
            super().command(command)
            (evidence / "builder-live.log").write_text("\n".join(self.lines) + "\n")

        def result(self, result: Any) -> None:
            super().result(result)
            (evidence / "builder-live.log").write_text("\n".join(self.lines) + "\n")
            try:
                values = json.loads(result.stdout or "null")
            except json.JSONDecodeError:
                return
            record_images(values)

    builder._CommandLog = TrackingLog
    builder.build_command = build_with_download_settings
    config_path = evidence / "base-image-config.json"
    payload = json.loads((core / "tools/tributo-runtime-full.json").read_text())
    payload.update(image=image, platform="linux/arm64")
    config_path.write_text(json.dumps(payload, indent=2) + "\n")
    test_exit = 1
    cleanup_errors = []

    def terminate(_signum: int, _frame: Any) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    try:
        config = builder.load_config(config_path, root=core)
        if args.reuse_core_build is None:
            builder.build_image(config, root=core, output_dir=evidence / "core-image")
        else:
            try:
                expected_digest = restore_core_image(
                    builder,
                    core=core,
                    previous=args.reuse_core_build.resolve(),
                    evidence=evidence,
                    image=image,
                    docker_env=docker_env,
                )
            finally:
                inspected = docker(["image", "inspect", image], check=False)
                if inspected.returncode == 0:
                    record_images(json.loads(inspected.stdout))
            require_image_digest(json.loads(inspected.stdout)[0]["Id"], expected_digest)
        child_env = dict(os.environ)
        for name in ("TRIBUTO_E2E_DEPENDENCY_EXPORT", "TRIBUTO_E2E_REUSE_RESULTS"):
            child_env.pop(name, None)
        child_env.update(
            TRIBUTO_CORE_DIR=str(core),
            TRIBUTO_E2E_EVIDENCE=str(evidence / "ray-jobs"),
            TRIBUTO_E2E_BASE_IMAGE=image,
            TRIBUTO_E2E_VENV_PATH="/opt/tributo/.venv",
            TRIBUTO_E2E_PIP_INDEX_URL=PIP_INDEX_URL,
            TRIBUTO_E2E_PIP_TIMEOUT=PIP_TIMEOUT,
        )
        test_exit = run_gate(
            gate_command(algorithms, tuple(args.cases)),
            cwd=algorithms,
            env=child_env,
        )
        if test_exit:
            raise SystemExit(test_exit)
    finally:
        builder._CommandLog = original_log
        builder.build_command = original_build_command
        # The builder has discovery and sealed manifests. Both exact IDs were
        # observed with our unique tag; shared pre-existing IDs are retained.
        tagged = docker(["image", "inspect", image], check=False)
        if tagged.returncode == 0:
            value = json.loads(tagged.stdout)[0]
            if value["Id"] in owned:
                deleted = docker(["image", "rm", image], check=False)
                if deleted.returncode:
                    cleanup_errors.append(deleted.stderr)
            else:
                cleanup_errors.append("Candidate tag has an unrecorded image ID")
        for image_id, record in reversed(list(owned.items())):
            if image_id in baseline:
                continue
            result = docker(["image", "inspect", image_id], check=False)
            if result.returncode:
                continue
            value = json.loads(result.stdout)[0]
            if (
                value.get("RepoTags")
                or value["Created"] != record["created"]
                or (value.get("Config", {}).get("Labels") or {}) != record["labels"]
            ):
                cleanup_errors.append(f"Refused ambiguous image cleanup: {image_id}")
                continue
            deleted = docker(["image", "rm", image_id], check=False)
            if deleted.returncode:
                cleanup_errors.append(deleted.stderr)
        try:
            snapshot("after")
        except (OSError, subprocess.CalledProcessError) as exc:
            cleanup_errors.append(f"Final Docker snapshot failed: {type(exc).__name__}")
        (evidence / "cleanup.json").write_text(
            json.dumps(
                {
                    "test_exit": test_exit,
                    "owned_image_ids": list(owned),
                    "cleanup_errors": cleanup_errors,
                    "shared_base_images_and_build_cache": "retained",
                },
                indent=2,
            )
            + "\n"
        )
        if cleanup_errors and test_exit == 0:
            raise RuntimeError("Image cleanup was incomplete; inspect cleanup.json")


if __name__ == "__main__":
    main()
