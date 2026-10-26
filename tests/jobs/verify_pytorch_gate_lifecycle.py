"""Manual Docker check for pinned aliases and cancellation cleanup, without training."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    jobs = Path(__file__).resolve().parent
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }

    def docker(*command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *command],
            env=environment,
            capture_output=True,
            text=True,
            check=check,
            timeout=30,
        )

    def snapshot(prefix: str) -> None:
        for name, command in (
            ("images", ("image", "ls", "--all", "--no-trunc")),
            ("dangling", ("image", "ls", "--filter", "dangling=true", "--no-trunc")),
            ("docker-df", ("system", "df")),
        ):
            (evidence / f"{prefix}-{name}.log").write_text(docker(*command).stdout)

    snapshot("before")
    inputs = dict(
        line.split("=", 1)
        for line in (jobs / "pytorch-e2e.env").read_text().splitlines()
        if line and not line.startswith("#")
    )

    # No pull/build: fail before creating resources if the exact bases are absent.
    def cached_image(name: str) -> tuple[str, dict[str, Any]]:
        for reference in (name, "docker.m.daocloud.io/" + name):
            result = docker("image", "inspect", reference, check=False)
            if result.returncode == 0:
                return reference, json.loads(result.stdout)[0]
        raise ValueError(f"Pinned base image is not cached: {name}")

    ray_image, ray_metadata = cached_image(inputs["RAY_IMAGE"])
    minio_image, minio_metadata = cached_image(inputs["MINIO_IMAGE"])
    bases = {}
    for name, inspected in ((ray_image, ray_metadata), (minio_image, minio_metadata)):
        assert inspected.get("RepoTags"), (
            "Keep a pre-existing base tag before alias tests"
        )
        bases[name] = inspected["Id"]
    assert bases[ray_image] != bases[minio_image], "Alias fixture needs distinct bases"
    project = f"tributo-pytorch-p2-{int(time.time())}-{os.getpid()}"
    alias = project + "-base:local"
    assert docker("image", "inspect", alias, check=False).returncode != 0
    for kind in ("container", "network", "volume"):
        command = [
            kind,
            "ls",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={project}",
        ]
        if kind == "container":
            command.append("--all")
        assert not docker(*command).stdout.strip(), "Compose project already exists"
    temporary = Path(tempfile.mkdtemp(prefix="tributo-pytorch-p2-"))
    config = evidence / "compose.json"
    config.write_text(
        json.dumps(
            {
                "services": {
                    "probe": {
                        "image": alias,
                        "pull_policy": "never",
                        "mem_limit": "128m",
                        "entrypoint": ["python", "-c", "import time; time.sleep(3600)"],
                        "volumes": ["state:/state"],
                    }
                },
                "volumes": {"state": {}},
            },
            indent=2,
        )
        + "\n"
    )
    compose = ["compose", "--project-name", project, "--file", str(config)]
    controller: subprocess.Popen[bytes] | None = None
    alias_created = False
    success = False
    cleanup_errors: list[str] = []
    try:
        # Deliberately make a run-owned alias point at the wrong, pre-existing base.
        docker("tag", minio_image, alias)
        alias_created = True
        function = re.search(
            r"^ensure_image\(\) \{.*?^\}",
            (jobs / "run_pytorch_e2e.sh").read_text(),
            re.MULTILINE | re.DOTALL,
        )
        assert function is not None
        shell_command = (
            "set -euo pipefail\n"
            'cached_docker() { [[ "$1" != pull && "$1" != build ]] || return 99; command docker "$@"; }\n'
            'DOCKER=(cached_docker)\nTRIBUTO_E2E_EVIDENCE="$1"\n'
            + function.group()
            + '\nensure_image "$2" "$3"\n'
        )
        result = subprocess.run(
            ["bash", "-c", shell_command, "verify", str(evidence), ray_image, alias],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        (evidence / "alias.log").write_text(result.stdout + result.stderr)
        assert (
            json.loads(docker("image", "inspect", alias).stdout)[0]["Id"]
            == bases[ray_image]
        )
        ready, cleaned = temporary / "ready", temporary / "cleaned"
        shell = temporary / "child.sh"
        shell.write_text(
            "set -Eeuo pipefail\n"
            "cleanup() {\n"
            "  status=$?; trap - EXIT INT TERM; set +e\n"
            '  docker compose --project-name "$2" --file "$1" down --volumes --remove-orphans --timeout 5 > "$5" 2>&1\n'
            '  printf "%s" "$?" > "$4"; exit "$status"\n'
            "}\n"
            "trap 'cleanup \"$@\"' EXIT\n"
            "trap 'exit 143' TERM\n"
            'docker compose --project-name "$2" --file "$1" up --detach --no-build --pull never\n'
            'printf ready > "$3"\n'
            "while :; do sleep 1; done\n"
        )
        code = (
            "import importlib.util,os,sys; from pathlib import Path; "
            "s=importlib.util.spec_from_file_location('runner',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "raise SystemExit(m.run_gate(['bash',*sys.argv[2:]],cwd=Path.cwd(),env=dict(os.environ)))"
        )
        with (evidence / "controller.log").open("w") as output:
            controller = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(jobs / "run_pytorch_cpu_issues.py"),
                    str(shell),
                    str(config),
                    project,
                    str(ready),
                    str(cleaned),
                    str(evidence / "child-cleanup.log"),
                ],
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 40
            while (
                not ready.exists()
                and controller.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert ready.exists(), (
                "Compose fixture did not start; inspect controller.log"
            )
            assert docker(
                "container",
                "ls",
                "--quiet",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ).stdout.strip()
            controller.send_signal(signal.SIGTERM)
            assert controller.wait(timeout=50) == 143
        assert cleaned.read_text() == "0", "The shell cleanup trap did not complete"
        for kind in ("container", "network", "volume"):
            command = [
                kind,
                "ls",
                "--quiet",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ]
            if kind == "container":
                command.append("--all")
            assert not docker(*command).stdout.strip(), kind
        success = True
    finally:
        if controller is not None and controller.poll() is None:
            controller.terminate()
            try:
                controller.wait(timeout=50)
            except subprocess.TimeoutExpired:
                controller.kill()
                controller.wait()
                cleanup_errors.append("Controller failed to finish cleanup")
        fallback = docker(
            *compose,
            "down",
            "--volumes",
            "--remove-orphans",
            "--timeout",
            "5",
            check=False,
        )
        if fallback.returncode:
            cleanup_errors.append(fallback.stderr)
        if alias_created:
            result = docker("image", "rm", alias, check=False)
            if result.returncode:
                cleanup_errors.append(result.stderr)
        subprocess.run(["trash", str(temporary)], check=True)
        remaining = {}
        for kind in ("container", "network", "volume"):
            command = [
                kind,
                "ls",
                "--quiet",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ]
            if kind == "container":
                command.append("--all")
            remaining[kind] = docker(*command).stdout.split()
        preserved = all(
            json.loads(docker("image", "inspect", image).stdout)[0]["Id"] == image_id
            for image, image_id in bases.items()
        )
        alias_absent = docker("image", "inspect", alias, check=False).returncode != 0
        snapshot("after")
        report = {
            "status": "passed"
            if success
            and not cleanup_errors
            and not any(remaining.values())
            and preserved
            and alias_absent
            else "failed",
            "project": project,
            "remaining_owned_resources": remaining,
            "run_alias_absent": alias_absent,
            "shared_bases_preserved": preserved,
            "cleanup_errors": cleanup_errors,
        }
        (evidence / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if report["status"] != "passed" and sys.exc_info()[0] is None:
            raise RuntimeError("Lifecycle check failed; inspect result.json")


if __name__ == "__main__":
    main()
