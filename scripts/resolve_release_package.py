"""Resolve the aggregate distribution version from its release tag."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_DISTRIBUTION = "tributo-algorithms"
_TAG_PATTERN = re.compile(
    r"^tributo-algorithms-v(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)$"
)


@dataclass(frozen=True)
class ReleasePackage:
    distribution: str
    version: str


def resolve_release_package(root: Path, tag: str) -> ReleasePackage:
    """Resolve and validate a strict ``tributo-algorithms-v<semver>`` tag."""
    matched = _TAG_PATTERN.fullmatch(tag)
    if matched is None:
        raise ValueError(
            "release tag must match tributo-algorithms-v<major>.<minor>.<patch>"
        )
    version = ".".join(
        (matched.group("major"), matched.group("minor"), matched.group("patch"))
    )
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream).get("project")
    if not isinstance(project, dict):
        raise ValueError("root pyproject.toml has no project metadata")
    distribution = project.get("name")
    package_version = project.get("version")
    if distribution != _DISTRIBUTION:
        raise ValueError(
            f"root project declares distribution {distribution!r}, "
            f"expected {_DISTRIBUTION!r}"
        )
    if package_version != version:
        raise ValueError(
            f"tag version {version!r} does not match project version "
            f"{package_version!r}"
        )
    return ReleasePackage(distribution=_DISTRIBUTION, version=version)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--github-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    release = resolve_release_package(args.root.resolve(), args.tag)
    values = {"distribution": release.distribution, "version": release.version}
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as stream:
            for name, value in values.items():
                stream.write(f"{name}={value}\n")
    print(json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
