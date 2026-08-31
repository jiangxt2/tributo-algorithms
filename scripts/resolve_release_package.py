"""Resolve one release tag to an exact workspace distribution."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_TAG_PATTERN = re.compile(
    r"^(?P<directory>[a-z0-9]+(?:-[a-z0-9]+)*)-v"
    r"(?P<version>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)$"
)


@dataclass(frozen=True)
class ReleasePackage:
    directory: str
    distribution: str
    version: str


def resolve_release_package(root: Path, tag: str) -> ReleasePackage:
    """Resolve and validate a strict ``<directory>-v<semver>`` tag."""
    matched = _TAG_PATTERN.fullmatch(tag)
    if matched is None:
        raise ValueError("release tag must match <directory>-v<major>.<minor>.<patch>")
    directory = matched.group("directory")
    version = ".".join(
        (matched.group("version"), matched.group("minor"), matched.group("patch"))
    )
    metadata_path = root / "packages" / directory / "pyproject.toml"
    if not metadata_path.is_file():
        raise ValueError(f"release tag names an unknown package directory: {directory}")
    with metadata_path.open("rb") as stream:
        project = tomllib.load(stream).get("project")
    if not isinstance(project, dict):
        raise ValueError(f"package {directory} has no project metadata")
    distribution = project.get("name")
    package_version = project.get("version")
    expected_distribution = f"tributo-algorithms-{directory}"
    if distribution != expected_distribution:
        raise ValueError(
            f"package {directory} declares distribution {distribution!r}, "
            f"expected {expected_distribution!r}"
        )
    if package_version != version:
        raise ValueError(
            f"tag version {version!r} does not match package version "
            f"{package_version!r}"
        )
    return ReleasePackage(
        directory=directory,
        distribution=expected_distribution,
        version=version,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--github-env", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    release = resolve_release_package(args.root.resolve(), args.tag)
    values = {
        "PACKAGE_DIRECTORY": release.directory,
        "DISTRIBUTION": release.distribution,
        "PACKAGE_VERSION": release.version,
    }
    if args.github_env is not None:
        with args.github_env.open("a", encoding="utf-8") as stream:
            for name, value in values.items():
                stream.write(f"{name}={value}\n")
    print(json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
