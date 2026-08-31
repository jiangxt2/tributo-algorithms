"""Verify descriptor-only conformance from an installed algorithm Wheel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json

from tributo.algorithms.conformance import validate_installed_algorithm_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--expected-entry-points", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    distribution = importlib.metadata.distribution(args.distribution)
    entry_points = tuple(
        entry_point
        for entry_point in distribution.entry_points
        if entry_point.group == "tributo.algorithms"
    )
    if not entry_points:
        raise ValueError(f"{args.distribution} has no tributo.algorithms Entry Points")
    if (
        args.expected_entry_points is not None
        and len(entry_points) != args.expected_entry_points
    ):
        raise ValueError(
            f"{args.distribution} has {len(entry_points)} algorithm Entry Points, "
            f"expected {args.expected_entry_points}"
        )
    reports = [
        validate_installed_algorithm_package(
            entry_point.load(),
            entry_point_name=entry_point.name,
        )
        for entry_point in entry_points
    ]
    if any(report.implementation_loaded for report in reports):
        raise ValueError("descriptor-only verification loaded an implementation")
    print(
        json.dumps(
            {
                "distribution": distribution.metadata["Name"],
                "entry_point_count": len(reports),
                "version": distribution.version,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
