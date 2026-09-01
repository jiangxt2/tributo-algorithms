"""Verify descriptor-only conformance from an installed algorithm Wheel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from typing import ClassVar

from tributo.algorithms.conformance import validate_installed_algorithm_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--expected-entry-points", type=int)
    parser.add_argument("--require-scalar-single-column-binding", action="store_true")
    return parser


def _verify_scalar_single_column_binding() -> None:
    """Prove the installed Core can construct a rank-one model input."""
    import numpy as np
    from tributo.inference.contracts import (
        InputBindingSpec,
        OutputBindingSpec,
        TensorInputBinding,
        TensorOutputBinding,
    )
    from tributo.inference.kernel import KernelBatchPredictor

    class EchoKernel:
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            return {"output": inputs["input"]}

        def close(self) -> None:
            return None

    class EchoFactory:
        factory_id: ClassVar[str] = "source-free-scalar-contract"

        def create(self) -> EchoKernel:
            return EchoKernel()

    predictor = KernelBatchPredictor(
        EchoFactory(),
        InputBindingSpec(
            tensors=(
                TensorInputBinding(
                    tensor_name="input",
                    columns=("feature",),
                    dtype="float32",
                    single_column_mode="scalar",
                ),
            )
        ),
        OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="output",
                    column="result",
                    semantic="tensor",
                    dtype="float32",
                ),
            )
        ),
    )
    try:
        result = predictor({"feature": np.asarray([1.0, 2.0], dtype=np.float32)})
    finally:
        predictor.close()
    if result["result"].shape != (2,):
        raise ValueError("installed Tributo Core does not preserve scalar batch rank")


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
    if args.require_scalar_single_column_binding:
        _verify_scalar_single_column_binding()
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
