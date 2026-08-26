"""Model containers shared by classical training and export."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SklearnModel:
    """A fitted estimator plus its stable ordered feature signature."""

    estimator: object
    feature_names: tuple[str, ...]
    task: str


@dataclass(frozen=True)
class TreeUnitModel:
    """One independently trained tree and final-forest metadata."""

    estimator: object
    feature_names: tuple[str, ...]
    task: str
    classes: tuple[object, ...]
    n_outputs: int


__all__ = ["SklearnModel", "TreeUnitModel"]
