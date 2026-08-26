"""Random Forest Hook implementations for Joblib and native unit runtimes."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from tributo.algorithms import (
    EnsembleUnitSpec,
    JoblibEstimatorRecipe,
    ParallelEnsembleAlgorithm,
)
from tributo.algorithms.spi import (
    AlgorithmExecutionContext,
    MaterializedTabularInputView,
)

from tributo_algorithms_classical.models import SklearnModel, TreeUnitModel


class PickleCodec:
    """Serialize bounded sklearn state for runtime digests and checkpoints."""

    def dumps(self, value: object) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, payload: bytes) -> object:
        return pickle.loads(payload)


def _arrays(
    inputs: Mapping[str, object],
) -> tuple[MaterializedTabularInputView, np.ndarray, np.ndarray]:
    view = cast(MaterializedTabularInputView, inputs["train"])
    columns = view.columns()
    features = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in view.feature_names]
    )
    if view.label_name is None:
        raise ValueError("Random Forest requires a label")
    labels = np.asarray(columns[view.label_name])
    return view, features, labels


class RandomForestJoblibRecipe(JoblibEstimatorRecipe):
    """Use sklearn's public Random Forest estimator with Ray Joblib."""

    def build_estimator(self, config: Mapping[str, Any]) -> object:
        common = {
            "n_estimators": int(config.get("n_estimators", 32)),
            "max_depth": config.get("max_depth"),
            "max_features": config.get("max_features", "sqrt"),
            "random_state": int(config.get("seed", 42)),
        }
        if config.get("task", "classification") == "regression":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**common)
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            **common,
            class_weight=config.get("class_weight"),
        )

    def fit_arguments(
        self,
        inputs: Mapping[str, object],
        config: Mapping[str, Any],
    ) -> tuple[tuple[object, ...], Mapping[str, object]]:
        del config
        view, features, labels = _arrays(inputs)
        self._feature_names = tuple(view.feature_names)
        return (features, labels), {}

    def parallelism_contract(self) -> Mapping[str, object]:
        return {"fit_operations": ("fit",), "exactness": "exact"}

    def extract_model(self, fitted_estimator: object) -> object:
        estimator = cast(Any, fitted_estimator)
        feature_names = tuple(
            str(name) for name in getattr(estimator, "feature_names_in_", ())
        )
        if not feature_names:
            feature_names = tuple(getattr(self, "_feature_names", ()))
        if not feature_names:
            feature_count = int(estimator.n_features_in_)
            feature_names = tuple(f"feature_{index}" for index in range(feature_count))
        task = "classification" if hasattr(estimator, "classes_") else "regression"
        return SklearnModel(estimator, feature_names, task)

    def model_codec(self) -> object:
        return PickleCodec()


class ExtraTreesJoblibRecipe(RandomForestJoblibRecipe):
    """Use sklearn's randomized Extra Trees estimator with Ray Joblib."""

    def build_estimator(self, config: Mapping[str, Any]) -> object:
        common = {
            "n_estimators": int(config.get("n_estimators", 32)),
            "max_depth": config.get("max_depth"),
            "max_features": config.get("max_features", "sqrt"),
            "random_state": int(config.get("seed", 42)),
        }
        if config.get("task", "classification") == "regression":
            from sklearn.ensemble import ExtraTreesRegressor

            return ExtraTreesRegressor(**common)
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            **common,
            class_weight=config.get("class_weight"),
        )


class RandomForestEnsemble(ParallelEnsembleAlgorithm[TreeUnitModel, SklearnModel]):
    """Expose one deterministic bootstrap tree as each independent unit."""

    def plan_units(
        self,
        config: Mapping[str, Any],
        input_descriptor: object,
        seed: int,
    ) -> tuple[EnsembleUnitSpec, ...]:
        del input_descriptor
        count = int(config.get("unit_count", config.get("n_estimators", 32)))
        seeds = np.random.RandomState(seed).randint(
            np.iinfo(np.int32).max,
            size=count,
        )
        payload = {
            "task": str(config.get("task", "classification")),
            "max_depth": config.get("max_depth"),
            "max_features": config.get("max_features", "sqrt"),
            "class_weight": config.get("class_weight"),
        }
        return tuple(
            EnsembleUnitSpec(
                unit_id=f"tree-{index}",
                seed=int(tree_seed),
                payload=payload,
            )
            for index, tree_seed in enumerate(seeds)
        )

    def fit_unit(
        self,
        unit: EnsembleUnitSpec,
        inputs: Mapping[str, object],
        context: AlgorithmExecutionContext,
    ) -> TreeUnitModel:
        del context
        view, features, labels = _arrays(inputs)
        sample_indices = np.random.RandomState(unit.seed).randint(
            0,
            len(labels),
            len(labels),
        )
        task = str(unit.payload.get("task", "classification"))
        if task == "regression":
            from sklearn.tree import DecisionTreeRegressor

            estimator = DecisionTreeRegressor(
                max_depth=unit.payload.get("max_depth"),
                max_features=unit.payload.get("max_features", "sqrt"),
                random_state=unit.seed,
            )
            classes: tuple[object, ...] = ()
        else:
            from sklearn.tree import DecisionTreeClassifier

            estimator = DecisionTreeClassifier(
                max_depth=unit.payload.get("max_depth"),
                max_features=unit.payload.get("max_features", "sqrt"),
                random_state=unit.seed,
                class_weight=unit.payload.get("class_weight"),
            )
            classes = tuple(np.unique(labels).tolist())
        estimator.fit(features[sample_indices], labels[sample_indices])
        return TreeUnitModel(
            estimator=estimator,
            feature_names=view.feature_names,
            task=task,
            classes=classes,
            n_outputs=1,
        )

    def merge_units(self, ordered_units: tuple[TreeUnitModel, ...]) -> object:
        return ordered_units

    def finalize_ensemble(self, merged: object) -> SklearnModel:
        units = cast(tuple[TreeUnitModel, ...], merged)
        if not units:
            raise ValueError("Random Forest requires at least one tree")
        first = units[0]
        if any(
            unit.feature_names != first.feature_names or unit.task != first.task
            for unit in units
        ):
            raise ValueError("Random Forest unit metadata is inconsistent")
        if first.task == "regression":
            from sklearn.ensemble import RandomForestRegressor

            forest: Any = RandomForestRegressor(n_estimators=len(units))
        else:
            from sklearn.ensemble import RandomForestClassifier

            forest = RandomForestClassifier(n_estimators=len(units))
            forest.classes_ = np.asarray(first.classes)
            forest.n_classes_ = len(first.classes)
        forest.estimators_ = [unit.estimator for unit in units]
        forest.n_features_in_ = len(first.feature_names)
        forest.n_outputs_ = first.n_outputs
        return SklearnModel(forest, first.feature_names, first.task)

    def unit_schema(self) -> Mapping[str, object]:
        return {
            "estimator": "sklearn.tree",
            "feature_names": "tuple[str]",
            "task": "classification|regression",
        }

    @property
    def retry_safe(self) -> bool:
        return True


class ExtraTreesEnsemble(RandomForestEnsemble):
    """Fit independent randomized Extra Trees without bootstrap sampling."""

    def fit_unit(
        self,
        unit: EnsembleUnitSpec,
        inputs: Mapping[str, object],
        context: AlgorithmExecutionContext,
    ) -> TreeUnitModel:
        del context
        view, features, labels = _arrays(inputs)
        task = str(unit.payload.get("task", "classification"))
        if task == "regression":
            from sklearn.tree import ExtraTreeRegressor

            estimator = ExtraTreeRegressor(
                max_depth=unit.payload.get("max_depth"),
                max_features=unit.payload.get("max_features", "sqrt"),
                random_state=unit.seed,
            )
            classes: tuple[object, ...] = ()
        else:
            from sklearn.tree import ExtraTreeClassifier

            estimator = ExtraTreeClassifier(
                max_depth=unit.payload.get("max_depth"),
                max_features=unit.payload.get("max_features", "sqrt"),
                random_state=unit.seed,
                class_weight=unit.payload.get("class_weight"),
            )
            classes = tuple(np.unique(labels).tolist())
        estimator.fit(features, labels)
        return TreeUnitModel(
            estimator=estimator,
            feature_names=view.feature_names,
            task=task,
            classes=classes,
            n_outputs=1,
        )

    def finalize_ensemble(self, merged: object) -> SklearnModel:
        units = cast(tuple[TreeUnitModel, ...], merged)
        if not units:
            raise ValueError("Extra Trees requires at least one tree")
        first = units[0]
        if any(
            unit.feature_names != first.feature_names or unit.task != first.task
            for unit in units
        ):
            raise ValueError("Extra Trees unit metadata is inconsistent")
        if first.task == "regression":
            from sklearn.ensemble import ExtraTreesRegressor

            ensemble: Any = ExtraTreesRegressor(n_estimators=len(units))
        else:
            from sklearn.ensemble import ExtraTreesClassifier

            ensemble = ExtraTreesClassifier(n_estimators=len(units))
            ensemble.classes_ = np.asarray(first.classes)
            ensemble.n_classes_ = len(first.classes)
        ensemble.estimators_ = [unit.estimator for unit in units]
        ensemble.n_features_in_ = len(first.feature_names)
        ensemble.n_outputs_ = first.n_outputs
        return SklearnModel(ensemble, first.feature_names, first.task)


__all__ = [
    "ExtraTreesEnsemble",
    "ExtraTreesJoblibRecipe",
    "PickleCodec",
    "RandomForestEnsemble",
    "RandomForestJoblibRecipe",
]
