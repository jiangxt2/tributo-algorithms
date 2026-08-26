"""Exact distributed difference-in-means causal estimator."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    ArtifactDraft,
    ContractBinding,
    ContractBindingSet,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MapReducePolicy,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    StateField,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.spi import AlgorithmExecutionContext, MapReduceAlgorithm
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

_PACKAGE = "tributo-algorithms-causal-core"
_VERSION = "0.1.0"
_STATE_SCHEMA = (
    StateField("treated_count", "int64", ()),
    StateField("control_count", "int64", ()),
    StateField("treated_sum", "float64", ()),
    StateField("control_sum", "float64", ()),
    StateField("treated_sum_squares", "float64", ()),
    StateField("control_sum_squares", "float64", ()),
    StateField("row_count", "int64", ()),
)
_DML_STATE_SCHEMA = (
    StateField("xtx", "float64", (None, None)),
    StateField("xty", "float64", (None,)),
    StateField("xtt", "float64", (None,)),
    StateField("yty", "float64", ()),
    StateField("ytt", "float64", ()),
    StateField("ttt", "float64", ()),
    StateField("treated_count", "int64", ()),
    StateField("control_count", "int64", ()),
    StateField("treated_sum", "float64", ()),
    StateField("control_sum", "float64", ()),
    StateField("row_count", "int64", ()),
    StateField("fold_xtx", "float64", (None, None, None)),
    StateField("fold_xty", "float64", (None, None)),
    StateField("fold_xtt", "float64", (None, None)),
    StateField("fold_yty", "float64", (None,)),
    StateField("fold_ytt", "float64", (None,)),
    StateField("fold_ttt", "float64", (None,)),
)
_IV_STATE_SCHEMA = (
    StateField("xtx", "float64", (None, None)),
    StateField("xty", "float64", (None,)),
    StateField("xtt", "float64", (None,)),
    StateField("xtz", "float64", (None,)),
    StateField("yty", "float64", ()),
    StateField("ytt", "float64", ()),
    StateField("ttt", "float64", ()),
    StateField("ytz", "float64", ()),
    StateField("ttz", "float64", ()),
    StateField("ztz", "float64", ()),
    StateField("treated_count", "int64", ()),
    StateField("control_count", "int64", ()),
    StateField("treated_sum", "float64", ()),
    StateField("control_sum", "float64", ()),
    StateField("row_count", "int64", ()),
)


@dataclass(frozen=True)
class CausalATEModel:
    method: str
    treatment: str
    outcome: str
    feature_names: tuple[str, ...]
    treated_count: int
    control_count: int
    treated_mean: float
    control_mean: float
    effect: float
    standard_error: float
    confidence_interval: tuple[float, float]
    policy_cost: float
    treat_all_policy: bool
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def report(self) -> dict[str, object]:
        if self.standard_error > 0:
            null_z_score = self.effect / self.standard_error
            null_p_value = math.erfc(abs(null_z_score) / math.sqrt(2.0))
            signal_to_noise = abs(null_z_score)
        else:
            null_z_score = None
            null_p_value = 0.0 if self.effect else 1.0
            signal_to_noise = None
        return {
            "api_version": 1,
            "problem": {
                "treatment": self.treatment,
                "outcome": self.outcome,
                "estimand": "ATE",
                "assumptions": [
                    "consistency",
                    "exchangeability_or_random_assignment",
                    "positivity",
                ],
            },
            "estimate": {
                "method": self.method,
                "effect": self.effect,
                "standard_error": self.standard_error,
                "confidence_interval": list(self.confidence_interval),
                "treated_mean": self.treated_mean,
                "control_mean": self.control_mean,
            },
            "coverage": {
                "treated": self.treated_count,
                "control": self.control_count,
            },
            "policy": {
                "cost": self.policy_cost,
                "treat_all": self.treat_all_policy,
            },
            "refutation": {
                "method": "analytic_null_effect",
                "null_effect": 0.0,
                "z_score": null_z_score,
                "two_sided_normal_p_value": null_p_value,
                "rejects_null_at_0_05": null_p_value < 0.05,
            },
            "sensitivity": {
                "method": "estimate_to_standard_error_ratio",
                "signal_to_noise": signal_to_noise,
            },
            "diagnostics": dict(sorted(self.diagnostics.items())),
        }


class DifferenceInMeansATE(
    MapReduceAlgorithm[Mapping[str, object], Mapping[str, object], CausalATEModel]
):
    """Merge exact treated/control sufficient statistics across Ray Workers."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        binding = plan.primary_input_binding
        self.treatment = str(plan.algorithm_config["treatment_col"])
        self.outcome = binding.label_name or ""
        self.feature_names = binding.feature_names
        if not self.outcome or self.treatment not in self.feature_names:
            raise AlgorithmConfigurationError(
                "ATE InputBinding must contain treatment and outcome columns"
            )

    def map_partition(
        self,
        batches: Iterable[Mapping[str, object]],
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        del context
        import numpy as np

        state = self.empty_partition()
        for batch in batches:
            if self.treatment not in batch or self.outcome not in batch:
                raise AlgorithmInputError(
                    "causal batch is missing treatment or outcome"
                )
            treatment = np.asarray(batch[self.treatment])
            outcome = np.asarray(batch[self.outcome], dtype=np.float64)
            if (
                treatment.ndim != 1
                or outcome.ndim != 1
                or treatment.shape != outcome.shape
            ):
                raise AlgorithmInputError("causal treatment and outcome rows disagree")
            if not np.isfinite(outcome).all() or not np.isin(treatment, (0, 1)).all():
                raise AlgorithmInputError(
                    "causal outcome must be finite and treatment must be binary"
                )
            treated = outcome[treatment == 1]
            control = outcome[treatment == 0]
            state = self.merge_states(
                state,
                {
                    "treated_count": np.asarray(treated.size, dtype=np.int64),
                    "control_count": np.asarray(control.size, dtype=np.int64),
                    "treated_sum": np.asarray(treated.sum(), dtype=np.float64),
                    "control_sum": np.asarray(control.sum(), dtype=np.float64),
                    "treated_sum_squares": np.asarray(
                        (treated**2).sum(), dtype=np.float64
                    ),
                    "control_sum_squares": np.asarray(
                        (control**2).sum(), dtype=np.float64
                    ),
                    "row_count": np.asarray(outcome.size, dtype=np.int64),
                },
            )
        return state

    def merge_states(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        import numpy as np

        return {
            field.name: np.asarray(
                np.asarray(left[field.name], dtype=field.dtype)
                + np.asarray(right[field.name], dtype=field.dtype),
                dtype=field.dtype,
            )
            for field in _STATE_SCHEMA
        }

    def finalize_model(self, state: Mapping[str, object]) -> CausalATEModel:
        import numpy as np

        treated_count = int(np.asarray(state["treated_count"]))
        control_count = int(np.asarray(state["control_count"]))
        if treated_count < 2 or control_count < 2:
            raise AlgorithmInputError(
                "ATE requires at least two treated and two control observations"
            )
        treated_sum = float(np.asarray(state["treated_sum"]))
        control_sum = float(np.asarray(state["control_sum"]))
        treated_mean = treated_sum / treated_count
        control_mean = control_sum / control_count
        treated_variance = max(
            0.0,
            (
                float(np.asarray(state["treated_sum_squares"]))
                - treated_count * treated_mean**2
            )
            / (treated_count - 1),
        )
        control_variance = max(
            0.0,
            (
                float(np.asarray(state["control_sum_squares"]))
                - control_count * control_mean**2
            )
            / (control_count - 1),
        )
        effect = treated_mean - control_mean
        standard_error = math.sqrt(
            treated_variance / treated_count + control_variance / control_count
        )
        z = float(self.plan.algorithm_config.get("confidence_z", 1.959963984540054))
        policy_cost = float(self.plan.algorithm_config.get("policy_cost", 0.0))
        return CausalATEModel(
            method="distributed_difference_in_means",
            treatment=self.treatment,
            outcome=self.outcome,
            feature_names=self.feature_names,
            treated_count=treated_count,
            control_count=control_count,
            treated_mean=treated_mean,
            control_mean=control_mean,
            effect=effect,
            standard_error=standard_error,
            confidence_interval=(
                effect - z * standard_error,
                effect + z * standard_error,
            ),
            policy_cost=policy_cost,
            treat_all_policy=effect > policy_cost,
        )

    def state_schema(self) -> tuple[StateField, ...]:
        return _STATE_SCHEMA

    def empty_partition(self) -> Mapping[str, object]:
        import numpy as np

        return {field.name: np.asarray(0, dtype=field.dtype) for field in _STATE_SCHEMA}

    def coverage_counts(self, state: Mapping[str, object]) -> Mapping[str, int]:
        import numpy as np

        return {
            "treated": int(np.asarray(state["treated_count"])),
            "control": int(np.asarray(state["control_count"])),
        }

    @property
    def retry_safe(self) -> bool:
        return True


class LinearDMLATE(
    MapReduceAlgorithm[Mapping[str, object], Mapping[str, object], CausalATEModel]
):
    """Estimate a partially linear ATE from distributed normal equations."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        binding = plan.primary_input_binding
        self.treatment = str(plan.algorithm_config["treatment_col"])
        self.outcome = binding.label_name or ""
        self.feature_names = binding.feature_names
        self.fold_count = int(self.plan.algorithm_config.get("cross_fit_folds", 5))
        if self.fold_count < 1 or self.fold_count > 20:
            raise AlgorithmConfigurationError(
                "cross_fit_folds must be between 1 and 20"
            )
        self.fold_column = self.plan.algorithm_config.get("fold_column")
        if self.fold_column is not None and not isinstance(self.fold_column, str):
            raise AlgorithmConfigurationError("fold_column must be a string")
        self.confounders = tuple(
            name
            for name in self.feature_names
            if name != self.treatment and name != self.fold_column
        )
        if not self.outcome or self.treatment not in self.feature_names:
            raise AlgorithmConfigurationError(
                "Linear DML requires treatment features and outcome label"
            )

    def map_partition(
        self,
        batches: Iterable[Mapping[str, object]],
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        del context
        import numpy as np

        state = self.empty_partition()
        for batch in batches:
            required = (*self.confounders, self.treatment, self.outcome)
            if self.fold_column is not None:
                required = (*required, self.fold_column)
            missing = [name for name in required if name not in batch]
            if missing:
                raise AlgorithmInputError(f"Linear DML batch is missing: {missing}")
            treatment = np.asarray(batch[self.treatment], dtype=np.float64)
            outcome = np.asarray(batch[self.outcome], dtype=np.float64)
            confounders = (
                np.column_stack(
                    [
                        np.asarray(batch[name], dtype=np.float64)
                        for name in self.confounders
                    ]
                )
                if self.confounders
                else np.empty((outcome.size, 0), dtype=np.float64)
            )
            design = np.column_stack(
                [np.ones(outcome.size, dtype=np.float64), confounders]
            )
            if (
                treatment.ndim != 1
                or outcome.ndim != 1
                or design.shape[0] != outcome.size
                or treatment.shape != outcome.shape
                or not np.isfinite(design).all()
                or not np.isfinite(outcome).all()
                or not np.isin(treatment, (0.0, 1.0)).all()
            ):
                raise AlgorithmInputError("Linear DML input is invalid")
            treated = treatment == 1
            control = ~treated
            partial = {
                "xtx": design.T @ design,
                "xty": design.T @ outcome,
                "xtt": design.T @ treatment,
                "yty": np.asarray(outcome @ outcome, dtype=np.float64),
                "ytt": np.asarray(outcome @ treatment, dtype=np.float64),
                "ttt": np.asarray(treatment @ treatment, dtype=np.float64),
                "treated_count": np.asarray(treated.sum(), dtype=np.int64),
                "control_count": np.asarray(control.sum(), dtype=np.int64),
                "treated_sum": np.asarray(outcome[treated].sum(), dtype=np.float64),
                "control_sum": np.asarray(outcome[control].sum(), dtype=np.float64),
                "row_count": np.asarray(outcome.size, dtype=np.int64),
            }
            if self.fold_count > 1:
                fold_values = (
                    np.asarray(batch[self.fold_column])
                    if self.fold_column is not None and self.fold_column in batch
                    else np.floor(
                        np.abs(
                            np.sum(
                                np.column_stack(
                                    [
                                        treatment,
                                        *[
                                            np.asarray(batch[name], dtype=np.float64)
                                            for name in self.confounders
                                        ],
                                    ]
                                )
                                * np.arange(1, len(self.confounders) + 2),
                                axis=1,
                            )
                            * 1_000_003
                        )
                    ).astype(np.int64)
                )
                if self.fold_column is not None:
                    try:
                        fold_values = np.asarray(fold_values, dtype=np.int64)
                    except (TypeError, ValueError):
                        fold_values = np.asarray(
                            [
                                int(
                                    hashlib.sha256(
                                        str(value).encode("utf-8")
                                    ).hexdigest(),
                                    16,
                                )
                                for value in fold_values
                            ],
                            dtype=np.int64,
                        )
                fold_values %= self.fold_count
                width = design.shape[1]
                fold_xtx = np.zeros((self.fold_count, width, width), dtype=np.float64)
                fold_xty = np.zeros((self.fold_count, width), dtype=np.float64)
                fold_xtt = np.zeros((self.fold_count, width), dtype=np.float64)
                fold_yty = np.zeros(self.fold_count, dtype=np.float64)
                fold_ytt = np.zeros(self.fold_count, dtype=np.float64)
                fold_ttt = np.zeros(self.fold_count, dtype=np.float64)
                for fold in range(self.fold_count):
                    selected = fold_values == fold
                    fold_design = design[selected]
                    fold_outcome = outcome[selected]
                    fold_treatment = treatment[selected]
                    fold_xtx[fold] = fold_design.T @ fold_design
                    fold_xty[fold] = fold_design.T @ fold_outcome
                    fold_xtt[fold] = fold_design.T @ fold_treatment
                    fold_yty[fold] = fold_outcome @ fold_outcome
                    fold_ytt[fold] = fold_outcome @ fold_treatment
                    fold_ttt[fold] = fold_treatment @ fold_treatment
                partial.update(
                    {
                        "fold_xtx": fold_xtx,
                        "fold_xty": fold_xty,
                        "fold_xtt": fold_xtt,
                        "fold_yty": fold_yty,
                        "fold_ytt": fold_ytt,
                        "fold_ttt": fold_ttt,
                    }
                )
            else:
                width = design.shape[1]
                partial.update(
                    {
                        "fold_xtx": np.zeros((1, width, width), dtype=np.float64),
                        "fold_xty": np.zeros((1, width), dtype=np.float64),
                        "fold_xtt": np.zeros((1, width), dtype=np.float64),
                        "fold_yty": np.zeros(1, dtype=np.float64),
                        "fold_ytt": np.zeros(1, dtype=np.float64),
                        "fold_ttt": np.zeros(1, dtype=np.float64),
                    }
                )
            state = self.merge_states(state, partial)
        return state

    def merge_states(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        import numpy as np

        return {
            field.name: np.asarray(
                np.asarray(left[field.name], dtype=field.dtype)
                + np.asarray(right[field.name], dtype=field.dtype),
                dtype=field.dtype,
            )
            for field in _DML_STATE_SCHEMA
        }

    def finalize_model(self, state: Mapping[str, object]) -> CausalATEModel:
        import numpy as np

        xtx = np.asarray(state["xtx"], dtype=np.float64)
        xty = np.asarray(state["xty"], dtype=np.float64)
        xtt = np.asarray(state["xtt"], dtype=np.float64)
        inverse = np.linalg.pinv(xtx)
        if self.fold_count > 1 and "fold_xtx" in state:
            fold_xtx = np.asarray(state["fold_xtx"], dtype=np.float64)
            fold_xty = np.asarray(state["fold_xty"], dtype=np.float64)
            fold_xtt = np.asarray(state["fold_xtt"], dtype=np.float64)
            fold_yty = np.asarray(state["fold_yty"], dtype=np.float64)
            fold_ytt = np.asarray(state["fold_ytt"], dtype=np.float64)
            fold_ttt = np.asarray(state["fold_ttt"], dtype=np.float64)
            residual_yt = 0.0
            residual_tt = 0.0
            residual_yy = 0.0
            for fold in range(self.fold_count):
                if not np.any(fold_xtx[fold]):
                    raise AlgorithmInputError(
                        "Linear DML cross-fitting produced an empty fold"
                    )
                train_xtx = xtx - fold_xtx[fold]
                train_xty = xty - fold_xty[fold]
                train_xtt = xtt - fold_xtt[fold]
                nuisance_inverse = np.linalg.pinv(train_xtx)
                beta_y = nuisance_inverse @ train_xty
                beta_t = nuisance_inverse @ train_xtt
                residual_yt += float(
                    fold_ytt[fold]
                    - beta_t @ fold_xty[fold]
                    - beta_y @ fold_xtt[fold]
                    + beta_t @ fold_xtx[fold] @ beta_y
                )
                residual_tt += float(
                    fold_ttt[fold]
                    - 2 * beta_t @ fold_xtt[fold]
                    + beta_t @ fold_xtx[fold] @ beta_t
                )
                residual_yy += float(
                    fold_yty[fold]
                    - 2 * beta_y @ fold_xty[fold]
                    + beta_y @ fold_xtx[fold] @ beta_y
                )
        else:
            residual_yt = float(np.asarray(state["ytt"])) - float(xtt @ inverse @ xty)
            residual_tt = float(np.asarray(state["ttt"])) - float(xtt @ inverse @ xtt)
            residual_yy = float(np.asarray(state["yty"])) - float(xty @ inverse @ xty)
        if residual_tt <= 1e-12:
            raise AlgorithmInputError("Linear DML treatment residual has no overlap")
        effect = residual_yt / residual_tt
        row_count = int(np.asarray(state["row_count"]))
        degrees = max(1, row_count - xtx.shape[0] - 1)
        residual_variance = max(
            0.0,
            (residual_yy - 2 * effect * residual_yt + effect**2 * residual_tt)
            / degrees,
        )
        standard_error = math.sqrt(residual_variance / residual_tt)
        treated_count = int(np.asarray(state["treated_count"]))
        control_count = int(np.asarray(state["control_count"]))
        if treated_count < 2 or control_count < 2:
            raise AlgorithmInputError("Linear DML requires treatment overlap")
        treated_mean = float(np.asarray(state["treated_sum"])) / treated_count
        control_mean = float(np.asarray(state["control_sum"])) / control_count
        z = float(self.plan.algorithm_config.get("confidence_z", 1.959963984540054))
        policy_cost = float(self.plan.algorithm_config.get("policy_cost", 0.0))
        return CausalATEModel(
            method="distributed_linear_dml",
            treatment=self.treatment,
            outcome=self.outcome,
            feature_names=self.feature_names,
            treated_count=treated_count,
            control_count=control_count,
            treated_mean=treated_mean,
            control_mean=control_mean,
            effect=effect,
            standard_error=standard_error,
            confidence_interval=(
                effect - z * standard_error,
                effect + z * standard_error,
            ),
            policy_cost=policy_cost,
            treat_all_policy=effect > policy_cost,
            diagnostics={"cross_fit_folds": float(self.fold_count)},
        )

    def state_schema(self) -> tuple[StateField, ...]:
        return _DML_STATE_SCHEMA

    def empty_partition(self) -> Mapping[str, object]:
        import numpy as np

        width = len(self.confounders) + 1
        return {
            "xtx": np.zeros((width, width), dtype=np.float64),
            "xty": np.zeros(width, dtype=np.float64),
            "xtt": np.zeros(width, dtype=np.float64),
            **{
                field.name: np.asarray(0, dtype=field.dtype)
                for field in _DML_STATE_SCHEMA
                if field.name not in {"xtx", "xty", "xtt"}
            },
            "fold_xtx": np.zeros((self.fold_count, width, width), dtype=np.float64),
            "fold_xty": np.zeros((self.fold_count, width), dtype=np.float64),
            "fold_xtt": np.zeros((self.fold_count, width), dtype=np.float64),
            "fold_yty": np.zeros(self.fold_count, dtype=np.float64),
            "fold_ytt": np.zeros(self.fold_count, dtype=np.float64),
            "fold_ttt": np.zeros(self.fold_count, dtype=np.float64),
        }

    def coverage_counts(self, state: Mapping[str, object]) -> Mapping[str, int]:
        import numpy as np

        return {
            "treated": int(np.asarray(state["treated_count"])),
            "control": int(np.asarray(state["control_count"])),
        }

    @property
    def retry_safe(self) -> bool:
        return True


class LinearIVATE(
    MapReduceAlgorithm[Mapping[str, object], Mapping[str, object], CausalATEModel]
):
    """Estimate a just-identified linear IV effect from distributed moments."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        binding = plan.primary_input_binding
        self.treatment = str(plan.algorithm_config["treatment_col"])
        self.instrument = str(plan.algorithm_config["instrument_col"])
        self.outcome = binding.label_name or ""
        self.feature_names = binding.feature_names
        self.confounders = tuple(
            name
            for name in self.feature_names
            if name not in {self.treatment, self.instrument}
        )
        if (
            not self.outcome
            or self.treatment not in self.feature_names
            or self.instrument not in self.feature_names
        ):
            raise AlgorithmConfigurationError(
                "Linear IV requires treatment, instrument, and outcome"
            )

    def map_partition(
        self,
        batches: Iterable[Mapping[str, object]],
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        del context
        import numpy as np

        state = self.empty_partition()
        for batch in batches:
            required = (
                *self.confounders,
                self.treatment,
                self.instrument,
                self.outcome,
            )
            missing = [name for name in required if name not in batch]
            if missing:
                raise AlgorithmInputError(f"Linear IV batch is missing: {missing}")
            treatment = np.asarray(batch[self.treatment], dtype=np.float64)
            instrument = np.asarray(batch[self.instrument], dtype=np.float64)
            outcome = np.asarray(batch[self.outcome], dtype=np.float64)
            confounders = (
                np.column_stack(
                    [
                        np.asarray(batch[name], dtype=np.float64)
                        for name in self.confounders
                    ]
                )
                if self.confounders
                else np.empty((outcome.size, 0), dtype=np.float64)
            )
            design = np.column_stack(
                [np.ones(outcome.size, dtype=np.float64), confounders]
            )
            if (
                treatment.ndim != 1
                or instrument.ndim != 1
                or outcome.ndim != 1
                or treatment.shape != outcome.shape
                or instrument.shape != outcome.shape
                or not np.isfinite(design).all()
                or not np.isfinite(outcome).all()
                or not np.isfinite(instrument).all()
                or not np.isin(treatment, (0.0, 1.0)).all()
            ):
                raise AlgorithmInputError("Linear IV input is invalid")
            treated = treatment == 1
            control = ~treated
            partial = {
                "xtx": design.T @ design,
                "xty": design.T @ outcome,
                "xtt": design.T @ treatment,
                "xtz": design.T @ instrument,
                "yty": np.asarray(outcome @ outcome, dtype=np.float64),
                "ytt": np.asarray(outcome @ treatment, dtype=np.float64),
                "ttt": np.asarray(treatment @ treatment, dtype=np.float64),
                "ytz": np.asarray(outcome @ instrument, dtype=np.float64),
                "ttz": np.asarray(treatment @ instrument, dtype=np.float64),
                "ztz": np.asarray(instrument @ instrument, dtype=np.float64),
                "treated_count": np.asarray(treated.sum(), dtype=np.int64),
                "control_count": np.asarray(control.sum(), dtype=np.int64),
                "treated_sum": np.asarray(outcome[treated].sum(), dtype=np.float64),
                "control_sum": np.asarray(outcome[control].sum(), dtype=np.float64),
                "row_count": np.asarray(outcome.size, dtype=np.int64),
            }
            state = self.merge_states(state, partial)
        return state

    def merge_states(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        import numpy as np

        return {
            item.name: np.asarray(
                np.asarray(left[item.name], dtype=item.dtype)
                + np.asarray(right[item.name], dtype=item.dtype),
                dtype=item.dtype,
            )
            for item in _IV_STATE_SCHEMA
        }

    def finalize_model(self, state: Mapping[str, object]) -> CausalATEModel:
        import numpy as np

        xtx = np.asarray(state["xtx"], dtype=np.float64)
        xty = np.asarray(state["xty"], dtype=np.float64)
        xtt = np.asarray(state["xtt"], dtype=np.float64)
        xtz = np.asarray(state["xtz"], dtype=np.float64)
        inverse = np.linalg.pinv(xtx)
        yz = float(np.asarray(state["ytz"])) - float(xtz @ inverse @ xty)
        tz = float(np.asarray(state["ttz"])) - float(xtz @ inverse @ xtt)
        zz = float(np.asarray(state["ztz"])) - float(xtz @ inverse @ xtz)
        tt = float(np.asarray(state["ttt"])) - float(xtt @ inverse @ xtt)
        if abs(tz) <= 1e-12 or zz <= 1e-12 or tt <= 1e-12:
            raise AlgorithmInputError("Linear IV has a weak or degenerate instrument")
        effect = yz / tz
        yy = float(np.asarray(state["yty"])) - float(xty @ inverse @ xty)
        yt = float(np.asarray(state["ytt"])) - float(xtt @ inverse @ xty)
        row_count = int(np.asarray(state["row_count"]))
        degrees = max(1, row_count - xtx.shape[0] - 1)
        residual_variance = max(
            0.0,
            (yy - 2 * effect * yt + effect**2 * tt) / degrees,
        )
        standard_error = math.sqrt(residual_variance * zz / (tz**2))
        first_stage_correlation = tz / math.sqrt(zz * tt)
        treated_count = int(np.asarray(state["treated_count"]))
        control_count = int(np.asarray(state["control_count"]))
        if treated_count < 2 or control_count < 2:
            raise AlgorithmInputError("Linear IV requires treatment overlap")
        treated_mean = float(np.asarray(state["treated_sum"])) / treated_count
        control_mean = float(np.asarray(state["control_sum"])) / control_count
        z_value = float(
            self.plan.algorithm_config.get("confidence_z", 1.959963984540054)
        )
        policy_cost = float(self.plan.algorithm_config.get("policy_cost", 0.0))
        return CausalATEModel(
            method="distributed_linear_2sls",
            treatment=self.treatment,
            outcome=self.outcome,
            feature_names=self.feature_names,
            treated_count=treated_count,
            control_count=control_count,
            treated_mean=treated_mean,
            control_mean=control_mean,
            effect=effect,
            standard_error=standard_error,
            confidence_interval=(
                effect - z_value * standard_error,
                effect + z_value * standard_error,
            ),
            policy_cost=policy_cost,
            treat_all_policy=effect > policy_cost,
            diagnostics={
                "first_stage_correlation": first_stage_correlation,
                "residualized_instrument_variance": zz,
            },
        )

    def state_schema(self) -> tuple[StateField, ...]:
        return _IV_STATE_SCHEMA

    def empty_partition(self) -> Mapping[str, object]:
        import numpy as np

        width = len(self.confounders) + 1
        return {
            "xtx": np.zeros((width, width), dtype=np.float64),
            "xty": np.zeros(width, dtype=np.float64),
            "xtt": np.zeros(width, dtype=np.float64),
            "xtz": np.zeros(width, dtype=np.float64),
            **{
                item.name: np.asarray(0, dtype=item.dtype)
                for item in _IV_STATE_SCHEMA
                if item.name not in {"xtx", "xty", "xtt", "xtz"}
            },
        }

    def coverage_counts(self, state: Mapping[str, object]) -> Mapping[str, int]:
        import numpy as np

        return {
            "treated": int(np.asarray(state["treated_count"])),
            "control": int(np.asarray(state["control_count"])),
        }

    @property
    def retry_safe(self) -> bool:
        return True


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DifferenceInMeansATE | LinearDMLATE | LinearIVATE:
    del artifacts
    if implementation is DifferenceInMeansATE:
        return DifferenceInMeansATE(plan)
    if implementation is LinearDMLATE:
        return LinearDMLATE(plan)
    if implementation is LinearIVATE:
        return LinearIVATE(plan)
    raise AlgorithmConfigurationError("causal implementation identity drifted")


def export_model(
    *,
    model: CausalATEModel,
    plan: ResolvedAlgorithmPlan,
    run_id: str | None = None,
) -> AlgorithmExecutionResult:
    import numpy as np
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.linear_model import LinearRegression
    from tributo.exporting.models import BundleOutputConfig, ExportSource, ExportTarget
    from tributo.exporting.service import BundleExportService

    if not isinstance(model, CausalATEModel):
        raise AlgorithmExecutionError("causal exporter received an invalid model")
    estimator = LinearRegression().fit(
        np.zeros((2, len(model.feature_names)), dtype=np.float32),
        np.full(2, model.effect, dtype=np.float32),
    )
    converted = convert_sklearn(
        estimator,
        initial_types=[
            ("float_input", FloatTensorType([None, len(model.feature_names)]))
        ],
        target_opset=18,
    )
    payload = converted.SerializeToString()
    report = model.report()
    report_payload = json.dumps(report, sort_keys=True).encode("utf-8")
    report_artifact = ArtifactDraft.from_payload(
        name="causal-report",
        kind="report",
        format="application/json",
        payload=report_payload,
    )
    output = plan.algorithm_config.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError("causal output.bundle_uri is required")
    source = ExportSource(
        source_kind="prebuilt_onnx",
        model_object=payload,
        feature_schema={"feature_names": list(model.feature_names)},
        metadata={
            "framework": "distributed_statistics",
            "task_type": "causal_effect_estimation",
            "causal_study": report,
            "producer_distribution": _PACKAGE,
        },
        source_fingerprint=report_artifact.sha256,
    )
    bundle = BundleExportService().export_bundle(
        source,
        BundleOutputConfig(
            bundle_uri=str(output["bundle_uri"]),
            request_id=run_id,
            run_id=run_id,
            targets=[
                ExportTarget(
                    name="effect-model",
                    format="onnx",
                    exporter_id="prebuilt-onnx-v1",
                ),
                ExportTarget(
                    name="causal-report",
                    format="json",
                    exporter_id="official-causal-report-v1",
                ),
            ],
            roles={"inference": "effect-model", "report": "causal-report"},
        ),
    )
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics={
            "effect": model.effect,
            "standard_error": model.standard_error,
            "treated_count": model.treated_count,
            "control_count": model.control_count,
        },
        outputs={
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
            "report_artifact_sha256": report_artifact.sha256,
        },
        artifacts=(report_artifact,),
    )


def _contract(contract_id: str, digest: str, validator: str) -> ContractBinding:
    return ContractBinding(
        contract_id=contract_id,
        schema_version=1,
        schema_digest=digest * 64,
        validator_ref=QualifiedReference.parse(
            f"tributo_algorithms_causal_core.contracts:{validator}"
        ),
    )


_SPEC = AlgorithmSpec(
    name="difference_in_means_ate",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="causal_inference",
    model_family="difference_in_means",
    data_modalities=("tabular",),
    lifecycle_kind="identify_estimate_policy",
    allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
    config_contract_ref="tributo.official.causal.ate.config.v1",
    input_contract_ref="tributo.official.causal.binary-treatment.v1",
    output_contract_ref="tributo.official.causal.report-bundle.v1",
)

_DML_SPEC = AlgorithmSpec(
    name="linear_dml_ate",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="causal_inference",
    model_family="linear_dml",
    data_modalities=("tabular",),
    lifecycle_kind="identify_crossfit_estimate_policy",
    allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
    config_contract_ref="tributo.official.causal.linear-dml.config.v1",
    input_contract_ref="tributo.official.causal.binary-treatment.v1",
    output_contract_ref="tributo.official.causal.report-bundle.v1",
)

_IV_SPEC = AlgorithmSpec(
    name="linear_iv_ate",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE, Capability.TUNABLE),
    learning_paradigm="causal_inference",
    model_family="linear_instrumental_variables",
    data_modalities=("tabular",),
    lifecycle_kind="identify_first_stage_estimate_policy",
    allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
    config_contract_ref="tributo.official.causal.linear-iv.config.v1",
    input_contract_ref="tributo.official.causal.binary-treatment-instrument.v1",
    output_contract_ref="tributo.official.causal.report-bundle.v1",
)

ATE_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_SPEC,
    implementation_id="tributo.official.causal.difference_in_means",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_causal_core.algorithm:DifferenceInMeansATE",
    executable_factory="tributo_algorithms_causal_core.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework=None,
    environment=EnvironmentSpec(
        environment_id="tributo.official.causal-core.v1",
        dependencies=(
            "numpy>=2,<3",
            "scikit-learn>=1.4,<2",
            "skl2onnx>=1.17",
            "tributo>=1,<2",
            f"{_PACKAGE}=={_VERSION}",
        ),
    ),
    allowed_config_keys=("confidence_z", "output", "policy_cost", "treatment_col"),
    strategy=DistributionStrategy.RAY_MAP_REDUCE,
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=MapReducePolicy(
        state_schema=_STATE_SCHEMA,
        max_partial_state_bytes=4096,
        reducer_ref=(
            "tributo_algorithms_causal_core.algorithm:DifferenceInMeansATE.merge_states"
        ),
        finalizer_ref=(
            "tributo_algorithms_causal_core.algorithm:"
            "DifferenceInMeansATE.finalize_model"
        ),
        commutative=True,
        max_retries=0,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_causal_core.algorithm:export_model",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_contract(_SPEC.config_contract_ref or "", "6", "ATEConfigValidator"),
        input=_contract(_SPEC.input_contract_ref or "", "2", "CausalInputValidator"),
        output=_contract(_SPEC.output_contract_ref or "", "3", "CausalOutputValidator"),
        coverage=_contract(
            "tributo.official.causal.treatment-coverage.v1",
            "4",
            "TreatmentCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

LINEAR_DML_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_DML_SPEC,
    implementation_id="tributo.official.causal.linear_dml",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_causal_core.algorithm:LinearDMLATE",
    executable_factory="tributo_algorithms_causal_core.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework=None,
    environment=ATE_DESCRIPTOR.registration.environment,
    allowed_config_keys=(
        "confidence_z",
        "cross_fit_folds",
        "fold_column",
        "output",
        "policy_cost",
        "treatment_col",
    ),
    strategy=DistributionStrategy.RAY_MAP_REDUCE,
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=MapReducePolicy(
        state_schema=_DML_STATE_SCHEMA,
        max_partial_state_bytes=64 * 1024 * 1024,
        reducer_ref=(
            "tributo_algorithms_causal_core.algorithm:LinearDMLATE.merge_states"
        ),
        finalizer_ref=(
            "tributo_algorithms_causal_core.algorithm:LinearDMLATE.finalize_model"
        ),
        commutative=True,
        max_retries=0,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_causal_core.algorithm:export_model",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_contract(
            _DML_SPEC.config_contract_ref or "", "6", "ATEConfigValidator"
        ),
        input=_contract(
            _DML_SPEC.input_contract_ref or "", "2", "CausalInputValidator"
        ),
        output=_contract(
            _DML_SPEC.output_contract_ref or "", "3", "CausalOutputValidator"
        ),
        coverage=_contract(
            "tributo.official.causal.dml-treatment-coverage.v1",
            "4",
            "TreatmentCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

LINEAR_IV_DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=_IV_SPEC,
    implementation_id="tributo.official.causal.linear_iv",
    implementation_version="1.0.0",
    implementation="tributo_algorithms_causal_core.algorithm:LinearIVATE",
    executable_factory="tributo_algorithms_causal_core.algorithm:create_algorithm",
    distribution=_PACKAGE,
    framework=None,
    environment=ATE_DESCRIPTOR.registration.environment,
    allowed_config_keys=(
        "confidence_z",
        "instrument_col",
        "output",
        "policy_cost",
        "treatment_col",
    ),
    strategy=DistributionStrategy.RAY_MAP_REDUCE,
    supported_worker_range=WorkerRange(2, 256),
    supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=MapReducePolicy(
        state_schema=_IV_STATE_SCHEMA,
        max_partial_state_bytes=64 * 1024 * 1024,
        reducer_ref="tributo_algorithms_causal_core.algorithm:LinearIVATE.merge_states",
        finalizer_ref=(
            "tributo_algorithms_causal_core.algorithm:LinearIVATE.finalize_model"
        ),
        commutative=True,
        max_retries=0,
    ),
    package_name=_PACKAGE,
    package_version=_VERSION,
    tributo_version_spec=">=1,<2",
    exporter="tributo_algorithms_causal_core.algorithm:export_model",
    flavor_id="onnx-runtime-v1",
    contract_bindings=ContractBindingSet(
        config=_contract(_IV_SPEC.config_contract_ref or "", "5", "IVConfigValidator"),
        input=_contract(_IV_SPEC.input_contract_ref or "", "2", "CausalInputValidator"),
        output=_contract(
            _IV_SPEC.output_contract_ref or "", "3", "CausalOutputValidator"
        ),
        coverage=_contract(
            "tributo.official.causal.iv-treatment-coverage.v1",
            "4",
            "TreatmentCoverageValidator",
        ),
    ),
    descriptor_api_version=2,
    is_default=True,
)

__all__ = [
    "ATE_DESCRIPTOR",
    "LINEAR_DML_DESCRIPTOR",
    "LINEAR_IV_DESCRIPTOR",
    "CausalATEModel",
    "DifferenceInMeansATE",
    "LinearDMLATE",
    "LinearIVATE",
]
