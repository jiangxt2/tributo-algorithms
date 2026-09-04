"""Tests for the typed dense TorchRecipe implementations."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
import ray
import ray.train
import ray.train.torch
import torch
from tributo.algorithms import (
    DistributionStrategy,
    TorchBatchContext,
    TorchBuildContext,
    TorchCompositeGlobalState,
    TorchGlobalLossContext,
    TorchRuntimeContext,
    TorchStageContext,
    TorchStageRunIdentity,
    TorchStepContext,
)
from tributo_algorithms_tabular_torch import (
    DNN_DESCRIPTOR,
    PU_DESCRIPTOR,
    DNNRecipe,
    PURecipe,
)
from tributo_algorithms_tabular_torch.contracts import (
    PUConfigValidator,
    PUCoverageValidator,
    TabularTorchConfigValidator,
)
from tributo_algorithms_tabular_torch.pu_reducer import (
    PUGlobalLossReducer,
    PURiskReducerPlan,
)


def _context(
    descriptor: object, config: dict[str, Any]
) -> tuple[TorchBuildContext, TorchBatchContext, TorchStepContext]:
    registration = descriptor.registration
    policy = registration.distribution_spec.policy
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        registration.spec.name,
        registration.implementation.implementation_id,
        hashlib.sha256(b"recipe").hexdigest(),
        policy.digest,
        policy.execution_plan.digest,
    )
    runtime = TorchRuntimeContext(
        algorithm_config=config,
        implementation_id=registration.implementation.implementation_id,
        world_size=1,
        policy_digest=policy.digest,
        execution_plan_digest=policy.execution_plan.digest,
        run_identity=identity,
        input_binding_digest="1" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    return (
        TorchBuildContext(runtime, stage),
        TorchBatchContext(stage, ("x0", "x1"), label_name="label"),
        TorchStepContext(stage, 0, 0),
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "x0": torch.tensor([-2.0, -1.0, 1.0, 2.0]),
        "x1": torch.tensor([-1.0, -0.5, 0.5, 1.0]),
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }


@pytest.mark.parametrize("descriptor", [DNN_DESCRIPTOR])
def test_descriptor_uses_unified_torch_runtime(descriptor: object) -> None:
    registration = descriptor.registration
    assert (
        registration.distribution_spec.strategy is DistributionStrategy.RAY_TRAIN_TORCH
    )
    assert registration.implementation.version == "2.0.0"
    assert registration.implementation.runtime_id == "tributo.ray_train_torch"
    assert registration.contract_bindings is not None
    assert all(
        binding["schema_version"] in {1, 2}
        for binding in registration.contract_bindings.to_dict().values()
        if binding
    )


def test_dnn_recipe_produces_sum_count_contribution() -> None:
    config = {"model": {"input_features": 2, "hidden_units": [4]}, "optimizer": {}}
    build, batch_context, step_context = _context(DNN_DESCRIPTOR, config)
    recipe = DNNRecipe()
    modules = recipe.build_modules(build)
    batch = recipe.adapt_batch(_batch(), batch_context)
    result = recipe.training_step(modules, batch, step_context)
    assert result.outputs["output"].shape == (4, 1)
    assert result.loss.normalizer == 4
    assert result.loss.numerator.ndim == 0
    assert result.metrics["accuracy"].normalizer == 4
    assert recipe.metric_plan(build.runtime).reducers == {
        "accuracy": "sum_count",
        "train_loss": "sum_count",
    }


@pytest.mark.parametrize("loss_type", ["nnpu", "upu"])
def test_pu_recipe_returns_composite_components(loss_type: str) -> None:
    config = {
        "model": {"input_features": 2, "hidden_units": [4]},
        "loss": {"type": loss_type, "class_prior": 0.4},
        "training": {"accumulation_steps": 1},
    }
    build, batch_context, step_context = _context(DNN_DESCRIPTOR, config)
    recipe = PURecipe()
    modules = recipe.build_modules(build)
    batch = recipe.adapt_batch(_batch(), batch_context)
    result = recipe.training_step(modules, batch, step_context)
    assert set(result.loss.differentiable_components) == {
        "positive_loss_sum",
        "positive_as_negative_sum",
        "unlabeled_negative_sum",
    }
    assert result.loss.normalizer_components == {
        "positive_count": 2.0,
        "unlabeled_count": 2.0,
    }
    assert result.metrics["observed_positive_recall"].normalizer == 2


def test_pu_reducer_preserves_global_branch_semantics() -> None:
    reducer = PUGlobalLossReducer()
    state = TorchCompositeGlobalState(
        components={
            "positive_loss_sum": 2.0,
            "positive_as_negative_sum": 8.0,
            "unlabeled_negative_sum": 1.0,
        },
        normalizers={"positive_count": 2.0, "unlabeled_count": 2.0},
    )
    result = reducer.reduce(
        {"loss": {"type": "nnpu", "class_prior": 0.4, "beta": 0.0, "gamma": 1.0}},
        state,
        TorchGlobalLossContext(1, "1" * 64, "2" * 64),
    )
    assert result.status == "accepted"
    assert result.branch == "nnpu_correction"
    assert set(result.coefficients) == set(state.components)


@pytest.mark.parametrize(
    ("loss_type", "expected_branch"),
    [("nnpu", "nnpu_normal"), ("upu", "upu")],
)
def test_pu_reducer_covers_normal_and_upu_branches(
    loss_type: str, expected_branch: str
) -> None:
    reducer = PUGlobalLossReducer()
    state = TorchCompositeGlobalState(
        components={
            "positive_loss_sum": 6.0,
            "positive_as_negative_sum": 4.0,
            "unlabeled_negative_sum": 10.0,
        },
        normalizers={"positive_count": 2.0, "unlabeled_count": 5.0},
    )
    result = reducer.reduce(
        {
            "loss": {
                "type": loss_type,
                "class_prior": 0.4,
                "beta": 0.1,
                "gamma": 0.8,
            }
        },
        state,
        TorchGlobalLossContext(1, "1" * 64, "2" * 64),
    )
    assert result.status == "accepted"
    assert result.branch == expected_branch
    assert result.metrics["train_loss"].normalizer == 1.0
    assert result.metrics["train_loss"].numerator == pytest.approx(2.4)


def test_pu_reducer_rejects_empty_positive_or_unlabeled_group() -> None:
    reducer = PUGlobalLossReducer()
    for normalizers in (
        {"positive_count": 0.0, "unlabeled_count": 2.0},
        {"positive_count": 2.0, "unlabeled_count": 0.0},
    ):
        state = TorchCompositeGlobalState(
            components={
                "positive_loss_sum": 1.0,
                "positive_as_negative_sum": 1.0,
                "unlabeled_negative_sum": 1.0,
            },
            normalizers=normalizers,
        )
        result = reducer.reduce(
            {"loss": {"type": "nnpu", "class_prior": 0.4}},
            state,
            TorchGlobalLossContext(1, "1" * 64, "2" * 64),
        )
        assert result.status == "rejected"
        assert result.failure_code == "pu.empty_group"


def test_pu_reducer_rejects_incomplete_global_components() -> None:
    reducer = PUGlobalLossReducer()
    state = TorchCompositeGlobalState(
        components={"positive_loss_sum": 1.0},
        normalizers={"positive_count": 1.0, "unlabeled_count": 1.0},
    )
    result = reducer.reduce(
        {"loss": {"type": "nnpu", "class_prior": 0.4}},
        state,
        TorchGlobalLossContext(1, "1" * 64, "2" * 64),
    )
    assert result.status == "rejected"
    assert result.failure_code == "pu.invalid_components"


def test_pu_risk_plan_is_immutable_and_validated() -> None:
    plan = PURiskReducerPlan("nnpu", 0.4, 0.1, 0.8)
    assert plan.mode == "nnpu"
    with pytest.raises(ValueError, match="class_prior"):
        PURiskReducerPlan("nnpu", 1.0)


def test_pu_contract_rejects_missing_class_prior() -> None:
    with pytest.raises(ValueError, match="class_prior"):
        PUConfigValidator().validate(
            {"loss": {"type": "nnpu"}, "output": {"bundle_uri": "/tmp/unused"}}
        )


@pytest.mark.parametrize(
    "validator",
    [TabularTorchConfigValidator(), PUConfigValidator()],
)
def test_tabular_config_rejects_empty_bundle_uri(validator: Any) -> None:
    value: dict[str, Any] = {"output": {"bundle_uri": ""}}
    if isinstance(validator, PUConfigValidator):
        value["loss"] = {"type": "nnpu", "class_prior": 0.4}
    with pytest.raises(ValueError, match="bundle_uri"):
        validator.validate(value)


def test_pu_coverage_contract_proves_group_partition() -> None:
    value = {
        "input_complete": True,
        "distributed": True,
        "workers": [
            {
                "input_rows": {
                    "train": 4,
                    "coverage.positive": 1,
                    "coverage.unlabeled": 3,
                }
            },
            {
                "input_rows": {
                    "train": 4,
                    "coverage.positive": 3,
                    "coverage.unlabeled": 1,
                }
            },
        ],
    }
    assert PUCoverageValidator().validate(value) == value


@pytest.mark.parametrize("descriptor", (DNN_DESCRIPTOR, PU_DESCRIPTOR))
def test_official_recipe_runs_through_the_core_worker_contract(
    descriptor: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tributo.integrations.algorithm_runtimes.ray_train_torch import (
        torch_recipe_train_loop_per_worker,
    )

    class Iterator:
        def iter_torch_batches(self, **kwargs: Any):
            del kwargs
            yield _batch()

    class TrainContext:
        def get_world_rank(self) -> int:
            return 0

        def get_world_size(self) -> int:
            return 1

    class RuntimeContext:
        def get_worker_id(self) -> str:
            return "worker-0"

        def get_node_id(self) -> str:
            return "node-0"

        def get_assigned_resources(self) -> dict[str, float]:
            return {"CPU": 1.0}

    reports: list[dict[str, Any]] = []
    monkeypatch.setattr(ray.train.torch, "prepare_model", lambda model: model)
    monkeypatch.setattr(ray.train, "get_context", lambda: TrainContext())
    monkeypatch.setattr(ray.train, "get_dataset_shard", lambda name: Iterator())
    monkeypatch.setattr(ray.train, "get_checkpoint", lambda: None)
    monkeypatch.setattr(
        ray.train,
        "report",
        lambda metrics, checkpoint=None: reports.append(dict(metrics)),
    )
    monkeypatch.setattr(ray, "get_runtime_context", lambda: RuntimeContext())

    registration = descriptor.registration
    policy = registration.distribution_spec.policy
    implementation_code_digest = registration.implementation.code_digest
    assert isinstance(implementation_code_digest, str)
    algorithm_config: dict[str, Any] = {
        "model": {"input_features": 2, "hidden_units": [4]},
        "optimizer": {"learning_rate": 0.01, "accumulation_steps": 1},
        "training": {"epochs": 1, "batch_size": 4},
    }
    if registration.spec.name == "pu":
        algorithm_config["loss"] = {"type": "nnpu", "class_prior": 0.4}
    identity = TorchStageRunIdentity(
        "aabbccdd",
        "11223344",
        "train",
        1,
        registration.spec.name,
        registration.implementation.implementation_id,
        implementation_code_digest,
        policy.digest,
        policy.execution_plan.digest,
    )
    runtime = TorchRuntimeContext(
        algorithm_config=algorithm_config,
        implementation_id=registration.implementation.implementation_id,
        world_size=1,
        policy_digest=policy.digest,
        execution_plan_digest=policy.execution_plan.digest,
        run_identity=identity,
        input_binding_digest="1" * 64,
    )
    stage = TorchStageContext(runtime, "train", 0, True, ("train",))
    config: dict[str, Any] = {
        **algorithm_config,
        "_core_implementation_ref": str(registration.implementation.implementation_ref),
        "_core_implementation_code_digest": implementation_code_digest,
        "_core_input_binding_digest": "1" * 64,
        "_core_state_layout": "replicated",
        "_core_checkpoint_owner_rank": 0,
        "_core_feature_names": ["x0", "x1"],
        "_core_label_name": "label",
        "_core_weight_name": None,
        "_core_policy_digest": policy.digest,
        "_core_execution_plan_digest": policy.execution_plan.digest,
        "_core_metric_reducers": {
            name: reducer.value for name, reducer in policy.metric_reducers.items()
        },
        "_core_stage_context": stage.to_dict(),
    }
    if registration.spec.name == "pu":
        config.update(
            {
                "_core_global_loss_reducer_ref": policy.global_loss_reducer_ref,
                "_core_global_loss_reducer_api_version": (
                    policy.global_loss_reducer_api_version
                ),
                "_core_global_loss_reducer_code_digest": (
                    policy.global_loss_reducer_code_digest
                ),
                "_core_composite_loss_schema_id": policy.composite_loss_schema_id,
            }
        )

    torch_recipe_train_loop_per_worker(config)

    assert len(reports) == 1
    assert reports[0]["execution_workers"][0]["input_rows"]["train"] == 4
    assert reports[0]["train_loss"] >= 0
