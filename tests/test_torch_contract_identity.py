"""Contract identity snapshots for the PyTorch migration."""

from __future__ import annotations

from typing import Any

import pytest
from tributo_algorithms_graph_pyg import GRAPHSAGE_DESCRIPTOR, RGCN_DESCRIPTOR
from tributo_algorithms_graph_pyg.contracts import (
    GraphConfigValidator,
    GraphOutputValidator,
    RelationalGraphConfigValidator,
)
from tributo_algorithms_multistage_torch import (
    DISTILLATION_DESCRIPTOR,
    PRETRAIN_FINETUNE_DESCRIPTOR,
)
from tributo_algorithms_multistage_torch.contracts import (
    DistillationConfigValidator,
    DistillationOutputValidator,
    PretrainFinetuneConfigValidator,
    PretrainFinetuneOutputValidator,
)
from tributo_algorithms_recsys_torch import JAGGED_DESCRIPTOR, TWO_TOWER_DESCRIPTOR
from tributo_algorithms_recsys_torch.contracts import (
    JaggedConfigValidator,
    JaggedOutputValidator,
    TwoTowerConfigValidator,
    TwoTowerOutputValidator,
    TwoTowerTensorInputValidator,
)
from tributo_algorithms_representation import TABULAR_AUTOENCODER_DESCRIPTOR
from tributo_algorithms_representation.contracts import (
    AutoencoderTensorInputValidator,
    RepresentationConfigValidator,
    RepresentationOutputValidator,
)
from tributo_algorithms_tabular_torch import DNN_DESCRIPTOR, PU_DESCRIPTOR
from tributo_algorithms_tabular_torch.contracts import (
    DNNTensorInputValidator,
    DNNTorchBundleOutputValidator,
    PUTensorInputValidator,
    PUTorchBundleOutputValidator,
    TabularTorchConfigValidator,
)
from tributo_algorithms_timeseries import TEMPORAL_CONV_DESCRIPTOR
from tributo_algorithms_timeseries.contracts import (
    TemporalConvTensorInputValidator,
    TimeSeriesConfigValidator,
    TimeSeriesOutputValidator,
)
from tributo_algorithms_timeseries.rnn_contracts import (
    GRUTensorInputValidator,
    LSTMTensorInputValidator,
    RNNConfigValidator,
    RNNOutputValidator,
)
from tributo_algorithms_timeseries.rnn_descriptor import GRU_DESCRIPTOR, LSTM_DESCRIPTOR
from tributo_algorithms_transformers_nlp import TOKEN_TRANSFORMER_DESCRIPTOR
from tributo_algorithms_transformers_nlp.contracts import (
    TokenTensorInputValidator,
    TransformerConfigValidator,
    TransformerOutputValidator,
)


@pytest.mark.parametrize(
    ("validator", "features", "label"),
    (
        (DNNTensorInputValidator, ["x0", "x1"], "label"),
        (PUTensorInputValidator, ["x0", "x1"], "label"),
        (TemporalConvTensorInputValidator, ["lag_1", "lag_0"], "label"),
        (LSTMTensorInputValidator, ["lag_1", "lag_0"], "label"),
        (GRUTensorInputValidator, ["lag_1", "lag_0"], "label"),
        (AutoencoderTensorInputValidator, ["x0", "x1"], None),
        (TokenTensorInputValidator, ["token_0", "token_1"], "label"),
        (TwoTowerTensorInputValidator, ["user_id", "item_id"], "label"),
    ),
)
def test_recipe_input_contracts_accept_matching_optional_evaluation_roles(
    validator: type[Any], features: list[str], label: str | None
) -> None:
    bindings = [
        {
            "name": role,
            "feature_names": list(features),
            "label_name": label,
            "sample_weight_name": None,
        }
        for role in ("train", "val", "test")
    ]
    value = {"primary_role": "train", "bindings": bindings, "descriptors": {}}
    assert validator().validate(value) == value

    bindings[1] = {**bindings[1], "feature_names": [*features, "drift"]}
    with pytest.raises(ValueError):
        validator().validate(value)


def test_all_torch_output_contracts_reject_incomplete_results() -> None:
    validators = (
        DNNTorchBundleOutputValidator,
        PUTorchBundleOutputValidator,
        TimeSeriesOutputValidator,
        RNNOutputValidator,
        RepresentationOutputValidator,
        TransformerOutputValidator,
        TwoTowerOutputValidator,
        GraphOutputValidator,
        JaggedOutputValidator,
        DistillationOutputValidator,
        PretrainFinetuneOutputValidator,
    )
    for validator in validators:
        for value in (
            {"status": "failed", "outputs": {}},
            {"status": "succeeded", "outputs": {}},
            {"status": "succeeded", "outputs": {"bundle_uri": 123}},
        ):
            with pytest.raises(ValueError):
                validator().validate(value)
        if validator is PretrainFinetuneOutputValidator:
            with pytest.raises(ValueError):
                validator().validate(
                    {
                        "status": "succeeded",
                        "outputs": {
                            "bundle_uri": "/tmp/model",
                            "composition_digest": "not-a-sha256",
                        },
                    }
                )


def test_migrated_config_contracts_require_string_bundle_uri() -> None:
    cases = (
        (
            TwoTowerConfigValidator,
            {"model": {"user_count": 1, "item_count": 1}, "output": {"bundle_uri": 1}},
        ),
        (
            JaggedConfigValidator,
            {
                "data": {
                    "user_col": "user",
                    "history_col": "history",
                    "candidate_col": "item",
                    "label_col": "label",
                    "inference_history_width": 1,
                },
                "model": {"user_count": 1, "item_count": 1, "embedding_dim": 1},
                "output": {"bundle_uri": True},
                "ray": {"storage_path": "/tmp/ray"},
                "training": {},
            },
        ),
        (
            DistillationConfigValidator,
            {
                "model": {
                    "input_features": 1,
                    "teacher_hidden": 1,
                    "student_hidden": 1,
                },
                "output": {"bundle_uri": ["invalid"]},
                "ray": {"storage_path": "/tmp/ray"},
            },
        ),
        (
            PretrainFinetuneConfigValidator,
            {
                "model": {"input_features": 1, "hidden_features": 1},
                "output": {"bundle_uri": 123},
                "ray": {"storage_path": "/tmp/ray"},
                "training": {},
            },
        ),
    )
    for validator, config in cases:
        with pytest.raises(ValueError, match="bundle_uri"):
            validator().validate(config)


def test_existing_v1_config_and_output_digests_remain_stable() -> None:
    assert TabularTorchConfigValidator.schema_digest == "5" * 64
    assert TimeSeriesConfigValidator.schema_digest == "5" * 64
    assert TimeSeriesOutputValidator.schema_digest == "7" * 64
    assert RNNConfigValidator.schema_digest == "9" * 64
    assert RNNOutputValidator.schema_digest == "b" * 64
    assert RepresentationConfigValidator.schema_digest == "9" * 64
    assert RepresentationOutputValidator.schema_digest == "b" * 64
    assert TransformerConfigValidator.schema_digest == "d" * 64
    assert TransformerOutputValidator.schema_digest == "f" * 64
    assert TwoTowerConfigValidator.schema_digest == "b" * 64
    assert GraphConfigValidator.schema_digest == "1" * 64
    assert RelationalGraphConfigValidator.schema_digest == "5" * 64
    assert GraphOutputValidator.schema_digest == "3" * 64
    assert JaggedConfigValidator.schema_digest == "9" * 64
    assert JaggedOutputValidator.schema_digest == "7" * 64
    assert DistillationConfigValidator.schema_digest == "a" * 64
    assert DistillationOutputValidator.schema_digest == "c" * 64
    assert PretrainFinetuneConfigValidator.schema_digest == "e" * 64
    assert PretrainFinetuneOutputValidator.schema_digest == "0" * 64


def test_all_torch_descriptors_freeze_contract_identity() -> None:
    descriptors = {
        "dnn": DNN_DESCRIPTOR,
        "pu": PU_DESCRIPTOR,
        "temporal_conv_classifier": TEMPORAL_CONV_DESCRIPTOR,
        "lstm_classifier": LSTM_DESCRIPTOR,
        "gru_classifier": GRU_DESCRIPTOR,
        "tabular_autoencoder": TABULAR_AUTOENCODER_DESCRIPTOR,
        "token_transformer_classifier": TOKEN_TRANSFORMER_DESCRIPTOR,
        "two_tower_recommender": TWO_TOWER_DESCRIPTOR,
        "graphsage_node_classifier": GRAPHSAGE_DESCRIPTOR,
        "rgcn_node_classifier": RGCN_DESCRIPTOR,
        "jagged_embedding_recommender": JAGGED_DESCRIPTOR,
        "teacher_student_distillation": DISTILLATION_DESCRIPTOR,
        "pretrain_finetune_classifier": PRETRAIN_FINETUNE_DESCRIPTOR,
    }
    expected = {
        "dnn": (
            "tributo.official.tabular_torch.dnn",
            "tributo.algorithm-config.dnn.v1|tributo.official.dnn.named-tensor.v2|tributo.official.dnn.onnx-bundle.v2|tributo.official.dnn.torch-coverage.v2",
        ),
        "pu": (
            "tributo.official.tabular_torch.pu",
            "tributo.algorithm-config.pu.v1|tributo.official.pu.named-tensor.v2|tributo.official.pu.onnx-bundle.v2|tributo.official.pu.torch-coverage.v2",
        ),
        "temporal_conv_classifier": (
            "tributo.official.timeseries.temporal_conv",
            "tributo.official.timeseries.tcn.config.v1|tributo.official.timeseries.tcn.window-tensor.v2|tributo.official.timeseries.tcn.onnx.v1|tributo.official.timeseries.tcn.torch-coverage.v2",
        ),
        "lstm_classifier": (
            "tributo.official.timeseries.lstm",
            "tributo.official.timeseries.lstm_classifier.config.v1|tributo.official.timeseries.lstm.window-tensor.v2|tributo.official.timeseries.lstm.onnx.v1|tributo.official.timeseries.lstm_classifier.torch-coverage.v2",
        ),
        "gru_classifier": (
            "tributo.official.timeseries.gru",
            "tributo.official.timeseries.gru_classifier.config.v1|tributo.official.timeseries.gru.window-tensor.v2|tributo.official.timeseries.gru.onnx.v1|tributo.official.timeseries.gru_classifier.torch-coverage.v2",
        ),
        "tabular_autoencoder": (
            "tributo.official.representation.tabular_autoencoder",
            "tributo.official.autoencoder.config.v1|tributo.official.autoencoder.tensor-input.v2|tributo.official.autoencoder.onnx.v1|tributo.official.autoencoder.torch-coverage.v2",
        ),
        "token_transformer_classifier": (
            "tributo.official.transformer.token_classifier",
            "tributo.official.transformer.config.v1|tributo.official.transformer.tokens.v2|tributo.official.transformer.onnx.v1|tributo.official.transformer.torch-coverage.v2",
        ),
        "two_tower_recommender": (
            "tributo.official.recsys_torch.two_tower",
            "tributo.official.two-tower.config.v1|tributo.official.two-tower.pairs.v2|tributo.official.two-tower.onnx.v1|tributo.official.two-tower.torch-coverage.v2",
        ),
        "graphsage_node_classifier": (
            "tributo.official.graph_pyg.graphsage",
            "tributo.official.graphsage.config.v1|tributo.official.graph.homogeneous-torch.v2|tributo.official.graphsage.bundle.v1|tributo.official.graphsage.torch-coverage.v2",
        ),
        "rgcn_node_classifier": (
            "tributo.official.graph_pyg.rgcn",
            "tributo.official.rgcn.config.v1|tributo.official.graph.relational-torch.v2|tributo.official.rgcn.bundle.v1|tributo.official.rgcn.torch-coverage.v2",
        ),
        "jagged_embedding_recommender": (
            "tributo.official.recsys_torch.jagged_embedding",
            "tributo.official.jagged-recsys.config.v2|tributo.official.jagged-recsys.interactions.v2|tributo.official.jagged-recsys.bundle.v1|tributo.official.jagged-recsys.torch-coverage.v2",
        ),
        "teacher_student_distillation": (
            "tributo.official.multistage_torch.distillation",
            "tributo.official.distillation.config.v1|tributo.official.distillation.dense-labeled.v2|tributo.official.distillation.student-bundle.v1|tributo.official.distillation.torch-component-coverage.v2",
        ),
        "pretrain_finetune_classifier": (
            "tributo.official.multistage_torch.pretrain_finetune",
            "tributo.official.pretrain-finetune.config.v1|tributo.official.pretrain-finetune.dense-labeled.v2|tributo.official.pretrain-finetune.bundle.v1|tributo.official.pretrain-finetune.torch-component-coverage.v2",
        ),
    }
    expected_validators = {
        "dnn": "TabularTorchConfigValidator|DNNTensorInputValidator|DNNTorchBundleOutputValidator|DNNTorchCoverageValidator",
        "pu": "PUConfigValidator|PUTensorInputValidator|PUTorchBundleOutputValidator|PUCoverageValidator",
        "temporal_conv_classifier": "TimeSeriesConfigValidator|TemporalConvTensorInputValidator|TimeSeriesOutputValidator|TemporalConvTorchCoverageValidator",
        "lstm_classifier": "RNNConfigValidator|LSTMTensorInputValidator|RNNOutputValidator|LSTMTorchCoverageValidator",
        "gru_classifier": "RNNConfigValidator|GRUTensorInputValidator|RNNOutputValidator|GRUTorchCoverageValidator",
        "tabular_autoencoder": "RepresentationConfigValidator|AutoencoderTensorInputValidator|RepresentationOutputValidator|AutoencoderTorchCoverageValidator",
        "token_transformer_classifier": "TransformerConfigValidator|TokenTensorInputValidator|TransformerOutputValidator|TransformerTorchCoverageValidator",
        "two_tower_recommender": "TwoTowerConfigValidator|TwoTowerTensorInputValidator|TwoTowerOutputValidator|TwoTowerTorchCoverageValidator",
        "graphsage_node_classifier": "GraphConfigValidator|GraphSAGETorchInputValidator|GraphOutputValidator|GraphSAGETorchCoverageValidator",
        "rgcn_node_classifier": "RelationalGraphConfigValidator|RGCNTorchInputValidator|GraphOutputValidator|RGCNTorchCoverageValidator",
        "jagged_embedding_recommender": "JaggedConfigValidator|JaggedTorchInputValidator|JaggedOutputValidator|JaggedTorchCoverageValidator",
        "teacher_student_distillation": "DistillationConfigValidator|DistillationTorchInputValidator|DistillationOutputValidator|DistillationTorchCoverageValidator",
        "pretrain_finetune_classifier": "PretrainFinetuneConfigValidator|PretrainFinetuneTorchInputValidator|PretrainFinetuneOutputValidator|PretrainFinetuneTorchCoverageValidator",
    }
    expected_digests = {
        "TabularTorchConfigValidator": "5" * 64,
        "PUConfigValidator": "6" * 64,
        "TimeSeriesConfigValidator": "5" * 64,
        "TimeSeriesOutputValidator": "7" * 64,
        "RNNConfigValidator": "9" * 64,
        "RNNOutputValidator": "b" * 64,
        "RepresentationConfigValidator": "9" * 64,
        "RepresentationOutputValidator": "b" * 64,
        "TransformerConfigValidator": "d" * 64,
        "TransformerOutputValidator": "f" * 64,
        "TwoTowerConfigValidator": "b" * 64,
        "TwoTowerOutputValidator": "d" * 64,
        "GraphConfigValidator": "1" * 64,
        "RelationalGraphConfigValidator": "5" * 64,
        "GraphOutputValidator": "3" * 64,
        "JaggedConfigValidator": "9" * 64,
        "JaggedOutputValidator": "7" * 64,
        "DistillationConfigValidator": "a" * 64,
        "DistillationOutputValidator": "c" * 64,
        "PretrainFinetuneConfigValidator": "e" * 64,
        "PretrainFinetuneOutputValidator": "0" * 64,
        "DNNTensorInputValidator": "38e53672a698d960b3e4bd221b7712190f330f90468f5e4d60c5adeb0642e80c",
        "DNNTorchBundleOutputValidator": "785f9b729a9d7b33900b4f15f4441de656fb52f7c3420afc1d6021172e01302f",
        "DNNTorchCoverageValidator": "a195d09564fff453deafa2e1911a1b3cd998d61c7d20af7cf2667b8fc753c238",
        "PUTensorInputValidator": "3dfe33eca244df5644f23d02dbc61d989fefff0a4f5e05d1d094260ab1e28aad",
        "PUTorchBundleOutputValidator": "1b6398112fe11485ee6474278e2a66381da6ba5a463fe54c9c70181f4d0fadb1",
        "PUCoverageValidator": "eea4663d7e0c3d4d0a78b6928457d7b1fc0e74bcd57b04526ee05dfdefadd2b2",
        "TemporalConvTensorInputValidator": "02bf293b7ef63e84100f53f345f437d74e7b1d60350c470c20cc70d4ed770375",
        "TemporalConvTorchCoverageValidator": "663883d1b1f6292ce6cec2429ca8f55141626148d5207a12b34b620fc6c8aa32",
        "LSTMTensorInputValidator": "15da86adaef72ec82256ab552ccc6059c24470cc5223abbf0fea1b001921ca72",
        "GRUTensorInputValidator": "a06badee550b4f5ff025bba32f0ae1441d3a7201171e1a7f953edbbe889ea541",
        "LSTMTorchCoverageValidator": "1e8050b5da739e6e5c0f2027e5822d03a0f01f76a602da5442ce563d38ef32b6",
        "GRUTorchCoverageValidator": "e941f2d6e3d9bafe9a1d2d19695959850b65e5995cd614505457801a93da3698",
        "AutoencoderTensorInputValidator": "5e30a479c9c3923c108f3d274080ba57f59f232666fc3797749d2090e13ea2af",
        "AutoencoderTorchCoverageValidator": "4f66f4c471a0c5893199a1fee410633e70968e54277b6758ab679f326d65d4f5",
        "TokenTensorInputValidator": "ac19a92d24930a10039dc519d2e86b34abe6ce271d684462c62332c425aa2cc0",
        "TransformerTorchCoverageValidator": "85ac0a0cafe4a53cefa1d3dbe44a97ca756152174c4ce51bd4d590864bc1126f",
        "TwoTowerTensorInputValidator": "dcac4f2f739f6d444915e1db5fae092e01c75a964143f38901e737845b108351",
        "TwoTowerTorchCoverageValidator": "052a6797376c220eb744fbd49f950a02119f45ae748b10412e287d4f7037eadb",
        "GraphSAGETorchInputValidator": "66027ea23bf51af4a9489f9495ebac81fc36f5a947ce5c69a4198f276203fae2",
        "RGCNTorchInputValidator": "d491b1f6ea7eda7745fd9e2339d44e42e146a61e33d458f56c9154d0eb1959c9",
        "GraphSAGETorchCoverageValidator": "6463c5601148af6f734db31dc27f6b8a4cba072aa3a7c87bce610926eebb00ba",
        "RGCNTorchCoverageValidator": "00c724db59a165b9c10bfe159e06910c84af00037f5d96f8676e86e6654eca7f",
        "JaggedTorchInputValidator": "385ec1288ab3815568006495f8e91c9674d4c16dcf3a1dc8d2d6ec8cd86061b5",
        "JaggedTorchCoverageValidator": "d95f71a86974bd83709a390281cd29e1979d923adcbbb18af8476c68230f48e5",
        "DistillationTorchInputValidator": "6e48dd0fb8fa43215998c8ce76597c3a077ff3e74299e038839a4b5f74f6c75a",
        "DistillationTorchCoverageValidator": "b728f0b899b90d8cda9028bb9a91be5f5eea13be5c43396f012ae3b810aea15c",
        "PretrainFinetuneTorchInputValidator": "c4ac81aeb88c53b3d975a81b8e4366d2c37bcbbb685ea3e63c89b8d9f6888ecf",
        "PretrainFinetuneTorchCoverageValidator": "c92b65bd0c3f8338e6148f31feb1b656ab9ed9069839ddfaa76b3e38098e1eb2",
    }
    assert set(descriptors) == set(expected)
    for name, descriptor in descriptors.items():
        registration = descriptor.registration
        contracts = registration.contract_bindings
        assert contracts is not None
        assert registration.implementation.implementation_id == expected[name][0]
        assert registration.implementation.version == "2.0.0"
        bindings = (
            contracts.config,
            contracts.input,
            contracts.output,
            contracts.coverage,
        )
        assert tuple(binding.contract_id for binding in bindings) == tuple(
            expected[name][1].split("|")
        )
        output_version = 2 if name in {"dnn", "pu"} else 1
        assert tuple(binding.schema_version for binding in bindings) == (
            1,
            2,
            output_version,
            2,
        )
        assert tuple(
            str(binding.validator_ref).rsplit(":", 1)[1] for binding in bindings
        ) == tuple(expected_validators[name].split("|"))
        for binding in bindings:
            validator_name = str(binding.validator_ref).rsplit(":", 1)[1]
            assert validator_name in expected_digests
            assert binding.schema_digest == expected_digests[validator_name]
