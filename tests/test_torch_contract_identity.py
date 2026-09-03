"""Contract identity snapshots for the PyTorch migration."""

from __future__ import annotations

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
)
from tributo_algorithms_representation import TABULAR_AUTOENCODER_DESCRIPTOR
from tributo_algorithms_representation.contracts import (
    RepresentationConfigValidator,
    RepresentationOutputValidator,
)
from tributo_algorithms_tabular_torch import DNN_DESCRIPTOR, PU_DESCRIPTOR
from tributo_algorithms_tabular_torch.contracts import (
    DNNTorchBundleOutputValidator,
    PUTorchBundleOutputValidator,
    TabularTorchConfigValidator,
)
from tributo_algorithms_timeseries import TEMPORAL_CONV_DESCRIPTOR
from tributo_algorithms_timeseries.contracts import (
    TimeSeriesConfigValidator,
    TimeSeriesOutputValidator,
)
from tributo_algorithms_timeseries.rnn_contracts import (
    RNNConfigValidator,
    RNNOutputValidator,
)
from tributo_algorithms_timeseries.rnn_descriptor import GRU_DESCRIPTOR, LSTM_DESCRIPTOR
from tributo_algorithms_transformers_nlp import TOKEN_TRANSFORMER_DESCRIPTOR
from tributo_algorithms_transformers_nlp.contracts import (
    TransformerConfigValidator,
    TransformerOutputValidator,
)


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
        "DNNTensorInputValidator": "0c4d416d22292f7b51c4919ab96e4f890c9a588f6f510d31a439eab4abfe39f8",
        "DNNTorchBundleOutputValidator": "785f9b729a9d7b33900b4f15f4441de656fb52f7c3420afc1d6021172e01302f",
        "DNNTorchCoverageValidator": "a195d09564fff453deafa2e1911a1b3cd998d61c7d20af7cf2667b8fc753c238",
        "PUTensorInputValidator": "be2e3a50d5edb882feb4cf96752c814dbe2be02adf72e87b51ed8e0f1240ea8f",
        "PUTorchBundleOutputValidator": "1b6398112fe11485ee6474278e2a66381da6ba5a463fe54c9c70181f4d0fadb1",
        "PUCoverageValidator": "eea4663d7e0c3d4d0a78b6928457d7b1fc0e74bcd57b04526ee05dfdefadd2b2",
        "TemporalConvTensorInputValidator": "c4bdfda122a04f56fa4cdb1c96db07f3b7994fb7128d20fa5abc6e4a28055cf0",
        "TemporalConvTorchCoverageValidator": "663883d1b1f6292ce6cec2429ca8f55141626148d5207a12b34b620fc6c8aa32",
        "LSTMTensorInputValidator": "938daf6220b5777d569ed20be334210c5322749d237a4f210112b7fe4a23ce33",
        "GRUTensorInputValidator": "08150527d908026fba93a12c6c56e727e2df79b0fd46be6995fbd19c74f3240b",
        "LSTMTorchCoverageValidator": "1e8050b5da739e6e5c0f2027e5822d03a0f01f76a602da5442ce563d38ef32b6",
        "GRUTorchCoverageValidator": "e941f2d6e3d9bafe9a1d2d19695959850b65e5995cd614505457801a93da3698",
        "AutoencoderTensorInputValidator": "cb113f830ec6cea8269f7e3a32a322676d006495d15d601cdf2d5fe724ca7da2",
        "AutoencoderTorchCoverageValidator": "4f66f4c471a0c5893199a1fee410633e70968e54277b6758ab679f326d65d4f5",
        "TokenTensorInputValidator": "3680c5c613f54e53b2f59bb9327a3220040aaaa4eba91c82da1c783838f22845",
        "TransformerTorchCoverageValidator": "85ac0a0cafe4a53cefa1d3dbe44a97ca756152174c4ce51bd4d590864bc1126f",
        "TwoTowerTensorInputValidator": "8afdcbccc2eb8b44220f232e4e52163f8ace134629d239cb351edf3c6d3fc75c",
        "TwoTowerTorchCoverageValidator": "052a6797376c220eb744fbd49f950a02119f45ae748b10412e287d4f7037eadb",
        "GraphSAGETorchInputValidator": "66027ea23bf51af4a9489f9495ebac81fc36f5a947ce5c69a4198f276203fae2",
        "RGCNTorchInputValidator": "d491b1f6ea7eda7745fd9e2339d44e42e146a61e33d458f56c9154d0eb1959c9",
        "GraphSAGETorchCoverageValidator": "6463c5601148af6f734db31dc27f6b8a4cba072aa3a7c87bce610926eebb00ba",
        "RGCNTorchCoverageValidator": "00c724db59a165b9c10bfe159e06910c84af00037f5d96f8676e86e6654eca7f",
        "JaggedTorchInputValidator": "a152a0ca35891b39599d93a06fe9f479da8684a79461a968a6a50f935dc5ffaf",
        "JaggedTorchCoverageValidator": "d95f71a86974bd83709a390281cd29e1979d923adcbbb18af8476c68230f48e5",
        "DistillationTorchInputValidator": "46ea3e591b5776f5ac30cb0cd45be363c7372ee711abe96a1a1c5bccf696d60a",
        "DistillationTorchCoverageValidator": "b728f0b899b90d8cda9028bb9a91be5f5eea13be5c43396f012ae3b810aea15c",
        "PretrainFinetuneTorchInputValidator": "75975fad1597ccf21285b98408ddc79f9c281445d620468453bfd1b68cfc4271",
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
