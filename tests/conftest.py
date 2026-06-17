"""pytestの共通設定と自動テストデータ生成"""

import os
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.network.ssl_feature_models import HubertConfig, HubertModel
from tests.test_utils import setup_data_and_config


def _create_tiny_hubert_config() -> HubertConfig:
    """テスト用の最小HubertConfigを作成する。"""
    return HubertConfig(
        conv_dim=(4, 4),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        num_feat_extract_layers=2,
        feat_extract_activation="gelu",
        feat_extract_norm="group",
        feat_proj_dropout=0.0,
        feat_proj_layer_norm=True,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        layerdrop=0.0,
        layer_norm_eps=1e-5,
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """テスト環境のセットアップ"""
    os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture(scope="session")
def input_data_dir() -> Path:
    """入力データディレクトリのパス"""
    return Path(__file__).parent / "input_data"


def _create_initialized_tiny_hubert_model() -> HubertModel:
    """パラメータを適切に初期化したテスト用HubertModelを作成する。"""
    config = _create_tiny_hubert_config()
    model = HubertModel(config)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.kaiming_uniform_(param)
        else:
            torch.nn.init.uniform_(param, -0.1, 0.1)
    model.eval()
    return model


@pytest.fixture(scope="session", autouse=True)
def patch_ssl_model_load() -> Generator[None, None, None]:
    """SSLモデルのロードをテスト用の小型モデルに差し替える"""
    mp = pytest.MonkeyPatch()

    def _dummy_load_ssl_model(
        model_type: str, model_path: object, device: torch.device
    ) -> HubertModel:
        model = _create_initialized_tiny_hubert_model()
        model.to(device)
        return model

    mp.setattr(
        "hiho_pytorch_base.network.ssl_feature_models.load_ssl_model",
        _dummy_load_ssl_model,
    )
    mp.setattr(
        "hiho_pytorch_base.network.predictor.load_ssl_model",
        _dummy_load_ssl_model,
    )
    yield
    mp.undo()


@pytest.fixture(scope="session")
def output_data_dir() -> UPath:
    """出力データディレクトリのパス"""
    return UPath(__file__).parent / "output_data"


@pytest.fixture(scope="session")
def base_config_path(input_data_dir: Path) -> Path:
    """ベース設定ファイルのパス"""
    return input_data_dir / "base_config.yaml"


@pytest.fixture(scope="session", autouse=True)
def data_and_config(
    base_config_path: Path, output_data_dir: UPath, patch_ssl_model_load: None
) -> Config:
    """データディレクトリと学習テスト用の設定のセットアップ"""
    data_dir = output_data_dir / "train_data"
    return setup_data_and_config(base_config_path, data_dir)


@pytest.fixture(scope="session")
def train_config(data_and_config: Config) -> Config:
    """学習テスト用設定"""
    return data_and_config


@pytest.fixture(scope="session")
def train_output_dir(output_data_dir: UPath) -> UPath:
    """学習結果ディレクトリのパス"""
    output_dir = output_data_dir / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
