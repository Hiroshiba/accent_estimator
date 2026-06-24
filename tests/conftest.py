"""pytestの共通設定と自動テストデータ生成"""

import os
from pathlib import Path

import pytest
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.utility.upath_utility import ensure_path
from tests.test_utils import setup_data_and_config


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """テスト環境のセットアップ"""
    os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture(scope="session")
def input_data_dir() -> Path:
    """入力データディレクトリのパス"""
    return Path(__file__).parent / "input_data"


@pytest.fixture(scope="session")
def output_data_dir() -> UPath:
    """出力データディレクトリのパス"""
    return UPath(__file__).parent / "output_data"


@pytest.fixture(scope="session")
def base_config_path(input_data_dir: Path) -> Path:
    """ベース設定ファイルのパス"""
    return input_data_dir / "base_config.yaml"


@pytest.fixture(scope="session", autouse=True)
def data_and_config(base_config_path: Path, output_data_dir: UPath) -> Config:
    """データディレクトリと学習テスト用の設定のセットアップ"""
    data_dir = output_data_dir / "train_data"
    return setup_data_and_config(base_config_path, data_dir)


@pytest.fixture(scope="session")
def train_config(data_and_config: Config) -> Config:
    """学習テスト用設定"""
    return data_and_config


@pytest.fixture(scope="session")
def train_output_dir(output_data_dir: UPath) -> Path:
    """学習結果ディレクトリのパス"""
    output_dir = ensure_path(output_data_dir / "train_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
