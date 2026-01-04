"""
学習システムのシンプルなテスト。

学習は重いので１回だけ行えば良いように設計されている。
"""

import shutil
from pathlib import Path

import pytest
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.dataset import DatasetType, create_dataset
from hiho_pytorch_base.model import Model
from hiho_pytorch_base.network.predictor import create_predictor
from scripts.generate import generate
from scripts.train import train


def pytest_collection_modifyitems(items: list[pytest.Item]):
    """end-to-endテストの実行順序を制御し、生成テストの前に学習テストを配置する。"""

    def get_sort_key(item: pytest.Item) -> int:
        match item.name:
            case name if "e2e_train" in name:
                return 1
            case name if "e2e_generate" in name:
                return 2
            case _:
                return 0

    items.sort(key=get_sort_key)


def test_dataset_creation(train_config: Config) -> None:
    """データセットの作成テスト"""
    datasets = create_dataset(train_config.dataset)

    assert datasets.train is not None
    assert datasets.test is not None
    assert datasets.eval is not None
    assert datasets.valid is not None


def test_model_creation(train_config: Config) -> None:
    """モデルの作成テスト"""
    predictor = create_predictor(train_config.network)
    model = Model(model_config=train_config.model, predictor=predictor)

    assert model is not None
    assert hasattr(model, "forward")


def test_e2e_train(train_config: Config, train_output_dir: UPath) -> None:
    """学習のe2eテスト"""
    output_dir = train_output_dir / "trained_model"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    config_path = train_output_dir / "config.yaml"
    train_config.save(config_path)

    train(config_path, output_dir)

    assert output_dir.exists()
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "snapshot.pth").exists()

    predictor_files = list(output_dir.glob("predictor_*.pth"))
    assert len(predictor_files) > 0


def test_e2e_generate(train_output_dir: UPath, tmp_path: Path) -> None:
    """生成のe2eテスト"""
    trained_model_dir = train_output_dir / "trained_model"
    if not trained_model_dir.exists():
        pytest.fail("train test not completed yet")

    generate_output_dir = tmp_path / "generate_output"

    generate(
        model_dir=trained_model_dir,
        predictor_iteration=None,
        config_path=None,
        predictor_path=None,
        dataset_type=DatasetType.EVAL,
        output_dir=generate_output_dir,
        use_gpu=False,
        num_files=None,
    )

    assert generate_output_dir.exists()
    assert (generate_output_dir / "arguments.yaml").exists()
