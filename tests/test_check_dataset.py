"""check_dataset.pyのテスト"""

from pathlib import Path

import pytest
from upath import UPath

from hiho_pytorch_base.config import Config
from scripts.check_dataset import check_dataset


def test_check_dataset_basic(train_config: Config, tmp_path: Path) -> None:
    """基本的なcheck_dataset実行テスト"""
    config_path = tmp_path / "test_config.yaml"
    train_config.save(UPath(config_path))

    check_dataset(UPath(config_path), trials=1, break_on_error=False)


def test_check_dataset_with_missing_data_files(
    train_config: Config, tmp_path: Path
) -> None:
    """存在しないデータファイルパスでのエラーテスト"""
    config_path = tmp_path / "missing_data_config.yaml"

    config_dict = train_config.to_dict()
    config_dict["dataset"]["train"]["wave_pathlist_path"] = "non_existent_pathlist.txt"

    Config.from_dict(config_dict).save(UPath(config_path))

    with pytest.raises(FileNotFoundError):
        check_dataset(UPath(config_path), trials=1, break_on_error=False)
