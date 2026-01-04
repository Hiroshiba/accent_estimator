"""設定のテスト"""

from pathlib import Path

import yaml
from upath import UPath

from hiho_pytorch_base.config import Config


def test_from_dict(base_config_path: Path):
    """辞書から設定オブジェクトを作るテスト"""
    d = yaml.safe_load(base_config_path.read_text())
    Config.from_dict(d)


def test_to_dict(base_config_path: Path):
    """設定オブジェクトの辞書に変換するテスト"""
    d = yaml.safe_load(base_config_path.read_text())
    Config.from_dict(d).to_dict()


def test_equal_base_config_and_reconstructed(base_config_path: Path):
    """設定の往復変換で同等性が保たれるテスト"""
    d = yaml.safe_load(base_config_path.read_text())
    base = Config.from_dict(d)
    base_re = Config.from_dict(base.to_dict())
    assert base == base_re


def test_load_and_save_roundtrip(base_config_path: Path, tmp_path: Path) -> None:
    """設定のload/saveの往復変換テスト"""
    config = Config.load(UPath(base_config_path))
    output_path = UPath(tmp_path) / "config.yaml"
    config.save(output_path)
    assert Config.load(output_path) == config
