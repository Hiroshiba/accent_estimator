"""テストの便利モジュール"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data.sampling_data import SamplingData
from hiho_pytorch_base.utility.upath_utility import ensure_path


def setup_data_and_config(base_config_path: Path, data_dir: UPath) -> Config:
    """テストデータをセットアップし、設定を作る"""
    config = Config.load(UPath(base_config_path))
    assert config.dataset.valid is not None

    config.dataset.train.root_dir = data_dir
    config.dataset.valid.root_dir = data_dir
    config.dataset.hdf5_cache_dir = data_dir / "hdf5_cache"

    root_dir = config.dataset.train.root_dir
    train_num, valid_num = 30, 10
    all_stems = list(map(str, range(train_num + valid_num)))

    def _setup_data(
        generator_func: Callable[[Path], None], data_type: str, extension: str
    ) -> None:
        train_pathlist_path = data_dir / f"train_{data_type}_pathlist.txt"
        valid_pathlist_path = data_dir / f"valid_{data_type}_pathlist.txt"

        setattr(config.dataset.train, f"{data_type}_pathlist_path", train_pathlist_path)
        setattr(config.dataset.valid, f"{data_type}_pathlist_path", valid_pathlist_path)

        data_dir_path = root_dir / data_type
        data_dir_path.mkdir(parents=True, exist_ok=True)

        all_relative_paths = [f"{data_type}/{stem}.{extension}" for stem in all_stems]
        for relative_path in all_relative_paths:
            file_path = root_dir / relative_path
            if not file_path.exists():
                generator_func(ensure_path(file_path))

        if not train_pathlist_path.exists():
            train_pathlist_path.write_text("\n".join(all_relative_paths[:train_num]))
        if not valid_pathlist_path.exists():
            valid_pathlist_path.write_text("\n".join(all_relative_paths[train_num:]))

    # 可変長データの長さを事前に決定
    variable_lengths = {}
    for stem in all_stems:
        variable_lengths[stem] = int(np.random.default_rng().integers(1, 30))

    # 固定長特徴ベクトル
    def generate_feature_vector(file_path: Path) -> None:
        feature_vector = (
            np.random.default_rng()
            .normal(size=config.network.feature_vector_size)
            .astype(np.float32)
        )
        np.save(file_path, feature_vector)

    _setup_data(generate_feature_vector, "feature_vector", "npy")

    # 可変長特徴データ
    def generate_feature_variable(file_path: Path) -> None:
        stem = file_path.stem
        variable_length = variable_lengths[stem]
        feature_variable = (
            np.random.default_rng()
            .normal(size=(variable_length, config.network.feature_variable_size))
            .astype(np.float32)
        )
        np.save(file_path, feature_variable)

    _setup_data(generate_feature_variable, "feature_variable", "npy")

    # サンプリングデータ
    def generate_target_vector(file_path: Path) -> None:
        array_length = config.dataset.frame_length
        array = np.random.default_rng().integers(
            0, config.network.target_vector_size, size=array_length, dtype=np.int64
        )
        sampling_data = SamplingData(array=array, rate=config.dataset.frame_rate)
        sampling_data.save(file_path)

    _setup_data(generate_target_vector, "target_vector", "npy")

    # 可変長回帰ターゲット
    def generate_target_variable(file_path: Path) -> None:
        stem = file_path.stem
        variable_length = variable_lengths[stem]
        array = (
            np.random.default_rng()
            .normal(size=(variable_length, config.network.target_vector_size))
            .astype(np.float32)
        )
        sampling_data = SamplingData(array=array, rate=1.0)
        sampling_data.save(file_path)

    _setup_data(generate_target_variable, "target_variable", "npy")

    # 回帰ターゲット
    def generate_target_scalar(file_path: Path) -> None:
        target_class = np.random.default_rng().integers(
            0, config.network.target_vector_size, dtype=np.int64
        )
        target_scalar = float(target_class) + np.random.default_rng().normal() * 0.1
        np.save(file_path, target_scalar)

    _setup_data(generate_target_scalar, "target_scalar", "npy")

    # 話者マッピング
    speaker_names = ["A", "B", "C"]
    speaker_dict = {name: [] for name in speaker_names}
    for stem in all_stems:
        speaker_name = speaker_names[int(stem) % len(speaker_names)]
        speaker_dict[speaker_name].append(stem)

    speaker_dict_path = data_dir / "speaker_dict.json"
    speaker_dict_path.write_text(json.dumps(speaker_dict))
    config.dataset.train.speaker_dict_path = speaker_dict_path
    config.dataset.valid.speaker_dict_path = speaker_dict_path

    return config
