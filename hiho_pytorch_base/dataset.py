"""データセットモジュール"""

import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import assert_never

import numpy
from pydantic import TypeAdapter
from torch.utils.data import Dataset as BaseDataset
from upath import UPath

from .config import DataFileConfig, DatasetConfig
from .data.data import InputData, OutputData, preprocess
from .data.sampling_data import SamplingData
from .utility.pathlist_utility import get_data_paths
from .utility.upath_utility import to_local_path


@dataclass
class LazyInputData:
    """遅延読み込み対応の入力データ構造"""

    feature_vector_path: UPath
    feature_variable_path: UPath
    target_vector_path: UPath
    target_variable_path: UPath
    target_scalar_path: UPath
    speaker_id: int

    def fetch(self) -> InputData:
        """ファイルからデータを読み込んでInputDataを生成"""
        return InputData(
            feature_vector=numpy.load(
                to_local_path(self.feature_vector_path), allow_pickle=True
            ),
            feature_variable=numpy.load(
                to_local_path(self.feature_variable_path), allow_pickle=True
            ),
            target_vector=SamplingData.load(to_local_path(self.target_vector_path)),
            target_variable=SamplingData.load(to_local_path(self.target_variable_path)),
            target_scalar=float(
                numpy.load(to_local_path(self.target_scalar_path), allow_pickle=True)
            ),
            speaker_id=self.speaker_id,
        )


def prefetch_datas(
    train_datas: list[LazyInputData],
    test_datas: list[LazyInputData],
    valid_datas: list[LazyInputData] | None,
    train_indices: list[int],
    train_batch_size: int,
    num_prefetch: int,
) -> Callable[[], None]:
    """データセットを学習順序に従って非同期で読み込み、停止関数を返す"""
    if num_prefetch <= 0:
        return lambda: None

    prefetch_order: list[LazyInputData] = []
    prefetch_order += [train_datas[i] for i in train_indices[:train_batch_size]]
    prefetch_order += test_datas
    prefetch_order += [train_datas[i] for i in train_indices[train_batch_size:]]
    if valid_datas is not None:
        prefetch_order += valid_datas

    executor = ThreadPoolExecutor(max_workers=num_prefetch)
    for data in prefetch_order:
        executor.submit(data.fetch)

    def close() -> None:
        executor.shutdown(wait=False, cancel_futures=True)

    return close


class Dataset(BaseDataset[OutputData]):
    """メインのデータセット"""

    def __init__(
        self,
        datas: list[LazyInputData],
        config: DatasetConfig,
        is_eval: bool,
    ):
        self.datas = datas
        self.config = config
        self.is_eval = is_eval

    def __len__(self):
        """データセットのサイズ"""
        return len(self.datas)

    def __getitem__(self, i: int) -> OutputData:
        """指定されたインデックスのデータを前処理して返す"""
        try:
            return preprocess(
                self.datas[i].fetch(),
                frame_rate=self.config.frame_rate,
                frame_length=self.config.frame_length,
                is_eval=self.is_eval,
            )
        except Exception as e:
            raise RuntimeError(
                f"データ処理に失敗しました: index={i} data={self.datas[i]}"
            ) from e


class DatasetType(str, Enum):
    """データセットタイプ"""

    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"
    VALID = "valid"


@dataclass
class DatasetCollection:
    """データセットコレクション"""

    train: Dataset
    """重みの更新に用いる"""

    test: Dataset
    """trainと同じドメインでモデルの過適合確認に用いる"""

    eval: Dataset | None
    """testと同じデータを評価に用いる"""

    valid: Dataset | None
    """trainやtestと異なり、評価専用に用いる"""

    def get(self, type: DatasetType) -> Dataset:
        """指定されたタイプのデータセットを返す"""
        match type:
            case DatasetType.TRAIN:
                return self.train
            case DatasetType.TEST:
                return self.test
            case DatasetType.EVAL:
                if self.eval is None:
                    raise ValueError("evalデータセットが設定されていません")
                return self.eval
            case DatasetType.VALID:
                if self.valid is None:
                    raise ValueError("validデータセットが設定されていません")
                return self.valid
            case _:
                assert_never(type)


def get_datas(config: DataFileConfig) -> list[LazyInputData]:
    """データを取得"""
    (
        fn_list,
        (
            feature_vector_pathmappings,
            feature_variable_pathmappings,
            target_vector_pathmappings,
            target_variable_pathmappings,
            target_scalar_pathmappings,
        ),
    ) = get_data_paths(
        config.root_dir,
        [
            config.feature_vector_pathlist_path,
            config.feature_variable_pathlist_path,
            config.target_vector_pathlist_path,
            config.target_variable_pathlist_path,
            config.target_scalar_pathlist_path,
        ],
    )

    fn_each_speaker = TypeAdapter(dict[str, list[str]]).validate_json(
        to_local_path(config.speaker_dict_path).read_text()
    )
    speaker_ids = {
        fn: speaker_id
        for speaker_id, fns in enumerate(fn_each_speaker.values())
        for fn in fns
    }

    datas = [
        LazyInputData(
            feature_vector_path=feature_vector_pathmappings[fn],
            feature_variable_path=feature_variable_pathmappings[fn],
            target_vector_path=target_vector_pathmappings[fn],
            target_variable_path=target_variable_pathmappings[fn],
            target_scalar_path=target_scalar_pathmappings[fn],
            speaker_id=speaker_ids[fn],
        )
        for fn in fn_list
    ]
    return datas


def create_dataset(config: DatasetConfig) -> DatasetCollection:
    """データセットを作成"""
    # TODO: accent_estimatorのようにHDF5に対応させ、docs/にドキュメントを書く
    datas = get_datas(config.train)

    if config.seed is not None:
        random.Random(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]
    if config.train_num is not None:
        trains = trains[: config.train_num]

    def _wrapper(datas: list[LazyInputData], is_eval: bool) -> Dataset:
        if is_eval:
            datas = datas * config.eval_times_num
        dataset = Dataset(datas=datas, config=config, is_eval=is_eval)
        return dataset

    return DatasetCollection(
        train=_wrapper(trains, is_eval=False),
        test=_wrapper(tests, is_eval=False),
        eval=(_wrapper(tests, is_eval=True) if config.eval_for_test else None),
        valid=(
            _wrapper(get_datas(config.valid), is_eval=True)
            if config.valid is not None
            else None
        ),
    )
