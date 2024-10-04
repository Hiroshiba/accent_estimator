from dataclasses import dataclass
from functools import partial
from itertools import chain, groupby
from typing import TypedDict

import numpy
import torch
from torch import Tensor
from torch.utils.data import Dataset

from accent_estimator.data.data import generate_position_array, vowel_to_id

from .config import DatasetConfig, DatasetFileConfig
from .utility.dataset_utility import (
    CachePath,
    HPath,
    get_stem_to_paths,
    load_numpy,
    read_text,
)

mora_phoneme_list = ["a", "i", "u", "e", "o", "I", "U", "E", "N", "cl", "pau", "sil"]


@dataclass
class DatasetInput:
    feature: numpy.ndarray
    phoneme_list: list[str]
    accent_start: list[bool]
    accent_end: list[bool]
    accent_phrase_start: list[bool]
    accent_phrase_end: list[bool]


@dataclass
class LazyDatasetInput:
    feature_path: HPath
    phoneme_list_path: HPath
    accent_start_path: HPath
    accent_end_path: HPath
    accent_phrase_start_path: HPath
    accent_phrase_end_path: HPath

    def generate(self):
        return DatasetInput(
            feature=load_numpy(self.feature_path),
            phoneme_list=read_text(self.phoneme_list_path).split(),
            accent_start=[
                bool(int(s)) for s in read_text(self.accent_start_path).split()
            ],
            accent_end=[bool(int(s)) for s in read_text(self.accent_end_path).split()],
            accent_phrase_start=[
                bool(int(s)) for s in read_text(self.accent_phrase_start_path).split()
            ],
            accent_phrase_end=[
                bool(int(s)) for s in read_text(self.accent_phrase_end_path).split()
            ],
        )


class DatasetOutput(TypedDict):
    vowel: Tensor
    mora_position: Tensor
    feature: Tensor
    frame_position: Tensor
    accent: Tensor


def preprocess(
    d: DatasetInput,
):
    feature = d.feature

    # モーラレベルにする
    assert len(d.phoneme_list) == len(
        d.accent_start
    ), f"len(d.phoneme_list)={len(d.phoneme_list)}, len(d.accent_start)={len(d.accent_start)}"

    mora_indexes = [i for i, p in enumerate(d.phoneme_list) if p in mora_phoneme_list]
    mora_indexes = mora_indexes[1:-1]  # 最初と最後のsilを除く
    accent_start = numpy.array([d.accent_start[i] for i in mora_indexes])
    accent_end = numpy.array([d.accent_end[i] for i in mora_indexes])
    accent_phrase_start = numpy.array([d.accent_phrase_start[i] for i in mora_indexes])
    accent_phrase_end = numpy.array([d.accent_phrase_end[i] for i in mora_indexes])

    # アクセント情報をまとめる
    accent = numpy.stack(
        [
            accent_start,
            accent_end,
            accent_phrase_start,
            accent_phrase_end,
        ],
        axis=1,
    )

    # 母音の音素情報
    vowel = numpy.array([vowel_to_id(d.phoneme_list[i]) for i in mora_indexes]).astype(
        numpy.int32
    )

    # 位置情報
    mora_position = generate_position_array(len(mora_indexes))
    frame_position = generate_position_array(len(feature))

    output_data = DatasetOutput(
        vowel=torch.from_numpy(vowel).long(),
        mora_position=torch.from_numpy(mora_position).float(),
        feature=torch.from_numpy(feature).float(),
        frame_position=torch.from_numpy(frame_position).float(),
        accent=torch.from_numpy(accent).long(),
    )
    return output_data


class FeatureTargetDataset(Dataset):
    def __init__(
        self,
        datas: list[DatasetInput | LazyDatasetInput],
    ):
        self.datas = datas
        self.preprocessor = partial(
            preprocess,
        )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        if isinstance(data, LazyDatasetInput):
            data = data.generate()
        return self.preprocessor(data)


def get_datas(config: DatasetFileConfig):
    (
        fn_list,
        feature_paths,
        phoneme_list_paths,
        accent_start_paths,
        accent_end_paths,
        accent_phrase_start_paths,
        accent_phrase_end_paths,
    ) = get_stem_to_paths(
        config.feature_glob,
        config.phoneme_list_glob,
        config.accent_start_glob,
        config.accent_end_glob,
        config.accent_phrase_start_glob,
        config.accent_phrase_end_glob,
    )

    datas = [
        LazyDatasetInput(
            feature_path=CachePath(feature_paths[fn]),
            phoneme_list_path=phoneme_list_paths[fn],
            accent_start_path=accent_start_paths[fn],
            accent_end_path=accent_end_paths[fn],
            accent_phrase_start_path=accent_phrase_start_paths[fn],
            accent_phrase_end_path=accent_phrase_end_paths[fn],
        )
        for fn in fn_list
    ]

    # 同じ音素列のものをまとめる
    # NOTE: ファイル名が`{コーパス種}_{index}`の形式であることを前提としている
    def keyfunc(d: tuple[str, LazyDatasetInput]):
        return "_".join(d[0].split("_")[-2:])

    fn_datas = sorted(zip(fn_list, datas), key=keyfunc)
    grouped_datas = {k: [d[1] for d in g] for k, g in groupby(fn_datas, key=keyfunc)}
    return grouped_datas


def create_dataset(config: DatasetConfig):
    grouped_datas = get_datas(config.train_file)
    keys = sorted(grouped_datas.keys())
    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(keys)
    if config.train_num is not None:
        keys = keys[: config.test_num + config.train_num]

    tests_keys, trains_keys = keys[: config.test_num], keys[config.test_num :]
    trains = list(chain.from_iterable(grouped_datas[k] for k in trains_keys))
    tests = list(chain.from_iterable(grouped_datas[k] for k in tests_keys))

    # grouped_valids = get_datas(config.valid_file)
    # valids = list(chain.from_iterable(grouped_valids.values()))

    def dataset_wrapper(datas, is_eval: bool):
        dataset = FeatureTargetDataset(
            datas=datas,
        )
        return dataset

    return {
        "train": dataset_wrapper(trains, is_eval=False),
        "test": dataset_wrapper(tests, is_eval=False),
        "eval": dataset_wrapper(tests, is_eval=True),
        # "valid": dataset_wrapper(valids, is_eval=True),
    }
