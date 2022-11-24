from dataclasses import dataclass
from functools import partial
from glob import glob
from itertools import chain, groupby
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy
import torch
from acoustic_feature_extractor.data.phoneme import OjtPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypedDict

from accent_estimator.config import DatasetConfig, DatasetFileConfig

mora_phoneme_list = ["a", "i", "u", "e", "o", "I", "U", "E", "N", "cl", "pau"]


def f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    weight: numpy.ndarray,
):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    output = numpy.array(
        [
            numpy.sum(a[a > 0] * b[a > 0]) / numpy.sum(b[a > 0])
            for a, b in zip(numpy.split(f0, indexes), numpy.split(weight, indexes))
        ],
        dtype=f0.dtype,
    )
    return output


def make_phoneme_array(phoneme_list: List[OjtPhoneme], rate: float, length: int):
    to_index = lambda x: int(x * rate)
    phoneme = numpy.zeros(length, dtype=numpy.int32)
    for p in phoneme_list:
        phoneme[to_index(p.start) : to_index(p.end)] = p.phoneme_id
    return phoneme[:length]


def make_index_array(split_second_list: List[float], rate: float, length: int):
    to_index = lambda x: int(x * rate)
    split_second_list = numpy.r_[0, split_second_list]
    array = numpy.ones(length, dtype=numpy.int64) * (len(split_second_list) - 1)
    for i in range(len(split_second_list) - 1):
        array[to_index(split_second_list[i]) : to_index(split_second_list[i + 1])] = i
    if len(array) < length:
        array = numpy.pad(array, (0, length - len(array)), "edge")
    return array[:length]


def split_mora(phoneme_list: List[OjtPhoneme]):
    vowel_indexes = [
        i for i, p in enumerate(phoneme_list) if p.phoneme in mora_phoneme_list
    ]
    vowel_phoneme_list = [phoneme_list[i] for i in vowel_indexes]
    consonant_phoneme_list: List[Optional[OjtPhoneme]] = [None] + [
        None if post - prev == 1 else phoneme_list[post - 1]
        for prev, post in zip(vowel_indexes[:-1], vowel_indexes[1:])
    ]
    return consonant_phoneme_list, vowel_phoneme_list


@dataclass
class InputData:
    f0: SamplingData
    phoneme_list: List[OjtPhoneme]
    volume: SamplingData
    accent_start: List[bool]
    accent_end: List[bool]
    accent_phrase_start: List[bool]
    accent_phrase_end: List[bool]


@dataclass
class LazyInputData:
    f0_path: Path
    phoneme_list_path: Path
    volume_path: Path
    accent_start_path: Path
    accent_end_path: Path
    accent_phrase_start_path: Path
    accent_phrase_end_path: Path

    def generate(self):
        return InputData(
            f0=SamplingData.load(self.f0_path),
            phoneme_list=OjtPhoneme.load_julius_list(self.phoneme_list_path),
            volume=SamplingData.load(self.volume_path),
            accent_start=[
                bool(int(s)) for s in self.accent_start_path.read_text().split()
            ],
            accent_end=[bool(int(s)) for s in self.accent_end_path.read_text().split()],
            accent_phrase_start=[
                bool(int(s)) for s in self.accent_phrase_start_path.read_text().split()
            ],
            accent_phrase_end=[
                bool(int(s)) for s in self.accent_phrase_end_path.read_text().split()
            ],
        )


class OutputData(TypedDict):
    accent_start: Tensor
    accent_end: Tensor
    accent_phrase_start: Tensor
    accent_phrase_end: Tensor
    frame_f0: Tensor
    frame_phoneme: Tensor
    frame_mora_index: Tensor
    mora_f0: Tensor
    mora_vowel: Tensor
    mora_consonant: Tensor


def preprocess(
    d: InputData,
    frame_rate: float,
    frame_phoneme_mask_max_second: float,
    frame_phoneme_mask_rate: float,
    mora_phoneme_mask_max_length: int,
    mora_phoneme_mask_rate: float,
):
    mora_indexes = [
        i for i, p in enumerate(d.phoneme_list) if p.phoneme in mora_phoneme_list
    ]

    accent_start = numpy.array([d.accent_start[i] for i in mora_indexes])
    accent_end = numpy.array([d.accent_end[i] for i in mora_indexes])
    accent_phrase_start = numpy.array([d.accent_phrase_start[i] for i in mora_indexes])
    accent_phrase_end = numpy.array([d.accent_phrase_end[i] for i in mora_indexes])

    mora_split_second_list = [
        p.end for p in d.phoneme_list[:-1] if p.phoneme in mora_phoneme_list
    ]

    f0 = d.f0.array.astype(numpy.float32)
    volume = d.volume.resample(frame_rate)
    phoneme = make_phoneme_array(
        phoneme_list=d.phoneme_list, rate=frame_rate, length=len(f0)
    )
    mora_index = make_index_array(
        split_second_list=mora_split_second_list, rate=frame_rate, length=len(f0)
    )

    min_length = min(len(f0), len(volume), len(phoneme), len(mora_index))
    f0 = f0[:min_length]
    volume = volume[:min_length]
    phoneme = phoneme[:min_length]
    mora_index = mora_index[:min_length]

    mora_f0 = f0_mean(
        f0=f0,
        rate=frame_rate,
        split_second_list=mora_split_second_list,
        weight=volume,
    )
    mora_f0[numpy.isnan(mora_f0)] = 0

    consonant_phoneme_list, vowel_phoneme_list = split_mora(d.phoneme_list)
    mora_vowel = numpy.array([p.phoneme_id for p in vowel_phoneme_list])
    mora_consonant = numpy.array(
        [p.phoneme_id if p is not None else -1 for p in consonant_phoneme_list]
    )

    if frame_phoneme_mask_max_second > 0 and frame_phoneme_mask_rate > 0:
        mask_num = int(round(frame_phoneme_mask_rate * len(phoneme) / frame_rate))
        mask_max_length = min(
            int(frame_rate * frame_phoneme_mask_max_second), len(phoneme)
        )
        for _ in range(mask_num):
            mask_length = numpy.random.randint(mask_max_length)
            mask_offset = numpy.random.randint(len(phoneme) - mask_length + 1)
            phoneme[mask_offset : mask_offset + mask_length] = -1

    if mora_phoneme_mask_max_length > 0 and mora_phoneme_mask_rate > 0:
        mask_num = int(round(mora_phoneme_mask_rate * len(mora_vowel)))
        mask_max_length = min(mora_phoneme_mask_max_length, len(mora_vowel))
        for _ in range(mask_num):
            mask_length = numpy.random.randint(mask_max_length)
            mask_offset = numpy.random.randint(len(mora_vowel) - mask_length + 1)
            mora_vowel[mask_offset : mask_offset + mask_length] = -1
            mora_consonant[mask_offset : mask_offset + mask_length] = -1

    output_data = OutputData(
        accent_start=torch.from_numpy(accent_start),
        accent_end=torch.from_numpy(accent_end),
        accent_phrase_start=torch.from_numpy(accent_phrase_start),
        accent_phrase_end=torch.from_numpy(accent_phrase_end),
        frame_f0=torch.from_numpy(f0).reshape(-1, 1),
        frame_phoneme=torch.from_numpy(phoneme),
        frame_mora_index=torch.from_numpy(mora_index),
        mora_f0=torch.from_numpy(mora_f0).reshape(-1, 1),
        mora_vowel=torch.from_numpy(mora_vowel),
        mora_consonant=torch.from_numpy(mora_consonant),
    )
    return output_data


class FeatureTargetDataset(Dataset):
    def __init__(
        self,
        datas: Sequence[Union[InputData, LazyInputData]],
        frame_rate: float,
        frame_phoneme_mask_max_second: float,
        frame_phoneme_mask_rate: float,
        mora_phoneme_mask_max_length: int,
        mora_phoneme_mask_rate: float,
    ):
        self.datas = datas
        self.preprocessor = partial(
            preprocess,
            frame_rate=frame_rate,
            frame_phoneme_mask_max_second=frame_phoneme_mask_max_second,
            frame_phoneme_mask_rate=frame_phoneme_mask_rate,
            mora_phoneme_mask_max_length=mora_phoneme_mask_max_length,
            mora_phoneme_mask_rate=mora_phoneme_mask_rate,
        )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        if isinstance(data, LazyInputData):
            data = data.generate()
        return self.preprocessor(data)


def get_datas(config: DatasetFileConfig):
    f0_paths = [Path(p) for p in sorted(glob(str(config.f0_glob)))]
    assert len(f0_paths) > 0, f"f0 files not ehough: {config.f0_glob}"

    phoneme_list_paths = [Path(p) for p in sorted(glob(str(config.phoneme_list_glob)))]
    assert len(phoneme_list_paths) == len(
        f0_paths
    ), f"phoneme list files not ehough: {config.phoneme_list_glob}"

    volume_paths = [Path(p) for p in sorted(glob(str(config.volume_glob)))]
    assert len(volume_paths) == len(
        f0_paths
    ), f"volume files not ehough: {config.volume_glob}"

    accent_start_paths = [Path(p) for p in sorted(glob(str(config.accent_start_glob)))]
    assert len(accent_start_paths) == len(
        f0_paths
    ), f"accent start files not ehough: {config.accent_start_glob}"

    accent_end_paths = [Path(p) for p in sorted(glob(str(config.accent_end_glob)))]
    assert len(accent_end_paths) == len(
        f0_paths
    ), f"accent end files not ehough: {config.accent_end_glob}"

    accent_phrase_start_paths = [
        Path(p) for p in sorted(glob(str(config.accent_phrase_start_glob)))
    ]
    assert len(accent_phrase_start_paths) == len(
        f0_paths
    ), f"accent phrase start files not ehough: {config.accent_phrase_start_glob}"

    accent_phrase_end_paths = [
        Path(p) for p in sorted(glob(str(config.accent_phrase_end_glob)))
    ]
    assert len(accent_phrase_end_paths) == len(
        f0_paths
    ), f"accent phrase end files not ehough: {config.accent_phrase_end_glob}"

    datas = [
        LazyInputData(
            f0_path=f0_path,
            phoneme_list_path=phoneme_list_path,
            volume_path=volume_path,
            accent_start_path=accent_start_path,
            accent_end_path=accent_end_path,
            accent_phrase_start_path=accent_phrase_start_path,
            accent_phrase_end_path=accent_phrase_end_path,
        )
        for (
            f0_path,
            phoneme_list_path,
            volume_path,
            accent_start_path,
            accent_end_path,
            accent_phrase_start_path,
            accent_phrase_end_path,
        ) in zip(
            f0_paths,
            phoneme_list_paths,
            volume_paths,
            accent_start_paths,
            accent_end_paths,
            accent_phrase_start_paths,
            accent_phrase_end_paths,
        )
    ]

    # 同じ音素列のものをまとめる
    # ファイル名が`{話者}_{コーパス種}_{index}`の形式であることを前提としている
    def keyfunc(d: LazyInputData):
        return "_".join(d.f0_path.stem.split("_")[-2:])

    datas = sorted(datas, key=keyfunc)
    grouped_datas = {k: list(g) for k, g in groupby(datas, key=keyfunc)}
    return grouped_datas


def create_dataset(config: DatasetConfig):
    grouped_datas = get_datas(config.train_file)
    keys = sorted(grouped_datas.keys())
    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(keys)

    tests_keys, trains_keys = keys[: config.test_num], keys[config.test_num :]
    trains = list(chain.from_iterable(grouped_datas[k] for k in trains_keys))
    tests = list(chain.from_iterable(grouped_datas[k] for k in tests_keys))

    grouped_valids = get_datas(config.valid_file)
    valids = list(chain.from_iterable(grouped_valids.values()))

    def dataset_wrapper(datas, is_test: bool):
        dataset = FeatureTargetDataset(
            datas=datas,
            frame_rate=config.frame_rate,
            frame_phoneme_mask_max_second=(
                config.frame_phoneme_mask_max_second if not is_test else 0
            ),
            frame_phoneme_mask_rate=(
                config.frame_phoneme_mask_rate if not is_test else 0
            ),
            mora_phoneme_mask_max_length=(
                config.mora_phoneme_mask_max_length if not is_test else 0
            ),
            mora_phoneme_mask_rate=(
                config.mora_phoneme_mask_rate if not is_test else 0
            ),
        )
        return dataset

    return {
        "train": dataset_wrapper(trains, is_test=False),
        "test": dataset_wrapper(tests, is_test=True),
        "eval": dataset_wrapper(tests, is_test=True),
        "valid": dataset_wrapper(valids, is_test=True),
    }
