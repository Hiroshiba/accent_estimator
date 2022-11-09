from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Sequence, Union

import numpy
from acoustic_feature_extractor.data.phoneme import OjtPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch import Tensor, as_tensor
from torch.utils.data import Dataset
from typing_extensions import TypedDict

from accent_estimator.config import DatasetConfig, DatasetFileConfig

mora_phoneme_list = ["a", "i", "u", "e", "o", "I", "U", "E", "N", "cl", "pau"]
voiced_phoneme_list = (
    ["a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N"]
    + ["n", "m", "y", "r", "w", "g", "z", "j", "d", "b"]
    + ["ny", "my", "ry", "gy", "by"]
)
unvoiced_mora_phoneme_list = ["A", "I", "U", "E", "O", "cl", "pau"]


def f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    weight: numpy.ndarray,
):
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    output = numpy.array(
        [
            numpy.sum(a[~numpy.isnan(a)] * b[~numpy.isnan(a)])
            / numpy.sum(b[~numpy.isnan(a)])
            for a, b in zip(numpy.split(f0, indexes), numpy.split(weight, indexes))
        ]
    )
    return output


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
    phoneme_f0: Tensor
    mora_f0: Tensor


def preprocess(d: InputData):
    mora_indexes = [
        i for i, p in enumerate(d.phoneme_list) if p.phoneme in mora_phoneme_list
    ]

    accent_start = numpy.array([d.accent_start[i] for i in mora_indexes])
    accent_end = numpy.array([d.accent_end[i] for i in mora_indexes])
    accent_phrase_start = numpy.array([d.accent_phrase_start[i] for i in mora_indexes])
    accent_phrase_end = numpy.array([d.accent_phrase_end[i] for i in mora_indexes])

    rate = d.f0.rate
    f0 = d.f0.array
    f0[f0 == 0] = numpy.nan

    volume = d.volume.resample(rate)

    min_length = min(len(f0), len(volume))
    f0 = f0[:min_length]
    volume = volume[:min_length]

    phoneme_f0 = f0_mean(
        f0=f0,
        rate=rate,
        split_second_list=[p.end for p in d.phoneme_list[:-1]],
        weight=volume,
    )
    phoneme_f0[numpy.isnan(phoneme_f0)] = 0

    phoneme_length = numpy.array([p.end - p.start for p in d.phoneme_list])
    mora_f0 = numpy.array([], dtype=numpy.float32)
    for i, diff in enumerate(numpy.diff(numpy.r_[0, mora_indexes])):
        index = mora_indexes[i]
        if diff == 1 or d.phoneme_list[index - 1].phoneme not in voiced_phoneme_list:
            mora_f0 = numpy.r_[mora_f0, phoneme_f0[index]]
        else:
            a = phoneme_f0[index - 1] * phoneme_length[index - 1]
            b = phoneme_f0[index] * phoneme_length[index]
            mora_f0 = numpy.r_[
                mora_f0,
                (a + b) / (phoneme_length[index] + phoneme_length[index - 1]),
            ]

    # mora_f0[
    #     [d.phoneme_list[i].phoneme in unvoiced_mora_phoneme_list for i in mora_indexes]
    # ] = 0

    output_data = OutputData(
        accent_start=as_tensor(accent_start),
        accent_end=as_tensor(accent_end),
        accent_phrase_start=as_tensor(accent_phrase_start),
        accent_phrase_end=as_tensor(accent_phrase_end),
        phoneme_f0=as_tensor(phoneme_f0).reshape(-1, 1),
        mora_f0=as_tensor(mora_f0).reshape(-1, 1),
    )
    return output_data


class FeatureTargetDataset(Dataset):
    def __init__(
        self,
        datas: Sequence[Union[InputData, LazyInputData]],
    ):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        if isinstance(data, LazyInputData):
            data = data.generate()
        return preprocess(data)


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
    return datas


def create_dataset(config: DatasetConfig):
    datas = get_datas(config.train_file)
    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]

    valids = get_datas(config.valid_file)

    def dataset_wrapper(datas, is_eval: bool):
        dataset = FeatureTargetDataset(
            datas=datas,
        )
        return dataset

    return {
        "train": dataset_wrapper(trains, is_eval=False),
        "test": dataset_wrapper(tests, is_eval=False),
        "eval": dataset_wrapper(tests, is_eval=True),
        "valid": dataset_wrapper(valids, is_eval=True),
    }
