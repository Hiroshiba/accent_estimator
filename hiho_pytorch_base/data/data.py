"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor

from .base import mora_phoneme_list, voiced_phoneme_list
from .phoneme import BasePhoneme
from .sampling_data import SamplingData
from .statistics import DataStatistics
from .wave import Wave


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    wave: numpy.ndarray
    sampling_rate: int
    phoneme_list: list[BasePhoneme]
    f0: SamplingData
    volume: SamplingData
    accent_start: list[bool]
    accent_end: list[bool]
    accent_phrase_start: list[bool]
    accent_phrase_end: list[bool]
    speaker_id: int


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    wave: Tensor
    phoneme_index: Tensor
    phoneme_id: Tensor
    vowel_index: Tensor
    mora_f0: Tensor
    accent: Tensor
    accent_target: Tensor
    accent_noise: Tensor
    accent_input: Tensor
    t: Tensor
    speaker_id: Tensor


def _f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: list[float],
    weight: numpy.ndarray,
) -> numpy.ndarray:
    """秒境界で区切った各区間ごとに、有声フレームの重み付き平均f0を算出する"""
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    with numpy.errstate(invalid="ignore"):
        mean_f0 = numpy.array(
            [
                numpy.sum(a[a > 0] * b[a > 0]) / numpy.sum(b[a > 0])
                for a, b in zip(
                    numpy.split(f0, indexes),
                    numpy.split(weight, indexes),
                    strict=True,
                )
            ]
        )
    mean_f0[numpy.isnan(mean_f0)] = (
        0  # NOTE: 有声フレームが無い区間は 0/0=nan になるため0埋め
    )
    return mean_f0


def preprocess(
    d: InputData,
    *,
    is_eval: bool,
    sampling_rate: int,
    frame_rate: float,
    statistics: DataStatistics,
) -> OutputData:
    """データ処理"""
    rng = numpy.random.default_rng()

    resampled = Wave(d.wave, d.sampling_rate).resample(sampling_rate)
    frame_length = round(len(resampled) / sampling_rate * frame_rate)

    assert len(d.phoneme_list) == len(d.accent_start), (
        f"音素列とアクセント列の長さが一致しません: "
        f"len(phoneme_list)={len(d.phoneme_list)}, len(accent_start)={len(d.accent_start)}"
    )

    mora_indexes = [
        i for i, p in enumerate(d.phoneme_list) if p.phoneme in mora_phoneme_list
    ]
    accent_start = numpy.array([d.accent_start[i] for i in mora_indexes])
    accent_end = numpy.array([d.accent_end[i] for i in mora_indexes])
    accent_phrase_start = numpy.array([d.accent_phrase_start[i] for i in mora_indexes])
    accent_phrase_end = numpy.array([d.accent_phrase_end[i] for i in mora_indexes])

    accent = numpy.stack(
        [accent_start, accent_end, accent_phrase_start, accent_phrase_end], axis=1
    )

    onehot = numpy.zeros((len(accent), 2, 4), dtype=numpy.float64)
    onehot[:, 0, :] = ~accent.astype(bool)
    onehot[:, 1, :] = accent.astype(bool)
    accent_target = (onehot - statistics.accent_mean) / statistics.accent_std

    if is_eval:
        t = 0.0
    else:
        t = float(_sigmoid(rng.standard_normal()))
    accent_noise = rng.standard_normal(accent_target.shape)
    accent_input = accent_noise + t * (accent_target - accent_noise)

    vowel_index = numpy.array(mora_indexes)

    phoneme_split_second_list = [float(p.end) for p in d.phoneme_list]
    phoneme_index = _make_index_array(
        split_second_list=phoneme_split_second_list,
        rate=frame_rate,
        length=frame_length,
    )

    phoneme_id = numpy.array([p.phoneme_id for p in d.phoneme_list], dtype=numpy.int64)

    f0 = d.f0.resample(frame_rate)[:, 0]
    volume = d.volume.resample(frame_rate)[:, 0]
    mora_f0 = _make_mora_f0(
        f0=f0,
        volume=volume,
        phoneme_list=d.phoneme_list,
        rate=frame_rate,
    )

    return OutputData(
        wave=torch.from_numpy(numpy.asarray(resampled, dtype=numpy.float32)),
        phoneme_index=torch.from_numpy(phoneme_index).long(),
        phoneme_id=torch.from_numpy(phoneme_id).long(),
        vowel_index=torch.from_numpy(vowel_index).long(),
        mora_f0=torch.from_numpy(mora_f0).float(),
        accent=torch.from_numpy(accent).long(),
        accent_target=torch.from_numpy(accent_target).float(),
        accent_noise=torch.from_numpy(accent_noise).float(),
        accent_input=torch.from_numpy(accent_input).float(),
        t=torch.tensor(t, dtype=torch.float32),
        speaker_id=torch.tensor(d.speaker_id).long(),
    )


def _sigmoid(a: float | numpy.ndarray) -> float | numpy.ndarray:
    """シグモイド関数"""
    return 1 / (1 + numpy.exp(-a))


def _make_mora_f0(
    f0: numpy.ndarray,
    volume: numpy.ndarray,
    phoneme_list: list[BasePhoneme],
    rate: float,
) -> numpy.ndarray:
    """フレームf0をモーラ区間ごとにvolume重み付き平均し、モーラ単位のf0に変換する"""
    length = min(len(f0), len(volume))
    f0 = f0[:length].astype(numpy.float64).copy()
    weight = volume[:length].astype(numpy.float64).copy()

    for p in phoneme_list:
        if p.phoneme not in voiced_phoneme_list:
            weight[int(p.start * rate) : int(p.end * rate)] = 0

    split_second_list = [
        p.end for p in phoneme_list[:-1] if p.phoneme in mora_phoneme_list
    ]
    return _f0_mean(
        f0=f0,
        rate=rate,
        split_second_list=split_second_list,
        weight=weight,
    )


def _make_index_array(
    split_second_list: list[float], rate: float, length: int
) -> numpy.ndarray:
    """秒単位の境界列をフレームインデックス配列に変換"""
    array = numpy.ones(length, dtype=numpy.int64) * (len(split_second_list) - 1)
    boundaries = numpy.r_[0.0, split_second_list]
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i] * rate)
        end = int(boundaries[i + 1] * rate)
        array[start:end] = i
    return array[:length]
