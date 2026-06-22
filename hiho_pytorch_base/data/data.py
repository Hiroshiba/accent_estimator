"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor

from .phoneme import BasePhoneme
from .sampling_data import SamplingData
from .wave import Wave

mora_phoneme_list = (
    "a",
    "i",
    "u",
    "e",
    "o",
    "I",
    "U",
    "E",
    "N",
    "cl",
    "pau",
    "sil",
)

voiced_phoneme_list = (
    ["a", "i", "u", "e", "o", "N"]
    + ["n", "m", "y", "r", "w", "g", "z", "j", "d", "b"]
    + ["ny", "my", "ry", "gy", "by", "gw"]
)


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
    speaker_id: Tensor


def _f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: list[float],
    weight: numpy.ndarray,
) -> numpy.ndarray:
    """秒境界で区切った各区間ごとに、有声フレームの重み付き平均でf0を平滑化する"""
    indexes = numpy.floor(numpy.array(split_second_list) * rate).astype(int)
    with numpy.errstate(invalid="ignore"):
        for a, b in zip(
            numpy.split(f0, indexes), numpy.split(weight, indexes), strict=True
        ):
            a[:] = numpy.sum(a[a > 0] * b[a > 0]) / numpy.sum(b[a > 0])
    f0[numpy.isnan(f0)] = 0  # NOTE: 有声フレームが無い区間は 0/0=nan になるため0埋め
    return f0


def preprocess(
    d: InputData, *, is_eval: bool, sampling_rate: int, frame_rate: float
) -> OutputData:
    """データ処理"""
    _ = is_eval

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
        mora_indexes=mora_indexes,
        rate=frame_rate,
    )

    return OutputData(
        wave=torch.from_numpy(numpy.asarray(resampled, dtype=numpy.float32)),
        phoneme_index=torch.from_numpy(phoneme_index).long(),
        phoneme_id=torch.from_numpy(phoneme_id).long(),
        vowel_index=torch.from_numpy(vowel_index).long(),
        mora_f0=torch.from_numpy(mora_f0).float(),
        accent=torch.from_numpy(accent).long(),
        speaker_id=torch.tensor(d.speaker_id).long(),
    )


def _make_mora_f0(
    f0: numpy.ndarray,
    volume: numpy.ndarray,
    phoneme_list: list[BasePhoneme],
    mora_indexes: list[int],
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
    frame_f0 = _f0_mean(
        f0=f0,
        rate=rate,
        split_second_list=split_second_list,
        weight=weight,
    )

    mora_f0 = numpy.zeros(len(mora_indexes), dtype=numpy.float64)
    for i, phoneme_idx in enumerate(mora_indexes):
        p = phoneme_list[phoneme_idx]
        start, end = int(p.start * rate), int(p.end * rate)
        if start == end:
            raise ValueError(
                f"モーラ区間がフレーム換算で長さ0です: phoneme={p.phoneme}, start={p.start}, end={p.end}"
            )
        mora_f0[i] = frame_f0[start:end][0]
    return mora_f0


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
