"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor

from .phoneme import BasePhoneme
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


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    wave: numpy.ndarray
    sampling_rate: int
    phoneme_list: list[BasePhoneme]
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
    accent: Tensor
    speaker_id: Tensor


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

    return OutputData(
        wave=torch.from_numpy(numpy.asarray(resampled, dtype=numpy.float32)),
        phoneme_index=torch.from_numpy(phoneme_index).long(),
        phoneme_id=torch.from_numpy(phoneme_id).long(),
        vowel_index=torch.from_numpy(vowel_index).long(),
        accent=torch.from_numpy(accent).long(),
        speaker_id=torch.tensor(d.speaker_id).long(),
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
