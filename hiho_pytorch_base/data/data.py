"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor

from .phoneme import BasePhoneme

vowels = ("pau", "a", "i", "u", "e", "o", "n", "cl")

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

    feature: numpy.ndarray
    phoneme_list: list[BasePhoneme]
    accent_start: list[bool]
    accent_end: list[bool]
    accent_phrase_start: list[bool]
    accent_phrase_end: list[bool]
    speaker_id: int


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    vowel: Tensor
    feature: Tensor
    mora_index: Tensor
    accent: Tensor
    speaker_id: Tensor


def preprocess(d: InputData, *, is_eval: bool) -> OutputData:
    """データ処理"""
    _ = is_eval

    frame_rate = 50

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

    vowel = numpy.array([vowel_to_id(d.phoneme_list[i].phoneme) for i in mora_indexes])

    mora_split_second_list = [float(d.phoneme_list[i].end) for i in mora_indexes]
    mora_index = _make_index_array(
        split_second_list=mora_split_second_list,
        rate=frame_rate,
        length=len(d.feature),
    )

    return OutputData(
        vowel=torch.from_numpy(vowel).long(),
        feature=torch.from_numpy(d.feature).float(),
        mora_index=torch.from_numpy(mora_index).long(),
        accent=torch.from_numpy(accent).long(),
        speaker_id=torch.tensor(d.speaker_id).long(),
    )


def vowel_to_id(vowel: str) -> int:
    """母音文字列を母音 ID に変換"""
    if vowel == "sil":
        vowel = "pau"
    vowel = vowel.lower()
    return vowels.index(vowel)


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
