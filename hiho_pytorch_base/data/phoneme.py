"""音素データ構造モジュール"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import Self

import numpy


class BasePhoneme:
    """julius 形式音素の基底クラス"""

    phoneme_list: tuple[str, ...]
    num_phoneme: int
    space_phoneme: str

    def __init__(self, phoneme: str, start: float, end: float):
        self.phoneme = phoneme
        self.start = numpy.round(start, decimals=4)
        self.end = numpy.round(end, decimals=4)

    def __repr__(self) -> str:  # noqa: D105
        return f"Phoneme(phoneme='{self.phoneme}', start={self.start}, end={self.end})"

    def __eq__(self, o: object) -> bool:  # noqa: D105
        return isinstance(o, BasePhoneme) and (
            self.phoneme == o.phoneme and self.start == o.start and self.end == o.end
        )

    def verify(self) -> None:
        """音素の整合性を検証"""
        assert self.start < self.end, f"{self} start must be less than end"
        assert self.phoneme in self.phoneme_list, f"{self} is not defined."

    @property
    def phoneme_id(self) -> int:
        """音素 ID を返す"""
        return self.phoneme_list.index(self.phoneme)

    @property
    def duration(self) -> float:
        """音素長を返す"""
        return self.end - self.start

    @property
    def onehot(self) -> numpy.ndarray:
        """音素 ID の onehot 表現を返す"""
        array = numpy.zeros(self.num_phoneme, dtype=bool)
        array[self.phoneme_id] = True
        return array

    @classmethod
    def parse(cls, s: str) -> Self:
        """julius 形式の 1 行文字列をパース"""
        words = s.split()
        return cls(start=float(words[0]), end=float(words[1]), phoneme=words[2])

    @classmethod
    @abstractmethod
    def convert(cls, phonemes: Sequence["BasePhoneme"]) -> list["BasePhoneme"]:
        """音素列の前後処理"""

    @classmethod
    def verify_list(cls, phonemes: Sequence["BasePhoneme"]) -> None:
        """音素列の整合性を検証"""
        assert phonemes[0].start == 0, f"{phonemes[0]} start must be 0."
        for phoneme in phonemes:
            phoneme.verify()
        for pre, post in zip(phonemes[:-1], phonemes[1:], strict=True):
            assert pre.end == post.start, f"{pre} and {post} must be continuous."

    @classmethod
    def loads_julius_list(cls, text: str) -> list["BasePhoneme"]:
        """julius 形式テキストから音素列を生成"""
        phonemes = [cls.parse(s) for s in text.split("\n") if len(s) > 0]
        phonemes = cls.convert(phonemes)
        cls.verify_list(phonemes)
        return phonemes


class OjtPhoneme(BasePhoneme):
    """OpenJTalk 互換の音素クラス"""

    phoneme_list = (
        "pau",
        "A",
        "E",
        "I",
        "N",
        "O",
        "U",
        "a",
        "b",
        "by",
        "ch",
        "cl",
        "d",
        "dy",
        "e",
        "f",
        "g",
        "gw",
        "gy",
        "h",
        "hy",
        "i",
        "j",
        "k",
        "kw",
        "ky",
        "m",
        "my",
        "n",
        "ny",
        "o",
        "p",
        "py",
        "r",
        "ry",
        "s",
        "sh",
        "t",
        "ts",
        "ty",
        "u",
        "v",
        "w",
        "y",
        "z",
    )
    num_phoneme = len(phoneme_list)
    space_phoneme = "pau"

    @classmethod
    def convert(cls, phonemes: Sequence["BasePhoneme"]) -> list["BasePhoneme"]:
        """前後のsilとbrをpauに変換する"""
        result = list(phonemes)
        for phoneme in result:
            if phoneme.phoneme == "br":
                phoneme.phoneme = cls.space_phoneme
        if "sil" in result[0].phoneme:
            result[0].phoneme = cls.space_phoneme
        if "sil" in result[-1].phoneme:
            result[-1].phoneme = cls.space_phoneme
        return result
