"""データ処理の共通モジュール"""

from pathlib import Path

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


def read_bool_list(path: Path) -> list[bool]:
    """空白区切りで 0/1 が並ぶテキストを bool 配列に変換"""
    return [bool(int(s)) for s in path.read_text().split()]
