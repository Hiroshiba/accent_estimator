"""統計情報モジュール"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Self

import numpy
from upath import UPath

from ..config import DataFileConfig, DatasetConfig
from ..utility.upath_utility import to_local_path
from .data import mora_phoneme_list, read_bool_list
from .phoneme import OjtPhoneme


@dataclass
class DataStatistics:
    """アクセント one-hot チャンネルごとの統計情報"""

    accent_mean: numpy.ndarray  # (2, 4)
    accent_std: numpy.ndarray  # (2, 4)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """辞書から統計情報を生成"""
        return cls(
            accent_mean=numpy.asarray(d["accent_mean"], dtype=numpy.float64),
            accent_std=numpy.asarray(d["accent_std"], dtype=numpy.float64),
        )

    def to_dict(self) -> dict[str, Any]:
        """統計情報を辞書に変換"""
        return {
            "accent_mean": self.accent_mean.tolist(),
            "accent_std": self.accent_std.tolist(),
        }


def _get_statistics_cache_key_and_info(
    config: DataFileConfig,
) -> tuple[str, dict[str, str | None]]:
    root_dir = None if config.root_dir is None else str(config.root_dir)

    phoneme_list_pathlist_text = to_local_path(
        config.phoneme_list_pathlist_path
    ).read_text()
    phoneme_list_pathlist_hash = hashlib.sha256(
        phoneme_list_pathlist_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    accent_start_pathlist_text = to_local_path(
        config.accent_start_pathlist_path
    ).read_text()
    accent_start_pathlist_hash = hashlib.sha256(
        accent_start_pathlist_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    accent_end_pathlist_text = to_local_path(
        config.accent_end_pathlist_path
    ).read_text()
    accent_end_pathlist_hash = hashlib.sha256(
        accent_end_pathlist_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    accent_phrase_start_pathlist_text = to_local_path(
        config.accent_phrase_start_pathlist_path
    ).read_text()
    accent_phrase_start_pathlist_hash = hashlib.sha256(
        accent_phrase_start_pathlist_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    accent_phrase_end_pathlist_text = to_local_path(
        config.accent_phrase_end_pathlist_path
    ).read_text()
    accent_phrase_end_pathlist_hash = hashlib.sha256(
        accent_phrase_end_pathlist_text.encode("utf-8", errors="surrogatepass")
    ).hexdigest()

    info = {
        "root_dir": root_dir,
        "phoneme_list_pathlist_path": str(config.phoneme_list_pathlist_path),
        "phoneme_list_pathlist_hash": phoneme_list_pathlist_hash,
        "accent_start_pathlist_path": str(config.accent_start_pathlist_path),
        "accent_start_pathlist_hash": accent_start_pathlist_hash,
        "accent_end_pathlist_path": str(config.accent_end_pathlist_path),
        "accent_end_pathlist_hash": accent_end_pathlist_hash,
        "accent_phrase_start_pathlist_path": str(
            config.accent_phrase_start_pathlist_path
        ),
        "accent_phrase_start_pathlist_hash": accent_phrase_start_pathlist_hash,
        "accent_phrase_end_pathlist_path": str(config.accent_phrase_end_pathlist_path),
        "accent_phrase_end_pathlist_hash": accent_phrase_end_pathlist_hash,
    }

    cache_key = hashlib.sha256(
        json.dumps(info, sort_keys=True, ensure_ascii=False).encode(
            "utf-8", errors="surrogatepass"
        )
    ).hexdigest()
    return cache_key, info


@dataclass(frozen=True)
class StatisticsDataInput:
    """統計情報計算用データ"""

    phoneme_list_path: UPath
    accent_start_path: UPath
    accent_end_path: UPath
    accent_phrase_start_path: UPath
    accent_phrase_end_path: UPath


def _load_statistics_item(d: StatisticsDataInput) -> numpy.ndarray:
    """アクセント one-hot 配列 (mL, 2, 4) を生成"""
    phoneme_list = OjtPhoneme.loads_julius_list(
        to_local_path(d.phoneme_list_path).read_text()
    )
    accent_start = read_bool_list(to_local_path(d.accent_start_path))
    accent_end = read_bool_list(to_local_path(d.accent_end_path))
    accent_phrase_start = read_bool_list(to_local_path(d.accent_phrase_start_path))
    accent_phrase_end = read_bool_list(to_local_path(d.accent_phrase_end_path))

    assert len(phoneme_list) == len(accent_start), (
        f"音素列とアクセント列の長さが一致しません: "
        f"len(phoneme_list)={len(phoneme_list)}, len(accent_start)={len(accent_start)}"
    )

    mora_indexes = [
        i for i, p in enumerate(phoneme_list) if p.phoneme in mora_phoneme_list
    ]
    accent = numpy.stack(
        [
            numpy.array([accent_start[i] for i in mora_indexes]),
            numpy.array([accent_end[i] for i in mora_indexes]),
            numpy.array([accent_phrase_start[i] for i in mora_indexes]),
            numpy.array([accent_phrase_end[i] for i in mora_indexes]),
        ],
        axis=1,
    )  # (mL, 4)

    onehot = numpy.zeros((len(accent), 2, 4), dtype=numpy.float64)
    onehot[:, 0, :] = ~accent
    onehot[:, 1, :] = accent
    return onehot


def _calc_statistics(
    datas: list[StatisticsDataInput], *, workers: int
) -> DataStatistics:
    """アクセント one-hot チャンネルごとの統計情報を計算する"""
    if workers <= 0:
        raise ValueError(f"workers must be > 0: {workers}")
    if len(datas) == 0:
        raise ValueError("datas is empty")

    print(f"統計情報を計算しています... (データ数: {len(datas)})")

    accent_count = numpy.zeros((2, 4), dtype=numpy.int64)
    accent_sum = numpy.zeros((2, 4), dtype=numpy.float64)
    accent_sumsq = numpy.zeros((2, 4), dtype=numpy.float64)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_load_statistics_item, d) for d in datas]
        for future in as_completed(futures):
            onehot = future.result()
            accent_count += onehot.shape[0]
            accent_sum += onehot.sum(axis=0)
            accent_sumsq += (onehot * onehot).sum(axis=0)

    accent_mean = accent_sum / accent_count
    accent_var = accent_sumsq / accent_count - accent_mean * accent_mean
    accent_std = numpy.sqrt(accent_var)

    return DataStatistics(accent_mean=accent_mean, accent_std=accent_std)


def get_or_calc_statistics(
    config: DatasetConfig, datas: list[StatisticsDataInput], *, workers: int
) -> DataStatistics:
    """統計情報を取得または計算する。statistics_cache_dirが未設定の場合はキャッシュせず毎回計算する"""
    if config.statistics_cache_dir is None:
        return _calc_statistics(datas, workers=workers)

    cache_key, info = _get_statistics_cache_key_and_info(config.train)
    cache_dir = config.statistics_cache_dir / cache_key
    info_path = cache_dir / "info.json"
    statistics_path = cache_dir / "statistics.json"

    if statistics_path.exists():
        print(f"統計情報をキャッシュから読み込みました: {statistics_path}")
        statistics_dict = json.loads(statistics_path.read_text())
        return DataStatistics.from_dict(statistics_dict)

    statistics = _calc_statistics(datas, workers=workers)

    cache_dir.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(json.dumps(statistics.to_dict(), ensure_ascii=False))
    info_path.write_text(json.dumps(info, ensure_ascii=False))

    return statistics
