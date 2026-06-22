"""データセットモジュール"""

import hashlib
import random
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import assert_never

import h5py
import numpy
from pydantic import TypeAdapter
from torch.utils.data import Dataset as BaseDataset
from upath import UPath

from .config import DataFileConfig, DatasetConfig
from .data.data import InputData, OutputData, preprocess
from .data.phoneme import OjtPhoneme
from .data.sampling_data import SamplingData
from .data.wave import Wave
from .utility.pathlist_utility import get_data_paths
from .utility.upath_utility import to_local_path


def _read_bool_list(path: Path) -> list[bool]:
    """空白区切りで 0/1 が並ぶテキストを bool 配列に変換"""
    return [bool(int(s)) for s in path.read_text().split()]


@dataclass
class LazyInputData:
    """遅延読み込み対応の入力データ構造"""

    wave_path: UPath
    phoneme_list_path: UPath
    f0_path: UPath
    volume_path: UPath
    accent_start_path: UPath
    accent_end_path: UPath
    accent_phrase_start_path: UPath
    accent_phrase_end_path: UPath
    speaker_id: int
    hdf5_cache_dir: UPath | None

    def _generate_hdf5_cache_filename(self) -> str:
        paths_str = "\n".join(
            [
                str(self.wave_path),
                str(self.phoneme_list_path),
                str(self.f0_path),
                str(self.volume_path),
                str(self.accent_start_path),
                str(self.accent_end_path),
                str(self.accent_phrase_start_path),
                str(self.accent_phrase_end_path),
            ]
        )
        hash_value = hashlib.sha256(paths_str.encode()).hexdigest()
        return f"{hash_value}.h5"

    def _get_hdf5_cache_path(self) -> UPath:
        if self.hdf5_cache_dir is None:
            raise RuntimeError("hdf5_cache_dirがNoneです")
        filename = self._generate_hdf5_cache_filename()
        return self.hdf5_cache_dir / filename

    def _get_manifest(self) -> dict[str, str]:
        return {
            "wave_path": str(self.wave_path),
            "phoneme_list_path": str(self.phoneme_list_path),
            "f0_path": str(self.f0_path),
            "volume_path": str(self.volume_path),
            "accent_start_path": str(self.accent_start_path),
            "accent_end_path": str(self.accent_end_path),
            "accent_phrase_start_path": str(self.accent_phrase_start_path),
            "accent_phrase_end_path": str(self.accent_phrase_end_path),
        }

    def _write_hdf5_cache(self, d: InputData) -> None:
        cache_path = self._get_hdf5_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            local_path = Path(tmp.name)

        with h5py.File(local_path, "w") as f:
            f.create_dataset("wave", data=d.wave)
            f.attrs["sampling_rate"] = d.sampling_rate
            phoneme_text = "\n".join(
                f"{p.start:.4f}\t{p.end:.4f}\t{p.phoneme}" for p in d.phoneme_list
            )
            f.create_dataset("phoneme_text", data=phoneme_text)
            f.create_dataset("f0_array", data=d.f0.array)
            f.create_dataset("f0_rate", data=d.f0.rate)
            f.create_dataset("volume_array", data=d.volume.array)
            f.create_dataset("volume_rate", data=d.volume.rate)
            f.create_dataset("accent_start", data=numpy.asarray(d.accent_start))
            f.create_dataset("accent_end", data=numpy.asarray(d.accent_end))
            f.create_dataset(
                "accent_phrase_start", data=numpy.asarray(d.accent_phrase_start)
            )
            f.create_dataset(
                "accent_phrase_end", data=numpy.asarray(d.accent_phrase_end)
            )
            manifest = self._get_manifest()
            for key, value in manifest.items():
                f.attrs[f"manifest_{key}"] = value

        cache_path.write_bytes(local_path.read_bytes())
        local_path.unlink()

    @staticmethod
    def _read_hdf5_cache(cache_path: Path, speaker_id: int) -> InputData:
        with h5py.File(cache_path, "r") as f:
            phoneme_text = numpy.asarray(f["phoneme_text"]).item()
            assert isinstance(phoneme_text, bytes)
            phoneme_text = phoneme_text.decode()
            phoneme_list = OjtPhoneme.loads_julius_list(phoneme_text)
            return InputData(
                wave=numpy.array(f["wave"]),
                sampling_rate=int(f.attrs["sampling_rate"]),  # type: ignore
                phoneme_list=phoneme_list,
                f0=SamplingData(
                    array=numpy.array(f["f0_array"]),
                    rate=float(numpy.array(f["f0_rate"])),
                ),
                volume=SamplingData(
                    array=numpy.array(f["volume_array"]),
                    rate=float(numpy.array(f["volume_rate"])),
                ),
                accent_start=numpy.array(f["accent_start"]).astype(bool).tolist(),
                accent_end=numpy.array(f["accent_end"]).astype(bool).tolist(),
                accent_phrase_start=(
                    numpy.array(f["accent_phrase_start"]).astype(bool).tolist()
                ),
                accent_phrase_end=(
                    numpy.array(f["accent_phrase_end"]).astype(bool).tolist()
                ),
                speaker_id=speaker_id,
            )

    def _fetch_from_files(self) -> InputData:
        loaded = Wave.load(to_local_path(self.wave_path))
        wave = loaded.wave
        if wave.ndim != 1:
            wave = wave.mean(axis=1)
        return InputData(
            wave=wave,
            sampling_rate=loaded.sampling_rate,
            phoneme_list=OjtPhoneme.loads_julius_list(
                to_local_path(self.phoneme_list_path).read_text()
            ),
            f0=SamplingData.load(to_local_path(self.f0_path)),
            volume=SamplingData.load(to_local_path(self.volume_path)),
            accent_start=_read_bool_list(to_local_path(self.accent_start_path)),
            accent_end=_read_bool_list(to_local_path(self.accent_end_path)),
            accent_phrase_start=_read_bool_list(
                to_local_path(self.accent_phrase_start_path)
            ),
            accent_phrase_end=_read_bool_list(
                to_local_path(self.accent_phrase_end_path)
            ),
            speaker_id=self.speaker_id,
        )

    def fetch(self) -> InputData:
        """ファイルからデータを読み込んでInputDataを生成"""
        if self.hdf5_cache_dir is None:
            return self._fetch_from_files()

        cache_path = self._get_hdf5_cache_path()
        if cache_path.exists():
            local_cache = to_local_path(cache_path)
            return self._read_hdf5_cache(local_cache, self.speaker_id)

        input_data = self._fetch_from_files()
        self._write_hdf5_cache(input_data)
        return input_data


def prefetch_datas(
    train_datas: list[LazyInputData],
    test_datas: list[LazyInputData],
    valid_datas: list[LazyInputData] | None,
    train_indices: list[int],
    train_batch_size: int,
    num_prefetch: int,
) -> Callable[[], None]:
    """データセットを学習順序に従って非同期で読み込み、停止関数を返す"""
    if num_prefetch <= 0:
        return lambda: None

    prefetch_order: list[LazyInputData] = []
    prefetch_order += [train_datas[i] for i in train_indices[:train_batch_size]]
    prefetch_order += test_datas
    prefetch_order += [train_datas[i] for i in train_indices[train_batch_size:]]
    if valid_datas is not None:
        prefetch_order += valid_datas

    executor = ThreadPoolExecutor(max_workers=num_prefetch)
    for data in prefetch_order:
        executor.submit(data.fetch)

    def close() -> None:
        executor.shutdown(wait=False, cancel_futures=True)

    return close


class Dataset(BaseDataset[OutputData]):
    """メインのデータセット"""

    def __init__(
        self,
        datas: list[LazyInputData],
        config: DatasetConfig,
        is_eval: bool,
    ):
        self.datas = datas
        self.config = config
        self.is_eval = is_eval

    def __len__(self):
        """データセットのサイズ"""
        return len(self.datas)

    def __getitem__(self, i: int) -> OutputData:
        """指定されたインデックスのデータを前処理して返す"""
        try:
            input_data = self.datas[i].fetch()
            return preprocess(
                input_data,
                is_eval=self.is_eval,
                sampling_rate=self.config.sampling_rate,
                frame_rate=self.config.frame_rate,
            )
        except Exception as e:
            raise RuntimeError(
                f"データ処理に失敗しました: index={i} data={self.datas[i]}"
            ) from e


class DatasetType(str, Enum):
    """データセットタイプ"""

    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"
    VALID = "valid"


@dataclass
class DatasetCollection:
    """データセットコレクション"""

    train: Dataset
    """重みの更新に用いる"""

    test: Dataset
    """trainと同じドメインでモデルの過適合確認に用いる"""

    eval: Dataset | None
    """testと同じデータを評価に用いる"""

    valid: Dataset | None
    """trainやtestと異なり、評価専用に用いる"""

    def get(self, type: DatasetType) -> Dataset:
        """指定されたタイプのデータセットを返す"""
        match type:
            case DatasetType.TRAIN:
                return self.train
            case DatasetType.TEST:
                return self.test
            case DatasetType.EVAL:
                if self.eval is None:
                    raise ValueError("evalデータセットが設定されていません")
                return self.eval
            case DatasetType.VALID:
                if self.valid is None:
                    raise ValueError("validデータセットが設定されていません")
                return self.valid
            case _:
                assert_never(type)


def get_datas(
    config: DataFileConfig, hdf5_cache_dir: UPath | None
) -> list[LazyInputData]:
    """データを取得"""
    (
        fn_list,
        (
            wave_pathmappings,
            phoneme_list_pathmappings,
            f0_pathmappings,
            volume_pathmappings,
            accent_start_pathmappings,
            accent_end_pathmappings,
            accent_phrase_start_pathmappings,
            accent_phrase_end_pathmappings,
        ),
    ) = get_data_paths(
        config.root_dir,
        [
            config.wave_pathlist_path,
            config.phoneme_list_pathlist_path,
            config.f0_pathlist_path,
            config.volume_pathlist_path,
            config.accent_start_pathlist_path,
            config.accent_end_pathlist_path,
            config.accent_phrase_start_pathlist_path,
            config.accent_phrase_end_pathlist_path,
        ],
    )

    fn_each_speaker = TypeAdapter(dict[str, list[str]]).validate_json(
        to_local_path(config.speaker_dict_path).read_text()
    )
    speaker_ids = {
        fn: speaker_id
        for speaker_id, fns in enumerate(fn_each_speaker.values())
        for fn in fns
    }

    datas = [
        LazyInputData(
            wave_path=wave_pathmappings[fn],
            phoneme_list_path=phoneme_list_pathmappings[fn],
            f0_path=f0_pathmappings[fn],
            volume_path=volume_pathmappings[fn],
            accent_start_path=accent_start_pathmappings[fn],
            accent_end_path=accent_end_pathmappings[fn],
            accent_phrase_start_path=accent_phrase_start_pathmappings[fn],
            accent_phrase_end_path=accent_phrase_end_pathmappings[fn],
            speaker_id=speaker_ids[fn],
            hdf5_cache_dir=hdf5_cache_dir,
        )
        for fn in fn_list
    ]
    return datas


def create_dataset(config: DatasetConfig) -> DatasetCollection:
    """データセットを作成"""
    datas = get_datas(config.train, config.hdf5_cache_dir)

    if config.seed is not None:
        random.Random(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]
    if config.train_num is not None:
        trains = trains[: config.train_num]

    def _wrapper(datas: list[LazyInputData], is_eval: bool) -> Dataset:
        if is_eval:
            datas = datas * config.eval_times_num
        dataset = Dataset(datas=datas, config=config, is_eval=is_eval)
        return dataset

    return DatasetCollection(
        train=_wrapper(trains, is_eval=False),
        test=_wrapper(tests, is_eval=False),
        eval=(_wrapper(tests, is_eval=True) if config.eval_for_test else None),
        valid=(
            _wrapper(get_datas(config.valid, config.hdf5_cache_dir), is_eval=True)
            if config.valid is not None
            else None
        ),
    )
