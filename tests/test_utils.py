"""テストの便利モジュール"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data.phoneme import OjtPhoneme
from hiho_pytorch_base.data.sampling_data import SamplingData

_FRAME_RATE = 50

_VOWELS = ("a", "i", "u", "e", "o")
_CONSONANTS = ("k", "s", "t", "n", "h", "m", "y", "r", "w", "g", "z", "d", "b", "p")


def _generate_phoneme_sequence(num_mora: int) -> list[OjtPhoneme]:
    """ランダムな音素列を生成。pau で挟み、各モーラはオプション子音+母音"""
    rng = np.random.default_rng()

    phoneme_strs: list[str] = ["pau"]
    for _ in range(num_mora):
        if rng.random() < 0.7:
            phoneme_strs.append(_CONSONANTS[int(rng.integers(0, len(_CONSONANTS)))])
        phoneme_strs.append(_VOWELS[int(rng.integers(0, len(_VOWELS)))])
    phoneme_strs.append("pau")

    durations_frames = rng.integers(2, 8, size=len(phoneme_strs))
    boundaries_frames = np.concatenate([[0], np.cumsum(durations_frames)])
    boundaries_seconds = boundaries_frames / _FRAME_RATE

    phonemes: list[OjtPhoneme] = []
    for i, phoneme in enumerate(phoneme_strs):
        phonemes.append(
            OjtPhoneme(
                phoneme=phoneme,
                start=float(boundaries_seconds[i]),
                end=float(boundaries_seconds[i + 1]),
            )
        )
    return phonemes


def _phonemes_to_julius_text(phonemes: list[OjtPhoneme]) -> str:
    return "\n".join(f"{p.start:.4f} {p.end:.4f} {p.phoneme}" for p in phonemes)


def setup_data_and_config(base_config_path: Path, data_dir: UPath) -> Config:
    """テストデータをセットアップし、設定を作る"""
    config = Config.load(UPath(base_config_path))
    assert config.dataset.valid is not None

    config.dataset.train.root_dir = data_dir
    config.dataset.valid.root_dir = data_dir
    config.dataset.hdf5_cache_dir = data_dir / "hdf5_cache"

    root_dir = config.dataset.train.root_dir
    train_num, valid_num = 30, 10
    all_stems = list(map(str, range(train_num + valid_num)))

    def _setup_data(
        generator_func: Callable[[Path], None], data_type: str, extension: str
    ) -> None:
        train_pathlist_path = data_dir / f"train_{data_type}_pathlist.txt"
        valid_pathlist_path = data_dir / f"valid_{data_type}_pathlist.txt"

        setattr(config.dataset.train, f"{data_type}_pathlist_path", train_pathlist_path)
        setattr(config.dataset.valid, f"{data_type}_pathlist_path", valid_pathlist_path)

        data_dir_path = root_dir / data_type
        data_dir_path.mkdir(parents=True, exist_ok=True)

        all_relative_paths = [f"{data_type}/{stem}.{extension}" for stem in all_stems]
        for relative_path in all_relative_paths:
            file_path = root_dir / relative_path
            if not file_path.exists():
                generator_func(file_path)

        if not train_pathlist_path.exists():
            train_pathlist_path.write_text("\n".join(all_relative_paths[:train_num]))
        if not valid_pathlist_path.exists():
            valid_pathlist_path.write_text("\n".join(all_relative_paths[train_num:]))

    # 音素列をstemごとに事前生成
    phoneme_lists = {}
    for stem in all_stems:
        num_mora = int(np.random.default_rng().integers(3, 12))
        phoneme_lists[stem] = _generate_phoneme_sequence(num_mora)

    # 音声波形
    _SAMPLE_RATE = 16000

    def generate_wave(file_path: Path) -> None:
        phonemes = phoneme_lists[file_path.stem]
        total_seconds = phonemes[-1].end
        num_samples = int(round(total_seconds * _SAMPLE_RATE))
        wave = np.random.default_rng().normal(size=num_samples).astype(np.float32)
        soundfile.write(str(file_path), wave, _SAMPLE_RATE)

    _setup_data(generate_wave, "wave", "wav")

    # 音素ラベル
    def generate_phoneme_list(file_path: Path) -> None:
        text = _phonemes_to_julius_text(phoneme_lists[file_path.stem])
        file_path.write_text(text)

    _setup_data(generate_phoneme_list, "phoneme_list", "lab")

    # フレーム基本周波数
    def generate_f0(file_path: Path) -> None:
        phonemes = phoneme_lists[file_path.stem]
        num_frames = int(round(phonemes[-1].end * _FRAME_RATE))
        f0 = np.random.default_rng().uniform(100.0, 300.0, size=num_frames)
        SamplingData(array=f0.astype(np.float32), rate=_FRAME_RATE).save(file_path)

    _setup_data(generate_f0, "f0", "npy")

    # フレーム音量
    def generate_volume(file_path: Path) -> None:
        phonemes = phoneme_lists[file_path.stem]
        num_frames = int(round(phonemes[-1].end * _FRAME_RATE))
        volume = np.random.default_rng().uniform(0.0, 1.0, size=num_frames)
        SamplingData(array=volume.astype(np.float32), rate=_FRAME_RATE).save(file_path)

    _setup_data(generate_volume, "volume", "npy")

    # アクセント核開始
    def generate_accent_start(file_path: Path) -> None:
        phonemes = phoneme_lists[file_path.stem]
        values = (np.random.default_rng().random(size=len(phonemes)) < 0.2).astype(
            np.int64
        )
        file_path.write_text(" ".join(str(int(v)) for v in values))

    _setup_data(generate_accent_start, "accent_start", "txt")

    # アクセント核終了
    def generate_accent_end(file_path: Path) -> None:
        phonemes = phoneme_lists[file_path.stem]
        values = (np.random.default_rng().random(size=len(phonemes)) < 0.2).astype(
            np.int64
        )
        file_path.write_text(" ".join(str(int(v)) for v in values))

    _setup_data(generate_accent_end, "accent_end", "txt")

    # アクセント句開始
    def generate_accent_phrase_start(file_path: Path) -> None:
        phonemes = phoneme_lists[file_path.stem]
        values = (np.random.default_rng().random(size=len(phonemes)) < 0.1).astype(
            np.int64
        )
        file_path.write_text(" ".join(str(int(v)) for v in values))

    _setup_data(generate_accent_phrase_start, "accent_phrase_start", "txt")

    # アクセント句終了
    def generate_accent_phrase_end(file_path: Path) -> None:
        phonemes = phoneme_lists[file_path.stem]
        values = (np.random.default_rng().random(size=len(phonemes)) < 0.1).astype(
            np.int64
        )
        file_path.write_text(" ".join(str(int(v)) for v in values))

    _setup_data(generate_accent_phrase_end, "accent_phrase_end", "txt")

    # 話者マッピング
    speaker_names = ["A", "B", "C"]
    speaker_dict = {name: [] for name in speaker_names}
    for stem in all_stems:
        speaker_name = speaker_names[int(stem) % len(speaker_names)]
        speaker_dict[speaker_name].append(stem)

    speaker_dict_path = data_dir / "speaker_dict.json"
    speaker_dict_path.write_text(json.dumps(speaker_dict))
    config.dataset.train.speaker_dict_path = speaker_dict_path
    config.dataset.valid.speaker_dict_path = speaker_dict_path

    return config
