"""バッチ処理モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .data.data import OutputData
from .utility.pytorch_utility import to_device


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    wave: Tensor  # (B, max(wL))
    phoneme_index: Tensor  # (B, max(fL))
    phoneme_id: Tensor  # (B, max(pL))
    vowel_index: Tensor  # (B, max(mL))
    accent: Tensor  # (B, max(mL), ?)
    speaker_id: Tensor  # (B,)
    wave_length: Tensor  # (B,)
    phoneme_length: Tensor  # (B,)
    mora_length: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.wave.shape[0]

    def to_device(self, device: str, non_blocking: bool) -> Self:
        """データを指定されたデバイスに移動"""
        self.wave = to_device(self.wave, device, non_blocking=non_blocking)
        self.phoneme_index = to_device(
            self.phoneme_index, device, non_blocking=non_blocking
        )
        self.phoneme_id = to_device(self.phoneme_id, device, non_blocking=non_blocking)
        self.vowel_index = to_device(
            self.vowel_index, device, non_blocking=non_blocking
        )
        self.accent = to_device(self.accent, device, non_blocking=non_blocking)
        self.speaker_id = to_device(self.speaker_id, device, non_blocking=non_blocking)
        self.wave_length = to_device(
            self.wave_length, device, non_blocking=non_blocking
        )
        self.phoneme_length = to_device(
            self.phoneme_length, device, non_blocking=non_blocking
        )
        self.mora_length = to_device(
            self.mora_length, device, non_blocking=non_blocking
        )
        return self


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    return BatchOutput(
        wave=pad_sequence([d.wave for d in data_list], batch_first=True),
        phoneme_index=pad_sequence(
            [d.phoneme_index for d in data_list], batch_first=True
        ),
        phoneme_id=pad_sequence([d.phoneme_id for d in data_list], batch_first=True),
        vowel_index=pad_sequence([d.vowel_index for d in data_list], batch_first=True),
        accent=pad_sequence([d.accent for d in data_list], batch_first=True),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
        wave_length=torch.tensor([d.wave.shape[0] for d in data_list]),
        phoneme_length=torch.tensor([d.phoneme_id.shape[0] for d in data_list]),
        mora_length=torch.tensor([d.vowel_index.shape[0] for d in data_list]),
    )
