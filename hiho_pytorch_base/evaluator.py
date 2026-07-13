"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, mse_loss

from .batch import BatchOutput
from .generator import Generator, GeneratorOutput
from .model import _precision_recall
from .network.transformer.utility import make_non_pad_mask
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    loss: Tensor
    precision_accent_start: Tensor
    precision_accent_end: Tensor
    precision_accent_phrase_start: Tensor
    precision_accent_phrase_end: Tensor
    recall_accent_start: Tensor
    recall_accent_end: Tensor
    recall_accent_phrase_start: Tensor
    recall_accent_phrase_end: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.precision_accent_start = detach_cpu(self.precision_accent_start)
        self.precision_accent_end = detach_cpu(self.precision_accent_end)
        self.precision_accent_phrase_start = detach_cpu(
            self.precision_accent_phrase_start
        )
        self.precision_accent_phrase_end = detach_cpu(self.precision_accent_phrase_end)
        self.recall_accent_start = detach_cpu(self.recall_accent_start)
        self.recall_accent_end = detach_cpu(self.recall_accent_end)
        self.recall_accent_phrase_start = detach_cpu(self.recall_accent_phrase_start)
        self.recall_accent_phrase_end = detach_cpu(self.recall_accent_phrase_end)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return -output.loss


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        output_result: GeneratorOutput = self.generator(
            wave=batch.wave,
            phoneme_index=batch.phoneme_index,
            phoneme_id=batch.phoneme_id,
            vowel_index=batch.vowel_index,
            mora_f0=batch.mora_f0,
            accent_noise=batch.accent_noise,
            wave_length=batch.wave_length,
            phoneme_length=batch.phoneme_length,
            mora_length=batch.mora_length,
        )

        output = output_result.accent_logit  # (B, max(mL), 2, 4)
        max_mora_length = output.size(1)
        mora_mask = make_non_pad_mask(batch.mora_length).to(output.device)

        flat_output = output[mora_mask]  # (sum(mL), 2, 4)
        flat_target = batch.accent[:, :max_mora_length][mora_mask]  # (sum(mL), 4)

        if self.generator.use_diffusion:
            target_onehot = self.generator.denormalize(
                batch.accent_target
            )  # (B, max(mL), 2, 4)
            flat_target_onehot = target_onehot[:, :max_mora_length][
                mora_mask
            ]  # (sum(mL), 2, 4)
            loss = mse_loss(flat_output, flat_target_onehot)
        else:
            loss = cross_entropy(flat_output, flat_target)

        precision_accent_start, recall_accent_start = _precision_recall(
            flat_output[:, :, 0], flat_target[:, 0]
        )
        precision_accent_end, recall_accent_end = _precision_recall(
            flat_output[:, :, 1], flat_target[:, 1]
        )
        precision_accent_phrase_start, recall_accent_phrase_start = _precision_recall(
            flat_output[:, :, 2], flat_target[:, 2]
        )
        precision_accent_phrase_end, recall_accent_phrase_end = _precision_recall(
            flat_output[:, :, 3], flat_target[:, 3]
        )

        return EvaluatorOutput(
            loss=loss,
            precision_accent_start=precision_accent_start,
            precision_accent_end=precision_accent_end,
            precision_accent_phrase_start=precision_accent_phrase_start,
            precision_accent_phrase_end=precision_accent_phrase_end,
            recall_accent_start=recall_accent_start,
            recall_accent_end=recall_accent_end,
            recall_accent_phrase_start=recall_accent_phrase_start,
            recall_accent_phrase_end=recall_accent_phrase_end,
            data_num=batch.data_num,
        )
