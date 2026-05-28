"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn

from .batch import BatchOutput
from .generator import Generator, GeneratorOutput
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    loss: Tensor
    accuracy: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.accuracy = detach_cpu(self.accuracy)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return output.accuracy


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        output_result: GeneratorOutput = self.generator(
            feature_vector=batch.feature_vector,
            feature_variable=batch.feature_variable,
            speaker_id=batch.speaker_id,
            length=batch.length,
        )

        output = output_result.vector_output  # (B, ?)
        target = batch.target_vector  # (B,)

        loss = torch.nn.functional.cross_entropy(output, target)

        indexes = torch.argmax(output, dim=1)  # (B,)
        correct = torch.eq(indexes, target).view(-1)  # (B,)
        accuracy = correct.float().mean()

        return EvaluatorOutput(
            loss=loss,
            accuracy=accuracy,
            data_num=batch.data_num,
        )
