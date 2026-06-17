"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self

from torch import Tensor, nn
from torch.nn.functional import cross_entropy

from .batch import BatchOutput
from .config import ModelConfig
from .network.predictor import Predictor
from .network.transformer.utility import make_non_pad_mask
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

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


def _precision_recall(
    output: Tensor,  # (N, 2)
    target: Tensor,  # (N,)
) -> tuple[Tensor, Tensor]:
    """2 クラス分類の precision と recall を計算"""
    pred_pos = output[:, 1] > output[:, 0]
    target_pos = target == 1
    tp = (pred_pos & target_pos).sum()
    fp = (pred_pos & ~target_pos).sum()
    fn = (~pred_pos & target_pos).sum()
    precision = tp.float() / (tp + fp).clamp(min=1).float()
    recall = tp.float() / (tp + fn).clamp(min=1).float()
    return precision, recall


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        output = self.predictor(
            vowel=batch.vowel,
            wave=batch.wave,
            mora_index=batch.mora_index,
            speaker_id=batch.speaker_id,
            wave_length=batch.wave_length,
            mora_length=batch.mora_length,
        )  # (B, max(mL), 2, 4)

        max_mora_length = output.size(1)
        mora_mask = make_non_pad_mask(batch.mora_length).to(
            output.device
        )  # (B, max(mL))

        flat_output = output[mora_mask]  # (sum(mL), 2, 4)
        flat_target = batch.accent[:, :max_mora_length][mora_mask]  # (sum(mL), 4)

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

        return ModelOutput(
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
