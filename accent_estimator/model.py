from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import TypedDict

from accent_estimator.config import ModelConfig
from accent_estimator.dataset import OutputData
from accent_estimator.network.predictor import Predictor


class ModelOutput(TypedDict):
    loss: Tensor
    loss_accent_start: Tensor
    loss_accent_end: Tensor
    loss_accent_phrase_start: Tensor
    loss_accent_phrase_end: Tensor
    precision_accent_start: Tensor
    precision_accent_end: Tensor
    precision_accent_phrase_start: Tensor
    precision_accent_phrase_end: Tensor
    recall_accent_start: Tensor
    recall_accent_end: Tensor
    recall_accent_phrase_start: Tensor
    recall_accent_phrase_end: Tensor
    data_num: int


def reduce_result(results: List[ModelOutput]):
    result: Dict[str, Any] = {}
    sum_data_num = sum([r["data_num"] for r in results])
    for key in set(results[0].keys()) - {"data_num"}:
        values = [r[key] * r["data_num"] for r in results]
        if isinstance(values[0], Tensor):
            result[key] = torch.stack(values).sum() / sum_data_num
        else:
            result[key] = sum(values) / sum_data_num
    return result


def calc(output: Tensor, target: Tensor):
    loss = F.binary_cross_entropy_with_logits(output, target.float())
    tp = ((output >= 0) & (target == 1)).float().sum()
    fp = ((output >= 0) & (target == 0)).float().sum()
    fn = ((output < 0) & (target == 1)).float().sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return loss, precision, recall


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, data: OutputData) -> ModelOutput:
        output_list: List[Tensor]
        _, output_list = self.predictor(
            frame_f0_list=data["frame_f0"],
            frame_phoneme_list=data["frame_phoneme"],
            frame_mora_index_list=data["frame_mora_index"],
            mora_f0_list=data["mora_f0"],
            mora_vowel_list=data["mora_vowel"],
            mora_consonant_list=data["mora_consonant"],
        )

        output = torch.cat(output_list)
        output_accent_start = output[:, 0]
        output_accent_end = output[:, 1]
        output_accent_phrase_start = output[:, 2]
        output_accent_phrase_end = output[:, 3]

        target_accent_start = torch.cat(data["accent_start"])
        target_accent_end = torch.cat(data["accent_end"])
        target_accent_phrase_start = torch.cat(data["accent_phrase_start"])
        target_accent_phrase_end = torch.cat(data["accent_phrase_end"])

        loss_accent_start, precision_accent_start, recall_accent_start = calc(
            output_accent_start, target_accent_start
        )
        loss_accent_end, precision_accent_end, recall_accent_end = calc(
            output_accent_end, target_accent_end
        )
        (
            loss_accent_phrase_start,
            precision_accent_phrase_start,
            recall_accent_phrase_start,
        ) = calc(output_accent_phrase_start, target_accent_phrase_start)
        (
            loss_accent_phrase_end,
            precision_accent_phrase_end,
            recall_accent_phrase_end,
        ) = calc(output_accent_phrase_end, target_accent_phrase_end)

        loss = (
            loss_accent_start
            + loss_accent_end
            + loss_accent_phrase_start
            + loss_accent_phrase_end
        )

        return ModelOutput(
            loss=loss,
            loss_accent_start=loss_accent_start,
            loss_accent_end=loss_accent_end,
            loss_accent_phrase_start=loss_accent_phrase_start,
            loss_accent_phrase_end=loss_accent_phrase_end,
            precision_accent_start=precision_accent_start,
            precision_accent_end=precision_accent_end,
            precision_accent_phrase_start=precision_accent_phrase_start,
            precision_accent_phrase_end=precision_accent_phrase_end,
            recall_accent_start=recall_accent_start,
            recall_accent_end=recall_accent_end,
            recall_accent_phrase_start=recall_accent_phrase_start,
            recall_accent_phrase_end=recall_accent_phrase_end,
            data_num=len(data),
        )
