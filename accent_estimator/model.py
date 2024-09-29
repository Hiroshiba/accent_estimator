from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import TypedDict

from accent_estimator.config import ModelConfig
from accent_estimator.dataset import DatasetOutput
from accent_estimator.network.predictor import Predictor


class ModelOutput(TypedDict):
    loss: Tensor
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
    tp = ((output[:, 1] > output[:, 0]) & (target == 1)).sum()
    fp = ((output[:, 1] > output[:, 0]) & (target == 0)).sum()
    fn = ((output[:, 1] <= output[:, 0]) & (target == 1)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, data: DatasetOutput) -> ModelOutput:
        output_list: List[Tensor] = self.predictor(
            vowel_list=data["vowel"],
            mora_position_list=data["mora_position"],
            feature_list=data["feature"],
            frame_position_list=data["frame_position"],
        )

        output = torch.cat(output_list)

        target_accent = torch.cat(data["accent"])

        loss = F.cross_entropy(output, target_accent)

        precision_accent_start, recall_accent_start = calc(
            output[:, :, 0], target_accent[:, 0]
        )
        precision_accent_end, recall_accent_end = calc(
            output[:, :, 1], target_accent[:, 1]
        )
        precision_accent_phrase_start, recall_accent_phrase_start = calc(
            output[:, :, 2], target_accent[:, 2]
        )
        precision_accent_phrase_end, recall_accent_phrase_end = calc(
            output[:, :, 3], target_accent[:, 3]
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
            data_num=len(data),
        )
