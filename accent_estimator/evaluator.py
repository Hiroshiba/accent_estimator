import torch
from torch import Tensor, nn
from typing_extensions import TypedDict

from accent_estimator.dataset import DatasetOutput
from accent_estimator.generator import Generator, GeneratorOutput
from accent_estimator.model import calc


class EvaluatorOutput(TypedDict):
    precision_accent_start: Tensor
    precision_accent_end: Tensor
    precision_accent_phrase_start: Tensor
    precision_accent_phrase_end: Tensor
    recall_accent_start: Tensor
    recall_accent_end: Tensor
    recall_accent_phrase_start: Tensor
    recall_accent_phrase_end: Tensor
    value: Tensor
    data_num: int


class Evaluator(nn.Module):
    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    def forward(self, data: DatasetOutput) -> EvaluatorOutput:
        output_list: list[GeneratorOutput] = self.generator(
            vowel_list=data["vowel"],
            feature_list=data["feature"],
            mora_index_list=data["mora_index"],
        )

        accent_start = torch.cat([o["accent_start"] for o in output_list])
        accent_end = torch.cat([o["accent_end"] for o in output_list])
        accent_phrase_start = torch.cat([o["accent_phrase_start"] for o in output_list])
        accent_phrase_end = torch.cat([o["accent_phrase_end"] for o in output_list])

        target_accent = torch.cat(data["accent"])

        precision_accent_start, recall_accent_start = calc(
            accent_start, target_accent[:, 0]
        )
        precision_accent_end, recall_accent_end = calc(accent_end, target_accent[:, 1])
        precision_accent_phrase_start, recall_accent_phrase_start = calc(
            accent_phrase_start, target_accent[:, 2]
        )
        precision_accent_phrase_end, recall_accent_phrase_end = calc(
            accent_phrase_end, target_accent[:, 3]
        )

        value = (
            precision_accent_start
            + precision_accent_end
            + precision_accent_phrase_start
            + precision_accent_phrase_end
            + recall_accent_start
            + recall_accent_end
            + recall_accent_phrase_start
            + recall_accent_phrase_end
        ) / 8

        return EvaluatorOutput(
            precision_accent_start=precision_accent_start,
            precision_accent_end=precision_accent_end,
            precision_accent_phrase_start=precision_accent_phrase_start,
            precision_accent_phrase_end=precision_accent_phrase_end,
            recall_accent_start=recall_accent_start,
            recall_accent_end=recall_accent_end,
            recall_accent_phrase_start=recall_accent_phrase_start,
            recall_accent_phrase_end=recall_accent_phrase_end,
            value=value,
            data_num=len(data),
        )
