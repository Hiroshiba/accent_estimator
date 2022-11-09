from typing import List

import torch
from torch import Tensor, nn
from typing_extensions import TypedDict

from accent_estimator.dataset import OutputData
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

    def forward(self, data: OutputData) -> EvaluatorOutput:
        device = data["mora_f0"][0].device

        output_list: List[GeneratorOutput] = self.generator(
            f0_list=data["mora_f0"],
        )

        target_accent_start = torch.cat(data["accent_start"])
        target_accent_end = torch.cat(data["accent_end"])
        target_accent_phrase_start = torch.cat(data["accent_phrase_start"])
        target_accent_phrase_end = torch.cat(data["accent_phrase_end"])

        output_accent_start = torch.cat(
            [output["accent_start"] for output in output_list]
        ).to(device)
        output_accent_end = torch.cat(
            [output["accent_end"] for output in output_list]
        ).to(device)
        output_accent_phrase_start = torch.cat(
            [output["accent_phrase_start"] for output in output_list]
        ).to(device)
        output_accent_phrase_end = torch.cat(
            [output["accent_phrase_end"] for output in output_list]
        ).to(device)

        _, precision_accent_start, recall_accent_start = calc(
            output_accent_start, target_accent_start
        )
        _, precision_accent_end, recall_accent_end = calc(
            output_accent_end, target_accent_end
        )
        _, precision_accent_phrase_start, recall_accent_phrase_start = calc(
            output_accent_phrase_start, target_accent_phrase_start
        )
        _, precision_accent_phrase_end, recall_accent_phrase_end = calc(
            output_accent_phrase_end, target_accent_phrase_end
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
