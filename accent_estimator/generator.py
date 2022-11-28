from pathlib import Path
from typing import Any, List, Optional, Union

import numpy
import torch
from torch import Tensor, nn
from typing_extensions import TypedDict

from accent_estimator.config import Config
from accent_estimator.network.predictor import Predictor, create_predictor


class GeneratorOutput(TypedDict):
    accent_start: Tensor
    accent_end: Tensor
    accent_phrase_start: Tensor
    accent_phrase_end: Tensor


def to_tensor(array: Union[Tensor, numpy.ndarray, Any]):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


class Generator(nn.Module):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def forward(
        self,
        frame_f0_list: List[Union[numpy.ndarray, Tensor]],
        frame_phoneme_list: List[Union[numpy.ndarray, Tensor]],
        frame_mora_index_list: List[Union[numpy.ndarray, Tensor]],
        mora_f0_list: List[Union[numpy.ndarray, Tensor]],
        mora_vowel_list: List[Union[numpy.ndarray, Tensor]],
        mora_consonant_list: List[Union[numpy.ndarray, Tensor]],
    ):
        frame_f0_list = [to_tensor(v).to(self.device) for v in frame_f0_list]
        frame_phoneme_list = [to_tensor(v).to(self.device) for v in frame_phoneme_list]
        frame_mora_index_list = [
            to_tensor(v).to(self.device) for v in frame_mora_index_list
        ]
        mora_f0_list = [to_tensor(v).to(self.device) for v in mora_f0_list]
        mora_vowel_list = [to_tensor(v).to(self.device) for v in mora_vowel_list]
        mora_consonant_list = [
            to_tensor(v).to(self.device) for v in mora_consonant_list
        ]

        with torch.inference_mode():
            if self.config.model.disable_mora_f0:
                mora_f0_list = [torch.zeros_like(v) for v in mora_f0_list]

            output_list = self.predictor.inference(
                frame_f0_list=frame_f0_list,
                frame_phoneme_list=frame_phoneme_list,
                frame_mora_index_list=frame_mora_index_list,
                mora_f0_list=mora_f0_list,
                mora_vowel_list=mora_vowel_list,
                mora_consonant_list=mora_consonant_list,
            )

        return [
            GeneratorOutput(
                accent_start=output[:, 0],
                accent_end=output[:, 1],
                accent_phrase_start=output[:, 2],
                accent_phrase_end=output[:, 3],
            )
            for output in output_list
        ]
