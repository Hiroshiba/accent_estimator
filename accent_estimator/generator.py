from pathlib import Path
from typing import Any, List, Union

import numpy
import torch
from torch import Tensor, nn
from typing_extensions import TypedDict

from accent_estimator.config import Config
from accent_estimator.data.data import generate_position_array
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
        vowel_list: List[Union[numpy.ndarray, Tensor]],
        feature_list: List[Union[numpy.ndarray, Tensor]],
    ):
        def prepare_tensors(
            array_list: List[Union[numpy.ndarray, Tensor]]
        ) -> List[Tensor]:
            return [to_tensor(array).to(self.device) for array in array_list]

        vowel_list = prepare_tensors(vowel_list)
        feature_list = prepare_tensors(feature_list)

        mora_position_list = prepare_tensors(
            [
                generate_position_array(t.shape[0]).astype(numpy.float32)
                for t in vowel_list
            ]
        )
        frame_position_list = prepare_tensors(
            [
                generate_position_array(t.shape[0]).astype(numpy.float32)
                for t in feature_list
            ]
        )

        with torch.inference_mode():
            output_list: List[Tensor] = self.predictor(
                vowel_list=vowel_list,
                mora_position_list=mora_position_list,
                feature_list=feature_list,
                frame_position_list=frame_position_list,
            )

        return [
            GeneratorOutput(
                accent_start=output[:, :, 0],
                accent_end=output[:, :, 1],
                accent_phrase_start=output[:, :, 2],
                accent_phrase_end=output[:, :, 3],
            )
            for output in output_list
        ]
