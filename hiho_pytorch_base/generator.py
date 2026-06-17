"""学習済みモデルからの推論モジュール"""

from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from torch import Tensor, nn

from .config import Config
from .network.predictor import Predictor, create_predictor

TensorLike = Tensor | numpy.ndarray


@dataclass
class GeneratorOutput:
    """生成したデータ"""

    accent_logit: Tensor  # (B, max(mL), 2, 4)
    mora_length: Tensor  # (B,)


def to_tensor(array: TensorLike, device: torch.device) -> Tensor:
    """データをTensorに変換する"""
    if not isinstance(array, Tensor | numpy.ndarray):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    tensor = tensor.to(device)
    return tensor


class Generator(nn.Module):
    """生成経路で推論するクラス"""

    def __init__(
        self,
        config: Config,
        predictor: Predictor | Path,
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

    @torch.no_grad()
    def forward(
        self,
        *,
        vowel: TensorLike,  # (B, max(mL))
        wave: TensorLike,  # (B, max(wL))
        mora_index: TensorLike,  # (B, max(fL))
        speaker_id: TensorLike,  # (B,)
        wave_length: TensorLike,  # (B,)
        mora_length: TensorLike,  # (B,)
    ) -> GeneratorOutput:
        """生成経路で推論する"""
        mora_length_tensor = to_tensor(mora_length, self.device)
        accent_logit = self.predictor(  # (B, max(mL), 2, 4)
            vowel=to_tensor(vowel, self.device),
            wave=to_tensor(wave, self.device),
            mora_index=to_tensor(mora_index, self.device),
            speaker_id=to_tensor(speaker_id, self.device),
            wave_length=to_tensor(wave_length, self.device),
            mora_length=mora_length_tensor,
        )
        return GeneratorOutput(
            accent_logit=accent_logit,
            mora_length=mora_length_tensor,
        )
