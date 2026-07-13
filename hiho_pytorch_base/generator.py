"""学習済みモデルからの推論モジュール"""

from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from torch import Tensor, nn

from .config import Config
from .data.statistics import DataStatistics
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
        self.use_diffusion = config.model.use_diffusion
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            statistics = DataStatistics(
                accent_mean=state_dict["accent_mean"].cpu().numpy(),
                accent_std=state_dict["accent_std"].cpu().numpy(),
            )
            predictor = create_predictor(config.network, statistics=statistics)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def denormalize(self, accent: Tensor) -> Tensor:  # (..., 2, 4)
        """正規化されたアクセントone-hotを非正規化する"""
        accent_mean: Tensor = self.predictor.accent_mean  # type: ignore
        accent_std: Tensor = self.predictor.accent_std  # type: ignore
        return accent * accent_std + accent_mean

    @torch.no_grad()
    def forward(
        self,
        *,
        wave: TensorLike,  # (B, max(wL))
        phoneme_index: TensorLike,  # (B, max(fL))
        phoneme_id: TensorLike,  # (B, max(pL))
        vowel_index: TensorLike,  # (B, max(mL))
        mora_f0: TensorLike,  # (B, max(mL))
        accent_noise: TensorLike,  # (B, max(mL), 2, 4)
        wave_length: TensorLike,  # (B,)
        phoneme_length: TensorLike,  # (B,)
        mora_length: TensorLike,  # (B,)
    ) -> GeneratorOutput:
        """生成経路で推論する"""
        wave_t = to_tensor(wave, self.device)
        phoneme_index_t = to_tensor(phoneme_index, self.device)
        phoneme_id_t = to_tensor(phoneme_id, self.device)
        vowel_index_t = to_tensor(vowel_index, self.device)
        mora_f0_t = to_tensor(mora_f0, self.device)
        wave_length_t = to_tensor(wave_length, self.device)
        phoneme_length_t = to_tensor(phoneme_length, self.device)
        mora_length_t = to_tensor(mora_length, self.device)

        if self.use_diffusion:
            accent_logit = self._sample_diffusion(
                wave=wave_t,
                phoneme_index=phoneme_index_t,
                phoneme_id=phoneme_id_t,
                vowel_index=vowel_index_t,
                mora_f0=mora_f0_t,
                accent_noise=to_tensor(accent_noise, self.device),
                wave_length=wave_length_t,
                phoneme_length=phoneme_length_t,
                mora_length=mora_length_t,
            )
        else:
            batch_size, max_mora_length = vowel_index_t.shape
            # use_diffusion=False時はPredictorに使われないため0埋め
            zero_accent_input = torch.zeros(
                batch_size, max_mora_length, 2, 4, device=self.device
            )
            zero_t = torch.zeros(batch_size, device=self.device)
            accent_logit = self.predictor(
                wave=wave_t,
                phoneme_index=phoneme_index_t,
                phoneme_id=phoneme_id_t,
                vowel_index=vowel_index_t,
                mora_f0=mora_f0_t,
                wave_length=wave_length_t,
                phoneme_length=phoneme_length_t,
                mora_length=mora_length_t,
                accent_input=zero_accent_input,
                t=zero_t,
            )

        return GeneratorOutput(
            accent_logit=accent_logit,
            mora_length=mora_length_t,
        )

    def _sample_diffusion(
        self,
        *,
        wave: Tensor,  # (B, max(wL))
        phoneme_index: Tensor,  # (B, max(fL))
        phoneme_id: Tensor,  # (B, max(pL))
        vowel_index: Tensor,  # (B, max(mL))
        mora_f0: Tensor,  # (B, max(mL))
        accent_noise: Tensor,  # (B, max(mL), 2, 4)
        wave_length: Tensor,  # (B,)
        phoneme_length: Tensor,  # (B,)
        mora_length: Tensor,  # (B,)
    ) -> Tensor:  # (B, max(mL), 2, 4)
        """Rectified Flowの逆過程でアクセントone-hotをサンプリングし非正規化する"""
        step_num = self.config.train.diffusion_step_num
        t_array = torch.linspace(0, 1, steps=step_num + 1, device=self.device)[:-1]
        delta_t = 1.0 / step_num

        accent_input = accent_noise.clone()
        for i in range(step_num):
            t = t_array[i].expand(accent_input.size(0))
            velocity = self.predictor(
                wave=wave,
                phoneme_index=phoneme_index,
                phoneme_id=phoneme_id,
                vowel_index=vowel_index,
                mora_f0=mora_f0,
                wave_length=wave_length,
                phoneme_length=phoneme_length,
                mora_length=mora_length,
                accent_input=accent_input,
                t=t,
            )  # (B, max(mL), 2, 4)
            accent_input = accent_input + velocity * delta_t

        return self.denormalize(accent_input)
