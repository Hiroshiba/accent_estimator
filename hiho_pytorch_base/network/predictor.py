"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import NetworkConfig
from ..data.data import vowels
from .conformer.encoder import Encoder
from .transformer.utility import make_non_pad_mask


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        vowel_size: int,
        vowel_embedding_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        frame_reduction_factor: int,
        feature_size: int,
        hidden_size: int,
        encoder: Encoder,
    ):
        super().__init__()

        self.vowel_embedder = nn.Embedding(vowel_size, vowel_embedding_size)
        self.speaker_embedder = nn.Embedding(speaker_size, speaker_embedding_size)
        self.frame_reduction_factor = frame_reduction_factor

        self.pre_mora = nn.Linear(
            feature_size + vowel_embedding_size + speaker_embedding_size, hidden_size
        )
        self.encoder = encoder
        self.post = nn.Linear(hidden_size, 4 * 2)

    def forward(  # noqa: D102
        self,
        *,
        vowel: Tensor,  # (B, max(mL))
        feature: Tensor,  # (B, max(fL), ?)
        mora_index: Tensor,  # (B, max(fL))
        speaker_id: Tensor,  # (B,)
        mora_length: Tensor,  # (B,)
        frame_length: Tensor,  # (B,)
    ) -> Tensor:  # (B, max(mL), 2, 4)
        if self.frame_reduction_factor > 1:
            reduced_feature = F.avg_pool1d(
                feature.transpose(1, 2),
                kernel_size=self.frame_reduction_factor,
                stride=self.frame_reduction_factor,
            ).transpose(1, 2)  # (B, max(fL)//r, ?)
            reduced_frame_length = frame_length // self.frame_reduction_factor
            reduced_mora_index = (
                mora_index[:, : reduced_feature.size(1)] // self.frame_reduction_factor
            )
        else:
            reduced_feature = feature
            reduced_frame_length = frame_length
            reduced_mora_index = mora_index

        mora_feature = self._aggregate_to_mora(
            feature=reduced_feature,
            frame_length=reduced_frame_length,
            mora_index=reduced_mora_index,
            mora_length=mora_length,
        )  # (B, max(mL), ?)

        max_mora_length = mora_feature.size(1)
        vowel_embed = self.vowel_embedder(vowel[:, :max_mora_length])  # (B, max(mL), ?)
        speaker_embed = (
            self.speaker_embedder(speaker_id)
            .unsqueeze(1)
            .expand(-1, max_mora_length, -1)
        )  # (B, max(mL), ?)

        h = torch.cat(
            [mora_feature, vowel_embed, speaker_embed], dim=2
        )  # (B, max(mL), ?)
        h = self.pre_mora(h)  # (B, max(mL), ?)

        mora_mask = (
            make_non_pad_mask(mora_length).unsqueeze(-2).to(h.device)
        )  # (B, 1, max(mL))
        h, _ = self.encoder(x=h, cond=None, mask=mora_mask)  # (B, max(mL), ?)

        output = self.post(h)  # (B, max(mL), 4*2)
        return output.reshape(output.size(0), output.size(1), 2, 4)

    def _aggregate_to_mora(
        self,
        feature: Tensor,  # (B, max(fL), ?)
        frame_length: Tensor,  # (B,)
        mora_index: Tensor,  # (B, max(fL))
        mora_length: Tensor,  # (B,)
    ) -> Tensor:  # (B, max(mL), ?)
        """フレーム特徴をモーラごとに平均集約する"""
        batch_size, _, num_feature = feature.shape
        max_mora_length = int(mora_length.max().item())
        device = feature.device

        frame_mask = make_non_pad_mask(frame_length).to(device)  # (B, max(fL))
        masked_index = torch.where(
            frame_mask, mora_index, torch.full_like(mora_index, max_mora_length)
        )  # (B, max(fL))

        x = feature.new_zeros(
            batch_size, max_mora_length + 1, num_feature
        )  # (B, max(mL)+1, ?)
        x = x.scatter_reduce(
            1,
            masked_index.unsqueeze(-1).expand(-1, -1, num_feature),
            feature,
            reduce="mean",
            include_self=False,
        )
        return x[:, :max_mora_length]


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.conformer_block_num,
        dropout_rate=config.conformer_dropout_rate,
        positional_dropout_rate=config.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.conformer_dropout_rate,
        use_macaron_style=True,
        use_conv_glu_module=True,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        vowel_size=len(vowels),
        vowel_embedding_size=config.vowel_embedding_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        frame_reduction_factor=config.frame_reduction_factor,
        feature_size=config.feature_size,
        hidden_size=config.hidden_size,
        encoder=encoder,
    )
