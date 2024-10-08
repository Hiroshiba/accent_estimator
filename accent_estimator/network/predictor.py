import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from ..config import NetworkConfig
from ..data.data import vowels
from ..network.conformer.encoder import MMEncoder
from ..network.transformer.utility import make_non_pad_mask


class Predictor(nn.Module):
    def __init__(
        self,
        vowel_size: int,
        vowel_embedding_size: int,
        frame_reduction_factor: int,
        feature_size: int,
        hidden_size: int,
        encoder: MMEncoder,
    ):
        super().__init__()

        self.vowel_embedder = nn.Embedding(
            num_embeddings=vowel_size, embedding_dim=vowel_embedding_size
        )
        self.pre_mora = nn.Linear(feature_size + vowel_embedding_size, hidden_size)

        self.pre_frame = nn.Linear(feature_size, hidden_size)
        self.frame_reduction_factor = frame_reduction_factor

        self.encoder = encoder

        output_size = 4 * 2
        self.post = torch.nn.Linear(hidden_size, output_size)

    def aggregate_feature(
        self,
        feature_list: list[Tensor],  # [(fL, ?)]
        frame_length: Tensor,  # (B,)
        mora_index_list: list[Tensor],  # [(fL, )]
        mora_length: Tensor,  # (B,)
    ) -> list[Tensor]:  # [(mL, ?)]
        """モーラごとに特徴量を集約"""
        device = feature_list[0].device
        num_feature = feature_list[0].shape[-1]

        feature_concat = torch.cat(feature_list, dim=0)  # (sum(fL), F)

        offsets = torch.cat(
            [torch.tensor([0], device=device), torch.cumsum(mora_length[:-1], dim=0)]
        )
        offsets_expanded = torch.repeat_interleave(offsets, frame_length)
        mora_index_concat = torch.cat(mora_index_list) + offsets_expanded

        x_concat = torch.zeros(
            mora_length.sum(), num_feature, device=device
        )  # (sum(mL), F)
        x_concat = x_concat.scatter_reduce(
            0,
            mora_index_concat.unsqueeze(-1).expand(-1, num_feature),
            feature_concat,
            reduce="mean",
            include_self=False,
        )

        x_list = []
        start = 0
        for length in mora_length:
            length_int = int(length.item())
            end = start + length_int
            x_list.append(x_concat[start:end])
            start = end
        return x_list

    def forward(
        self,
        vowel_list: list[Tensor],  # [(mL, )]
        feature_list: list[Tensor],  # [(fL, ?)]
        mora_index_list: list[Tensor],  # [(fL, )]
    ):
        """
        B: batch size
        mL: mora length
        fL: frame length
        """
        device = feature_list[0].device

        # モーラごとに特徴量を集約
        mora_length = torch.tensor([t.shape[0] for t in vowel_list], device=device)
        frame_length = torch.tensor([t.shape[0] for t in feature_list], device=device)
        mora_feature_list = self.aggregate_feature(
            feature_list=feature_list,
            frame_length=frame_length,
            mora_index_list=mora_index_list,
            mora_length=mora_length,
        )  # [(mL, ?)]

        # モーラレベル
        mh = pad_sequence(mora_feature_list, batch_first=True)  # (B, mL, ?)
        mv = pad_sequence(vowel_list, batch_first=True)  # (B, mL)
        mv = self.vowel_embedder(mv)  # (B, mL, ?)
        mh = torch.cat((mv, mh), dim=2)  # (B, mL, ?)

        mh = self.pre_mora(mh)  # (B, mL, ?)

        mora_mask = (
            make_non_pad_mask(mora_length).unsqueeze(-2).to(mh.device)
        )  # (B, mL, 1)

        # フレームレベル
        fh = pad_sequence(feature_list, batch_first=True)  # (B, fL, ?)

        fh = self.pre_frame(fh)  # (B, fL, ?)
        fh = F.max_pool1d(
            fh.transpose(1, 2),
            kernel_size=self.frame_reduction_factor,
            stride=self.frame_reduction_factor,
        ).transpose(
            1, 2
        )  # (B, fLr, ?)

        frame_mask = (
            make_non_pad_mask(frame_length // self.frame_reduction_factor)
            .unsqueeze(-2)
            .to(fh.device)
        )  # (B, fLr, 1)

        # Encoder
        mh, _, _, _ = self.encoder(
            x_a=mh, x_b=fh, mask_a=mora_mask, mask_b=frame_mask
        )  # (B, mL, ?)

        # Post
        output = self.post(mh)  # (B, mL, ?)
        return [
            output[i, :l].reshape(l, 2, 4)  # (mL, 2, 4)
            for i, l in enumerate(mora_length)
        ]


def create_predictor(config: NetworkConfig):
    config.mm_conformer_config.hidden_size = config.hidden_size
    config.mm_conformer_config.feed_forward_hidden_size = config.hidden_size * 4
    encoder = MMEncoder(config.mm_conformer_config)
    return Predictor(
        vowel_size=len(vowels),
        vowel_embedding_size=config.vowel_embedding_size,
        frame_reduction_factor=config.frame_reduction_factor,
        feature_size=config.feature_size,
        hidden_size=config.hidden_size,
        encoder=encoder,
    )
