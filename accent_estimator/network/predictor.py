from typing import List

import torch
from torch import Tensor, nn
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
        feature_size: int,
        position_size: int,
        hidden_size: int,
        encoder: MMEncoder,
    ):
        super().__init__()

        self.vowel_embedder = nn.Embedding(
            num_embeddings=vowel_size, embedding_dim=vowel_embedding_size
        )
        self.pre_mora = nn.Linear(vowel_embedding_size + position_size, hidden_size)

        self.pre_frame = nn.Linear(feature_size + position_size, hidden_size)

        self.encoder = encoder

        output_size = 4 * 2
        self.post = torch.nn.Linear(hidden_size, output_size)

    def forward(
        self,
        vowel_list: List[Tensor],  # [(mL, )]
        mora_position_list: List[Tensor],  # [(mL, ?)]
        feature_list: List[Tensor],  # [(fL, ?)]
        frame_position_list: List[Tensor],  # [(fL, ?)]
    ):
        """
        B: batch size
        mL: mora length
        fL: frame length
        """
        # モーラレベル
        mora_length_list = [t.shape[0] for t in mora_position_list]

        mh = pad_sequence(vowel_list, batch_first=True)  # (B, mL)
        mp = pad_sequence(mora_position_list, batch_first=True)  # (B, mL, ?)

        mh = self.vowel_embedder(mh)  # (B, mL, ?)
        mh = torch.cat((mh, mp), dim=2)  # (B, mL, ?)
        mh = self.pre_mora(mh)  # (B, mL, ?)

        mora_mask = make_non_pad_mask(mora_length_list).unsqueeze(-2).to(mh.device)

        # フレームレベル
        frame_length_list = [t.shape[0] for t in frame_position_list]
        fh = pad_sequence(feature_list, batch_first=True)  # (B, fL, ?)
        fp = pad_sequence(frame_position_list, batch_first=True)  # (B, fL, ?)

        fh = torch.cat((fh, fp), dim=2)  # (B, fL, ?)
        fh = self.pre_frame(fh)

        frame_mask = make_non_pad_mask(frame_length_list).unsqueeze(-2).to(fh.device)

        # Encoder
        mh, _, _, _ = self.encoder(
            x_a=mh, x_b=fh, mask_a=mora_mask, mask_b=frame_mask
        )  # (B, mL, ?)

        # Post
        output = self.post(mh)  # (B, mL, ?)
        return [
            output[i, :l].reshape(l, 2, 4)  # (mL, 2, 4)
            for i, l in enumerate(mora_length_list)
        ]


def create_predictor(config: NetworkConfig):
    config.mm_conformer_config.hidden_size = config.hidden_size
    config.mm_conformer_config.feed_forward_hidden_size = config.hidden_size * 4
    encoder = MMEncoder(config.mm_conformer_config)
    return Predictor(
        vowel_size=len(vowels),
        vowel_embedding_size=config.vowel_embedding_size,
        feature_size=config.feature_size,
        position_size=2,
        hidden_size=config.hidden_size,
        encoder=encoder,
    )
