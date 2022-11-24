from typing import List

import torch
from espnet_pytorch_library.conformer.encoder import Encoder
from espnet_pytorch_library.nets_utils import make_non_pad_mask
from espnet_pytorch_library.tacotron2.decoder import Postnet
from espnet_pytorch_library.transformer.decoder import Decoder
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from accent_estimator.config import NetworkConfig
from accent_estimator.network.index_positional_encoder import IndexPositionalEncoder


class Predictor(nn.Module):
    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        hidden_size: int,
        encoder_block_num: int,
        attention_heads: int,
        decoder_block_num: int,
        post_layer_num: int,
    ):
        super().__init__()

        # encoder (frame level)
        self.encoder_phoneme_embedder = nn.Embedding(
            num_embeddings=phoneme_size + 1,  # with empty
            embedding_dim=phoneme_embedding_size,
        )

        self.pre_encoder = nn.Sequential(
            nn.Linear(1 + phoneme_embedding_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )
        self.mora_positional_encoder = IndexPositionalEncoder(
            d_model=hidden_size,
            dropout_rate=0.1,
        )

        self.encoder = Encoder(
            idim=1 + phoneme_embedding_size,
            attention_dim=hidden_size,
            attention_heads=attention_heads,
            linear_units=hidden_size * 4,
            num_blocks=encoder_block_num,
            input_layer=None,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            positionwise_layer_type="conv1d",
            positionwise_conv_kernel_size=3,
            macaron_style=True,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            activation_type="swish",
            use_cnn_module=True,
            cnn_module_kernel=31,
        )

        # decoder (mora level)
        self.decoder_phoneme_embedder = nn.Embedding(
            num_embeddings=phoneme_size + 1,  # with empty
            embedding_dim=phoneme_embedding_size,
            padding_idx=0,
        )

        self.decoder = Decoder(
            odim=1 + phoneme_embedding_size,
            attention_dim=hidden_size,
            attention_heads=attention_heads,
            linear_units=hidden_size * 4,
            num_blocks=decoder_block_num,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            self_attention_dropout_rate=0.1,
            src_attention_dropout_rate=0.1,
            input_layer="linear",
            use_output_layer=False,
        )

        self.post = torch.nn.Linear(hidden_size, 4)

        if post_layer_num > 0:
            self.postnet = Postnet(
                idim=4,
                odim=4,
                n_layers=post_layer_num,
                n_chans=hidden_size,
                n_filts=5,
                use_batch_norm=True,
                dropout_rate=0.5,
            )
        else:
            self.postnet = None

    def _mask(self, length: Tensor):
        x_masks = make_non_pad_mask(length).to(length.device)
        return x_masks.unsqueeze(-2)

    def forward(
        self,
        frame_f0_list: List[Tensor],  # [(fL, 1)]
        frame_phoneme_list: List[Tensor],  # [(fL, )]
        frame_mora_index_list: List[Tensor],  # [(fL, )]
        mora_f0_list: List[Tensor],  # [(mL, 1)]
        mora_vowel_list: List[Tensor],  # [(mL, )]
        mora_consonant_list: List[Tensor],  # [(mL, )]
    ):
        """
        B: batch size
        fL: frame length
        mL: mora length
        """
        device = frame_f0_list[0].device
        batch_size = len(frame_f0_list)

        frame_length_list = [t.shape[0] for t in frame_f0_list]
        fh = pad_sequence(frame_f0_list, batch_first=True)  # (B, fL, ?)
        fp = pad_sequence(frame_phoneme_list, batch_first=True)  # (B, fL)
        fp = self.encoder_phoneme_embedder(fp + 1)  # (B, fL, ?)
        fh = torch.cat((fh, fp), dim=2)  # (B, fL, ?)

        fh = self.pre_encoder(fh)  # (B, fL, ?)
        fmi = pad_sequence(frame_mora_index_list, batch_first=True)  # (B, fL)
        fh = self.mora_positional_encoder(fh, index=fmi)  # (B, fL, ?)

        frame_mask = self._mask(torch.tensor(frame_length_list, device=device))
        fh, _ = self.encoder(fh, frame_mask)

        mora_length_list = [t.shape[0] for t in mora_f0_list]
        mh = pad_sequence(mora_f0_list, batch_first=True)  # (B, mL, ?)
        mv = pad_sequence(mora_vowel_list, batch_first=True)  # (B, mL)
        mc = pad_sequence(mora_consonant_list, batch_first=True)  # (B, mL)
        mp = self.decoder_phoneme_embedder(mv + 1) + self.decoder_phoneme_embedder(
            mc + 1
        )  # (B, mL, ?)
        mh = torch.cat((mh, mp), dim=2)  # (B, mL, ?)

        mora_mask = self._mask(torch.tensor(mora_length_list, device=device))
        mh, _ = self.decoder(mh, mora_mask, fh, frame_mask)

        output1 = self.post(mh)
        if self.postnet is not None:
            output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        else:
            output2 = output1

        return (
            [output1[i, :l] for i, l in enumerate(mora_length_list)],
            [output2[i, :l] for i, l in enumerate(mora_length_list)],
        )

    def inference(
        self,
        frame_f0_list: List[Tensor],  # [(fL, 1)]
        frame_phoneme_list: List[Tensor],  # [(fL, 1)]
        frame_mora_index_list: List[Tensor],  # [(fL, )]
        mora_f0_list: List[Tensor],  # [(mL, 1)]
        mora_vowel_list: List[Tensor],  # [(mL, 1)]
        mora_consonant_list: List[Tensor],  # [(mL, 1)]
    ):
        _, h = self(
            frame_f0_list=frame_f0_list,
            frame_phoneme_list=frame_phoneme_list,
            frame_mora_index_list=frame_mora_index_list,
            mora_f0_list=mora_f0_list,
            mora_vowel_list=mora_vowel_list,
            mora_consonant_list=mora_consonant_list,
        )
        return h


def create_predictor(config: NetworkConfig):
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        hidden_size=config.hidden_size,
        encoder_block_num=config.encoder_block_num,
        attention_heads=config.attention_heads,
        decoder_block_num=config.decoder_block_num,
        post_layer_num=config.post_layer_num,
    )
