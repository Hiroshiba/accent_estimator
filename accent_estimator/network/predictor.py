from typing import List

import torch
from accent_estimator.config import NetworkConfig
from espnet_pytorch_library.conformer.encoder import Encoder
from espnet_pytorch_library.nets_utils import make_non_pad_mask
from espnet_pytorch_library.tacotron2.decoder import Postnet
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence


class Predictor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        block_num: int,
        post_layer_num: int,
    ):
        super().__init__()

        self.pre = torch.nn.Linear(1, hidden_size)

        self.encoder = Encoder(
            idim=None,
            attention_dim=hidden_size,
            attention_heads=2,
            linear_units=hidden_size * 4,
            num_blocks=block_num,
            input_layer=None,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
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
        f0_list: List[Tensor],  # [(length, )]
    ):
        length_list = [t.shape[0] for t in f0_list]

        length = torch.tensor(length_list, device=f0_list[0].device)
        h = pad_sequence(f0_list, batch_first=True)  # (batch_size, length, ?)

        h = self.pre(h)

        mask = self._mask(length)
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        if self.postnet is not None:
            output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        else:
            output2 = output1

        return (
            [output1[i, :l] for i, l in enumerate(length_list)],
            [output2[i, :l] for i, l in enumerate(length_list)],
        )


def create_predictor(config: NetworkConfig):
    return Predictor(
        hidden_size=config.hidden_size,
        block_num=config.block_num,
        post_layer_num=config.post_layer_num,
    )
