# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm

from ..transformer.attention import MultiHeadedAttention


class OneEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        conv_module: Optional[nn.Module],
        feed_forward: nn.Module,
        dropout_rate: float,
    ):
        super().__init__()
        self.conv_module = conv_module
        self.feed_forward = feed_forward
        self.norm_ff = LayerNorm(hidden_size, eps=1e-12)
        self.norm_mha = LayerNorm(hidden_size, eps=1e-12)
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(hidden_size, eps=1e-12)
            self.norm_final = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward_pre(
        self,
        x: Tensor,  # (B, T, ?)
    ):
        """アテンション前までの処理"""
        residual = x
        x = self.norm_mha(x)

        return x, residual

    def forward_post(
        self,
        x: Tensor,  # (B, T, ?)
        residual: Tensor,  # (B, T, ?)
    ):
        """アテンション後からの処理"""
        x = residual + self.dropout(x)

        if self.conv_module is not None:
            residual = x
            x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))

        residual = x  # FIXME: 再代入なくして最初のxを足す形でも良いかも
        x = self.norm_ff(x)
        x = residual + self.dropout(self.feed_forward(x))

        return x


class MMEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        self_attn: MultiHeadedAttention,
        conv_module_a: Optional[nn.Module],
        conv_module_b: Optional[nn.Module],
        feed_forward_a: nn.Module,
        feed_forward_b: nn.Module,
        dropout_rate: float,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.one_encoder_layer_a = OneEncoderLayer(
            hidden_size=hidden_size,
            conv_module=conv_module_a,
            feed_forward=feed_forward_a,
            dropout_rate=dropout_rate,
        )
        self.one_encoder_layer_b = OneEncoderLayer(
            hidden_size=hidden_size,
            conv_module=conv_module_b,
            feed_forward=feed_forward_b,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        x_a: Tensor,  # (B, Ta, ?)
        x_b: Tensor,  # (B, Tb, ?)
        pos_emb_a: Tensor,  # (1, Ta, ?)
        pos_emb_b: Tensor,  # (1, Tb, ?)
        mask_a: Tensor,  # (B, 1, Ta)
        mask_b: Tensor,  # (B, 1, Tb)
    ):
        x_a, residual_a = self.one_encoder_layer_a.forward_pre(x_a)
        x_b, residual_b = self.one_encoder_layer_b.forward_pre(x_b)

        # concat、attention、split
        x = torch.cat((x_a, x_b), dim=1)
        pos_emb = torch.cat((pos_emb_a, pos_emb_b), dim=1)
        mask = torch.cat((mask_a, mask_b), dim=2)
        x_att = self.self_attn(x, x, x, pos_emb, mask)
        x_a = x_att[:, : x_a.size(1)]
        x_b = x_att[:, x_a.size(1) :]

        x_a = self.one_encoder_layer_a.forward_post(x_a, residual_a)
        x_b = self.one_encoder_layer_b.forward_post(x_b, residual_b)

        return x_a, x_b, pos_emb_a, pos_emb_b, mask_a, mask_b
