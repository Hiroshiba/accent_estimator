# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
from torch import Tensor, nn

from accent_estimator.config import MMConformerConfig

from ..transformer.attention import RelPositionMultiHeadedAttention
from ..transformer.embedding import RelPositionalEncoding
from ..transformer.multi_layer_conv import FastSpeechTwoConv
from .convolution import ConvGLUModule
from .encoder_layer import MMEncoderLayer


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MMEncoder(nn.Module):
    def __init__(self, config: MMConformerConfig):
        super().__init__()

        hidden_size = config.hidden_size
        block_num = config.block_num
        dropout_rate = config.dropout_rate
        positional_dropout_rate = config.positional_dropout_rate
        attention_head_size = config.attention_head_size
        attention_dropout_rate = config.attention_dropout_rate
        use_conv_glu_module = config.use_conv_glu_module
        conv_glu_module_kernel_size_a = config.conv_glu_module_kernel_size_a
        conv_glu_module_kernel_size_b = config.conv_glu_module_kernel_size_b
        feed_forward_hidden_size = config.feed_forward_hidden_size
        feed_forward_kernel_size_a = config.feed_forward_kernel_size_a
        feed_forward_kernel_size_b = config.feed_forward_kernel_size_b

        self.embed = RelPositionalEncoding(hidden_size, positional_dropout_rate)

        self.encoders = nn.ModuleList(
            MMEncoderLayer(
                hidden_size=hidden_size,
                self_attn=RelPositionMultiHeadedAttention(
                    head_size=attention_head_size,
                    hidden_size=hidden_size,
                    dropout_rate=attention_dropout_rate,
                ),
                conv_module_a=(
                    ConvGLUModule(
                        hidden_size=hidden_size,
                        kernel_size=conv_glu_module_kernel_size_a,
                        activation=Swish(),
                    )
                    if use_conv_glu_module
                    else None
                ),
                conv_module_b=(
                    ConvGLUModule(
                        hidden_size=hidden_size,
                        kernel_size=conv_glu_module_kernel_size_b,
                        activation=Swish(),
                    )
                    if use_conv_glu_module
                    else None
                ),
                feed_forward_a=FastSpeechTwoConv(
                    inout_size=hidden_size,
                    hidden_size=feed_forward_hidden_size,
                    kernel_size=feed_forward_kernel_size_a,
                    dropout_rate=dropout_rate,
                ),
                feed_forward_b=FastSpeechTwoConv(
                    inout_size=hidden_size,
                    hidden_size=feed_forward_hidden_size,
                    kernel_size=feed_forward_kernel_size_b,
                    dropout_rate=dropout_rate,
                ),
                dropout_rate=dropout_rate,
            )
            for _ in range(block_num)
        )
        self.after_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(
        self,
        x_a: Tensor,  # (B, Ta, ?)
        x_b: Tensor,  # (B, Tb, ?)
        mask_a: Tensor,  # (B, 1, Ta)
        mask_b: Tensor,  # (B, 1, Tb)
    ):
        x_a, pos_emb_a = self.embed(x_a)
        x_b, pos_emb_b = self.embed(x_b)
        for encoder in self.encoders:
            x_a, x_b, pos_emb_a, pos_emb_b, mask_a, mask_b = encoder(
                x_a=x_a,
                x_b=x_b,
                pos_emb_a=pos_emb_a,
                pos_emb_b=pos_emb_b,
                mask_a=mask_a,
                mask_b=mask_b,
            )
        x_a = self.after_norm(x_a)
        x_b = self.after_norm(x_b)
        return x_a, x_b, mask_a, mask_b
