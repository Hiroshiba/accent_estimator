# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
from typing import Optional

import torch
from torch import Tensor, nn


class MultiHeadedAttention(nn.Module):
    """
    Td: デコーダー側の系列長
    Te: エンコーダー側の系列長
    """

    def __init__(self, head_size: int, hidden_size: int, dropout_rate: float):
        super().__init__()

        assert hidden_size % head_size == 0
        self.partial_hidden_size = hidden_size // head_size
        self.head_size = head_size
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self,
        query: Tensor,  # (B, Td, ?)
        key: Tensor,  # (B, Te, ?)
        value: Tensor,  # (B, Te, ?)
    ):
        batch_size = query.size(0)
        q = self.linear_q(query).view(
            batch_size, -1, self.head_size, self.partial_hidden_size
        )
        k = self.linear_k(key).view(
            batch_size, -1, self.head_size, self.partial_hidden_size
        )
        v = self.linear_v(value).view(
            batch_size, -1, self.head_size, self.partial_hidden_size
        )
        q = q.transpose(1, 2)  # (B, H, Td, ?)
        k = k.transpose(1, 2)  # (B, H, Te, ?)
        v = v.transpose(1, 2)  # (B, H, Te, ?)
        return q, k, v

    def forward_attention(
        self,
        value: Tensor,  # (B, H, Te, ?)
        score: Tensor,  # (B, H, Td, Te)
        mask: Optional[Tensor],  # (B, Td, Te) or (B, 1, Te)
    ):
        batch_size = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (B, 1, Td or 1, Te)
            min_value = torch.finfo(score.dtype).min
            score = score.masked_fill(mask, min_value)
            attn = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)  # (B, H, Td, Te)
        else:
            attn = torch.softmax(score, dim=-1)  # (B, H, Td, Te)

        x = torch.matmul(self.dropout(attn), value)  # (B, H, Td, ?)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.head_size * self.partial_hidden_size)
        )  # (B, Td, ?)

        return self.linear_out(x)  # (B, Td, ?)

    def forward(
        self,
        query: Tensor,  # (B, Td, ?)
        key: Tensor,  # (B, Te, ?)
        value: Tensor,  # (B, Te, ?)
        mask: Optional[Tensor],  # (B, Td, Te) or (B, 1, Te)
    ):
        query, key, value = self.forward_qkv(query, key, value)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.partial_hidden_size
        )  # (B, H, Td, Te)
        return self.forward_attention(value, score, mask)  # (B, Td, ?)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """
    https://arxiv.org/abs/1901.02860 Section 3.3
    TODO: Self-attention 専用？
    """

    def __init__(self, head_size: int, hidden_size: int, dropout_rate: float):
        super().__init__(
            head_size=head_size, hidden_size=hidden_size, dropout_rate=dropout_rate
        )

        self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_k = nn.Parameter(
            torch.Tensor(self.head_size, self.partial_hidden_size)
        )
        self.bias_p = nn.Parameter(
            torch.Tensor(self.head_size, self.partial_hidden_size)
        )
        torch.nn.init.xavier_uniform_(self.bias_k)
        torch.nn.init.xavier_uniform_(self.bias_p)

    def rel_shift(
        self,
        x: Tensor,  # (B, H, Td, 2*Td-1)
    ):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]
        return x  # (B, H, Td, Td)

    def forward(
        self,
        query: Tensor,  # (B, Td, ?)
        key: Tensor,  # (B, Te, ?)
        value: Tensor,  # (B, Te, ?)
        pos_emb: Tensor,  # (B, 2*Td-1, ?)
        mask: Optional[Tensor],  # (B, Td, Te) or (B, 1, Te)
    ):
        query, key, value = self.forward_qkv(query, key, value)
        query = query.transpose(1, 2)  # (B, Td, H, ?)

        batch_size = pos_emb.size(0)
        pos_emb = self.linear_pos(pos_emb).view(
            batch_size, -1, self.head_size, self.partial_hidden_size
        )
        pos_emb = pos_emb.transpose(1, 2)  # (B, H, 2*Td-1, ?)

        query_k = (query + self.bias_k).transpose(1, 2)  # (B, H, Td, ?)
        query_p = (query + self.bias_p).transpose(1, 2)  # (B, H, Td, ?)

        score_k = torch.matmul(query_k, key.transpose(-2, -1))  # (B, H, Td, Te)

        score_p = torch.matmul(query_p, pos_emb.transpose(-2, -1))  # (B, H, Td, 2*Td-1)
        score_p = self.rel_shift(score_p)  # (B, H, Td, Td)

        score = (score_k + score_p) / math.sqrt(
            self.partial_hidden_size
        )  # (B, H, Td, Te)

        return self.forward_attention(value, score, mask)
