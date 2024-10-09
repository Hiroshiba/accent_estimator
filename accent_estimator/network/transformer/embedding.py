# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(
        self, hidden_size: int, dropout_rate: float, cycle_length: float = 10000
    ):
        super(PositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.xscale = math.sqrt(self.hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cycle_length = cycle_length
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 5000))

    def extend_pe(self, x: Tensor):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.hidden_size)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, dtype=torch.float32)
            * -(math.log(self.cycle_length) / self.hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RelPositionalEncoding(nn.Module):
    def __init__(
        self, hidden_size: int, dropout_rate: float, cycle_length: float = 10000
    ):
        super(RelPositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.xscale = math.sqrt(self.hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cycle_length = cycle_length
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 5000))

    def extend_pe(self, x: Tensor):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.hidden_size)
        pe_negative = torch.zeros(x.size(1), self.hidden_size)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, dtype=torch.float32)
            * -(math.log(self.cycle_length) / self.hidden_size)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(
        self,
        x: Tensor,  # (B, T, ?)
    ):
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)
