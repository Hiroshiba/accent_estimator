from espnet_pytorch_library.transformer.embedding import PositionalEncoding
from torch import Tensor, nn


class IndexPositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()

        self.pe = PositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout_rate,
            max_len=max_len,
        )

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        self.pe.extend_pe(x)
        x = x * self.pe.xscale + self.pe.pe[:, index].squeeze(0)
        return x
