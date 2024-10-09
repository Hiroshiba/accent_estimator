from torch import Tensor, nn

from .transformer.embedding import PositionalEncoding


class IndexPositionalEncoder(nn.Module):
    def __init__(
        self, hidden_size: int, dropout_rate: float, cycle_length: int = 10000
    ):
        super().__init__()

        self.pe = PositionalEncoding(
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            cycle_length=cycle_length,
        )

    def forward(self, x: Tensor, index: Tensor) -> Tensor:
        self.pe.extend_pe(x)
        x = x * self.pe.xscale + self.pe.pe[:, index].squeeze(0)
        return x
