"""多層CNNモジュール"""

from torch import Tensor, nn


class Cnn(nn.Module):
    """多層CNN"""

    def __init__(
        self,
        hidden_size: int,
        layer_num: int,
        kernel_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if layer_num <= 0:
            raise ValueError("layer_numは1以上である必要があります")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_sizeは奇数である必要があります")

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                )
                for _ in range(layer_num)
            ]
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: Tensor,  # (B, T, ?)
        cond: Tensor | None,  # (B, T, ?)
        mask: Tensor,  # (B, 1, T)
    ) -> tuple[Tensor, Tensor]:
        """系列を多層CNNで変換する"""
        if cond is not None:
            raise ValueError("Cnnはcondに対応していません")

        h = x.transpose(1, 2)  # (B, ?, T)
        mask_float = mask.to(h.dtype)
        for i, conv in enumerate(self.convs):
            h = h * mask_float
            h = conv(h)  # (B, ?, T)
            if i + 1 < len(self.convs):
                h = self.dropout(self.activation(h))  # (B, ?, T)
        h = h * mask_float
        return h.transpose(1, 2), mask
