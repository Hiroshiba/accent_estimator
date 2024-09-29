# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from torch import Tensor, nn


class FastSpeechTwoConv(nn.Module):
    """
    FastSpeechの2層Conv
    """

    def __init__(
        self,
        inout_size: int,
        hidden_size: int,
        kernel_size: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            inout_size,
            hidden_size,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv1d(
            hidden_size,
            inout_size,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: Tensor,  # (B, T, ?)
    ):
        x = x.transpose(1, 2)  # (B, ?, T)
        x = self.activation(self.conv1(x))  # (B, ?, T)
        x = self.conv2(self.dropout(x))  # (B, ?, T)
        x = x.transpose(1, 2)  # (B, T, ?)
        return x
