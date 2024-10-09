# Original Code Copyright ESPnet
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch


def make_pad_mask(length: torch.Tensor):
    maxlen = length.max()
    mask = torch.arange(maxlen, dtype=torch.int64, device=length.device).unsqueeze(
        0
    ) >= length.unsqueeze(1)
    return mask


def make_non_pad_mask(length: torch.Tensor):
    return ~make_pad_mask(length=length)
