"""学習済みモデルをONNX形式にエクスポートする"""

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.network.predictor import Predictor, create_predictor


class PredictorWrapper(nn.Module):
    """ONNXエクスポート用のPredictorラッパー"""

    # TODO: Predictor.forward は scatter_reduce(reduce='mean') を含むため
    # そのままでは ONNX に変換できない。ONNX 化方針の見直しが必要。

    def __init__(self, predictor: Predictor) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(  # noqa: D102
        self,
        vowel: Tensor,  # (B, max(mL))
        wave: Tensor,  # (B, max(wL))
        mora_index: Tensor,  # (B, max(fL))
        speaker_id: Tensor,  # (B,)
        wave_length: Tensor,  # (B,)
        mora_length: Tensor,  # (B,)
    ) -> Tensor:
        return self.predictor(
            vowel=vowel,
            wave=wave,
            mora_index=mora_index,
            speaker_id=speaker_id,
            wave_length=wave_length,
            mora_length=mora_length,
        )


def export_onnx(config_yaml_path: UPath, output_path: Path, verbose: bool) -> None:
    """設定からPredictorを作成してONNX形式でエクスポートする"""
    output_path.parent.mkdir(exist_ok=True, parents=True)

    config = Config.load(config_yaml_path)

    predictor = create_predictor(config.network)
    wrapper = PredictorWrapper(predictor)
    wrapper.eval()

    batch_size = 1
    max_mora_length = 10
    max_frame_length = max_mora_length * 5
    wave_length_samples = max_frame_length * 320  # HuBERT stride

    vowel = torch.randint(0, 8, (batch_size, max_mora_length))
    wave = torch.randn(batch_size, wave_length_samples)
    wave_length = torch.tensor([wave_length_samples] * batch_size)
    mora_index = (
        torch.repeat_interleave(
            torch.arange(max_mora_length),
            repeats=max_frame_length // max_mora_length,
        )[:max_frame_length]
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    speaker_id = torch.randint(0, config.network.speaker_size, (batch_size,))
    mora_length = torch.tensor([max_mora_length] * batch_size)

    torch.onnx.export(
        wrapper,
        (vowel, wave, mora_index, speaker_id, wave_length, mora_length),
        str(output_path),
        input_names=[
            "vowel",
            "wave",
            "mora_index",
            "speaker_id",
            "wave_length",
            "mora_length",
        ],
        output_names=["accent_logit"],
        dynamic_axes={
            "vowel": {0: "batch_size", 1: "max_mora_length"},
            "wave": {0: "batch_size", 1: "max_wave_length"},
            "mora_index": {0: "batch_size", 1: "max_frame_length"},
            "speaker_id": {0: "batch_size"},
            "wave_length": {0: "batch_size"},
            "mora_length": {0: "batch_size"},
            "accent_logit": {0: "batch_size", 1: "max_mora_length"},
        },
        verbose=verbose,
    )
    print(f"ONNX model exported to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=UPath)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    export_onnx(**vars(parser.parse_args()))
