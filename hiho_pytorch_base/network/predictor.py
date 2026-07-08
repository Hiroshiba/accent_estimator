"""メインのネットワークモジュール"""

import math
from typing import Any

import torch
from torch import Tensor, nn

from ..config import NetworkConfig
from ..data.phoneme import OjtPhoneme
from .conformer.encoder import Encoder
from .ssl_feature_models import HubertModel, create_base_hubert_config, load_ssl_model
from .transformer.utility import make_non_pad_mask

_SSL_MODEL_STATE_DICT_PREFIX = "ssl_model."


def _remove_ssl_model_state_dict(
    module: nn.Module,
    state_dict: dict[str, Tensor],
    prefix: str,
    local_metadata: Any,
) -> None:
    """state_dictからSSLモデルのキーを除く"""
    for key in list(state_dict.keys()):
        if key.startswith(prefix + _SSL_MODEL_STATE_DICT_PREFIX):
            del state_dict[key]


def _reject_ssl_model_state_dict(
    module: nn.Module,
    state_dict: dict[str, Tensor],
    prefix: str,
    local_metadata: Any,
    *args: Any,
) -> None:
    """読み込み時にSSLモデルのキーを拒否する"""
    ssl_model_keys = [
        key
        for key in state_dict.keys()
        if key.startswith(prefix + _SSL_MODEL_STATE_DICT_PREFIX)
    ]
    if len(ssl_model_keys) > 0:
        raise RuntimeError(
            f"Predictorのstate_dictにSSLモデルのキーが含まれています: {ssl_model_keys}"
        )


def _ignore_ssl_model_missing(module: nn.Module, incompatible_keys: Any) -> None:
    """読み込み時にSSLモデルのキー欠落を許容する"""
    for key in list(incompatible_keys.missing_keys):
        if key.startswith(_SSL_MODEL_STATE_DICT_PREFIX):
            incompatible_keys.missing_keys.remove(key)


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        ssl_model: HubertModel,
        sampling_rate: int,
        frame_rate: float,
        phoneme_size: int,
        phoneme_embedding_size: int,
        hidden_size: int,
        encoder: Encoder,
        use_f0: bool,
        use_phoneme: bool,
    ):
        super().__init__()

        self.ssl_model = ssl_model
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate
        self.use_f0 = use_f0
        self.use_phoneme = use_phoneme
        feature_size = ssl_model.hidden_size

        self.layer_weight = nn.Parameter(torch.zeros(ssl_model.num_hidden_layers))

        self.phoneme_embedder = nn.Embedding(phoneme_size, phoneme_embedding_size)

        input_size = feature_size
        if use_phoneme:
            input_size += phoneme_embedding_size
        if use_f0:
            input_size += 1

        self.pre_phoneme = nn.Linear(input_size, hidden_size)
        self.encoder = encoder
        self.post = nn.Linear(hidden_size, 4 * 2)

        self.register_state_dict_post_hook(_remove_ssl_model_state_dict)
        self.register_load_state_dict_pre_hook(_reject_ssl_model_state_dict)
        self.register_load_state_dict_post_hook(_ignore_ssl_model_missing)

    def forward(  # noqa: D102
        self,
        *,
        wave: Tensor,  # (B, max(wL))
        phoneme_index: Tensor,  # (B, max(fL))
        phoneme_id: Tensor,  # (B, max(pL))
        vowel_index: Tensor,  # (B, max(mL))
        mora_f0: Tensor,  # (B, max(mL))
        wave_length: Tensor,  # (B,)
        phoneme_length: Tensor,  # (B,)
        mora_length: Tensor,  # (B,)
    ) -> Tensor:  # (B, max(mL), 2, 4)
        attention_mask = make_non_pad_mask(wave_length).long()  # (B, max(wL))
        with torch.no_grad():
            hidden_layers = self.ssl_model.extract_hidden_layers(wave, attention_mask)
        feature = torch.stack(hidden_layers, dim=-1)  # (B, max(fL), ?, 12)

        layer_weight = torch.softmax(self.layer_weight, dim=0)  # (12,)
        feature = (feature * layer_weight).sum(dim=-1)  # (B, max(fL), ?)

        fL = min(int(feature.size(1)), int(phoneme_index.size(1)))
        feature = feature[:, :fL, :]
        phoneme_index = phoneme_index[:, :fL]

        # NOTE: HuBERTの出力長は75%が1フレーム短いので微調整
        # FIXME: 本当はより正確に計算すべき
        frame_length = torch.round(
            wave_length.float() / self.sampling_rate * self.frame_rate
        ).long()  # (B,)
        frame_length = torch.clamp(frame_length, max=fL)

        phoneme_feature = self._aggregate_to_phoneme(
            feature=feature,
            frame_length=frame_length,
            phoneme_index=phoneme_index,
            phoneme_length=phoneme_length,
        )  # (B, max(pL), ?)

        max_phoneme_length = phoneme_feature.size(1)
        h = phoneme_feature  # (B, max(pL), ?)
        if self.use_phoneme:
            phoneme_id_embed = self.phoneme_embedder(
                phoneme_id[:, :max_phoneme_length]
            )  # (B, max(pL), ?)
            h = torch.cat([h, phoneme_id_embed], dim=2)  # (B, max(pL), ?)
        if self.use_f0:
            phoneme_f0 = self._scatter_mora_to_phoneme(
                mora_f0=mora_f0,
                vowel_index=vowel_index,
                mora_length=mora_length,
                max_phoneme_length=max_phoneme_length,
            )  # (B, max(pL), 1)
            h = torch.cat([h, phoneme_f0], dim=2)  # (B, max(pL), ?)
        h = self.pre_phoneme(h)  # (B, max(pL), ?)

        phoneme_mask = (
            make_non_pad_mask(phoneme_length).unsqueeze(-2).to(h.device)
        )  # (B, 1, max(pL))
        h, _ = self.encoder(x=h, cond=None, mask=phoneme_mask)  # (B, max(pL), ?)

        mora_h = self._select_vowel(
            phoneme_h=h, vowel_index=vowel_index, mora_length=mora_length
        )  # (B, max(mL), ?)

        output = self.post(mora_h)  # (B, max(mL), 4*2)
        return output.reshape(output.size(0), output.size(1), 2, 4)

    def _aggregate_to_phoneme(
        self,
        feature: Tensor,  # (B, max(fL), ?)
        frame_length: Tensor,  # (B,)
        phoneme_index: Tensor,  # (B, max(fL))
        phoneme_length: Tensor,  # (B,)
    ) -> Tensor:  # (B, max(pL), ?)
        """フレーム特徴を音素ごとに平均集約する"""
        batch_size, _, num_feature = feature.shape
        max_phoneme_length = int(phoneme_length.max().item())
        device = feature.device

        frame_mask = make_non_pad_mask(frame_length).to(device)  # (B, max(fL))
        masked_index = torch.where(
            frame_mask,
            phoneme_index,
            torch.full_like(phoneme_index, max_phoneme_length),
        )  # (B, max(fL))

        x = feature.new_zeros(
            batch_size, max_phoneme_length + 1, num_feature
        )  # (B, max(pL)+1, ?)
        x = x.scatter_reduce(
            1,
            masked_index.unsqueeze(-1).expand(-1, -1, num_feature),
            feature,
            reduce="mean",
            include_self=False,
        )
        return x[:, :max_phoneme_length]

    def _scatter_mora_to_phoneme(
        self,
        mora_f0: Tensor,  # (B, max(mL))
        vowel_index: Tensor,  # (B, max(mL))
        mora_length: Tensor,  # (B,)
        max_phoneme_length: int,
    ) -> Tensor:  # (B, max(pL), 1)
        """モーラ単位のf0を母音位置に配置し、子音位置は0埋めの音素単位f0にする"""
        batch_size = mora_f0.size(0)
        mora_mask = make_non_pad_mask(mora_length).to(mora_f0.device)  # (B, max(mL))
        masked_index = torch.where(
            mora_mask, vowel_index, torch.full_like(vowel_index, max_phoneme_length)
        )  # (B, max(mL))
        phoneme_f0 = mora_f0.new_zeros(batch_size, max_phoneme_length + 1)
        phoneme_f0 = phoneme_f0.scatter(1, masked_index, mora_f0)
        return phoneme_f0[:, :max_phoneme_length].unsqueeze(-1)

    def _select_vowel(
        self,
        phoneme_h: Tensor,  # (B, max(pL), ?)
        vowel_index: Tensor,  # (B, max(mL))
        mora_length: Tensor,  # (B,)
    ) -> Tensor:  # (B, max(mL), ?)
        """音素単位の特徴から母音位置のモーラ特徴を取り出す"""
        max_mora_length = int(mora_length.max().item())
        vowel_index = vowel_index[:, :max_mora_length]  # (B, max(mL))
        num_feature = phoneme_h.size(2)
        index = vowel_index.unsqueeze(-1).expand(-1, -1, num_feature)  # (B, max(mL), ?)
        return torch.gather(phoneme_h, 1, index)


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    hubert_config = create_base_hubert_config()
    expected_frame_rate = config.sampling_rate / math.prod(hubert_config.conv_stride)
    assert config.frame_rate == expected_frame_rate, (
        f"network.frame_rateがSSLモデルの実際のフレームレートと一致しません: "
        f"config={config.frame_rate}, expected={expected_frame_rate}"
    )
    ssl_model = load_ssl_model(
        model_type=config.ssl_model_type,
        model_path=config.ssl_model_path,
        device=torch.device("cpu"),
    )
    ssl_model.freeze()

    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.conformer_block_num,
        dropout_rate=config.conformer_dropout_rate,
        positional_dropout_rate=config.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.conformer_dropout_rate,
        use_macaron_style=True,
        use_conv_glu_module=config.conformer_use_conv_glu_module,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        ssl_model=ssl_model,
        sampling_rate=config.sampling_rate,
        frame_rate=config.frame_rate,
        phoneme_size=OjtPhoneme.num_phoneme,
        phoneme_embedding_size=config.phoneme_embedding_size,
        hidden_size=config.hidden_size,
        encoder=encoder,
        use_f0=config.use_f0,
        use_phoneme=config.use_phoneme,
    )
