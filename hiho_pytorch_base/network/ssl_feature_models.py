"""Hugging Face の HuBERT 実装に寄せた推論モデルを提供する。"""

import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from upath import UPath

from ..utility.upath_utility import to_local_path

# HuBERT 実装は Hugging Face Transformers の HubertModel を参考にした。
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
# fairseq is licensed under the MIT License, Copyright (c) Facebook, Inc. and its affiliates.
# Fp32MaskedGroupNorm は NVIDIA DeepLearningExamples の fp32_group_norm.py を参考にした。
# NVIDIA DeepLearningExamples は BSD 3-Clause License で公開されている。

MODEL_SAMPLE_RATE = 16000

CheckpointKind = Literal["fairseq", "huggingface"]


@dataclass(frozen=True)
class HubertConfig:
    """HuBERT base の構成を表す。"""

    conv_dim: tuple[int, ...]
    conv_kernel: tuple[int, ...]
    conv_stride: tuple[int, ...]
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_conv_pos_embeddings: int
    num_conv_pos_embedding_groups: int
    num_feat_extract_layers: int
    feat_extract_activation: Literal["gelu"]
    feat_extract_norm: Literal["group", "layer"]
    feat_proj_dropout: float
    feat_proj_layer_norm: bool
    hidden_dropout: float
    activation_dropout: float
    attention_dropout: float
    layerdrop: float
    layer_norm_eps: float


def _get_conv_output_lengths(
    input_lengths: Tensor | None,
    kernel_size: int,
    stride: int,
) -> Tensor | None:
    if input_lengths is None:
        return None
    output_lengths = torch.div(
        input_lengths - kernel_size,
        stride,
        rounding_mode="floor",
    )
    output_lengths = output_lengths + 1
    if bool(torch.any(output_lengths <= 0).item()):
        raise ValueError("畳み込み後の長さが0以下です")
    return output_lengths


class Fp32MaskedGroupNorm(nn.Module):
    """padding を除いた統計量で GroupNorm を計算する。"""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float,
        affine: bool,
    ) -> None:
        super().__init__()
        if num_groups != num_channels:
            raise ValueError(
                f"MaskedGroupNorm は num_groups と num_channels が同じ場合だけ対応しています: {num_groups}, {num_channels}"
            )
        self._group_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )

    def forward(
        self,
        hidden_states: Tensor,
        hidden_lengths: Tensor | None,
    ) -> Tensor:
        """正規化後の特徴量を返す。"""
        if hidden_lengths is None:
            return self._forward_group_norm(hidden_states)

        self._validate_hidden_lengths(
            hidden_states=hidden_states,
            hidden_lengths=hidden_lengths,
        )
        original_hidden_states = hidden_states
        hidden_states = hidden_states.float()
        hidden_lengths = hidden_lengths.to(
            device=hidden_states.device, dtype=torch.long
        )
        normalized_hidden_states = torch.zeros_like(hidden_states)
        weight = self._group_norm.weight
        bias = self._group_norm.bias
        for batch_index in range(hidden_states.shape[0]):
            hidden_length = int(hidden_lengths[batch_index].item())
            valid_hidden_states = hidden_states[
                batch_index : batch_index + 1,
                :,
                :hidden_length,
            ]
            normalized_hidden_states[
                batch_index : batch_index + 1,
                :,
                :hidden_length,
            ] = F.group_norm(
                valid_hidden_states,
                self._group_norm.num_groups,
                weight.float() if weight is not None else None,
                bias.float() if bias is not None else None,
                self._group_norm.eps,
            )
        return normalized_hidden_states.type_as(original_hidden_states)

    def _forward_group_norm(self, hidden_states: Tensor) -> Tensor:
        weight = self._group_norm.weight
        bias = self._group_norm.bias
        hidden_states_float = F.group_norm(
            hidden_states.float(),
            self._group_norm.num_groups,
            weight.float() if weight is not None else None,
            bias.float() if bias is not None else None,
            self._group_norm.eps,
        )
        return hidden_states_float.type_as(hidden_states)

    def _validate_hidden_lengths(
        self,
        hidden_states: Tensor,
        hidden_lengths: Tensor,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError("GroupNorm 入力は3次元にしてください")
        if hidden_lengths.ndim != 1:
            raise ValueError("hidden_lengths は1次元にしてください")
        if hidden_lengths.shape[0] != hidden_states.shape[0]:
            raise ValueError("hidden_lengths の batch size が入力と一致しません")
        if bool(torch.any(hidden_lengths <= 0).item()):
            raise ValueError("hidden_lengths が0以下です")
        if bool(torch.any(hidden_lengths > hidden_states.shape[-1]).item()):
            raise ValueError("hidden_lengths が系列長を超えています")


class HubertGroupNormConvLayer(nn.Module):
    """GroupNorm 付き畳み込み層を表す。"""

    def __init__(self, config: HubertConfig, layer_id: int) -> None:
        super().__init__()
        in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=False,
        )
        self.layer_norm = Fp32MaskedGroupNorm(
            num_groups=out_conv_dim,
            num_channels=out_conv_dim,
            eps=config.layer_norm_eps,
            affine=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        hidden_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """畳み込み特徴量を返す。"""
        hidden_states = self.conv(hidden_states)
        hidden_lengths = _get_conv_output_lengths(
            input_lengths=hidden_lengths,
            kernel_size=self.conv.kernel_size[0],
            stride=self.conv.stride[0],
        )
        hidden_states = self.layer_norm(hidden_states, hidden_lengths)
        return F.gelu(hidden_states), hidden_lengths


class HubertNoLayerNormConvLayer(nn.Module):
    """正規化なし畳み込み層を表す。"""

    def __init__(self, config: HubertConfig, layer_id: int) -> None:
        super().__init__()
        in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        hidden_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """畳み込み特徴量を返す。"""
        hidden_states = self.conv(hidden_states)
        hidden_lengths = _get_conv_output_lengths(
            input_lengths=hidden_lengths,
            kernel_size=self.conv.kernel_size[0],
            stride=self.conv.stride[0],
        )
        return F.gelu(hidden_states), hidden_lengths


class HubertLayerNormConvLayer(nn.Module):
    """LayerNorm 付き畳み込み層を表す。"""

    def __init__(self, config: HubertConfig, layer_id: int) -> None:
        super().__init__()
        in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        out_conv_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=False,
        )
        self.layer_norm = nn.LayerNorm(out_conv_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        hidden_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """畳み込み特徴量を返す。"""
        hidden_states = self.conv(hidden_states)
        hidden_lengths = _get_conv_output_lengths(
            input_lengths=hidden_lengths,
            kernel_size=self.conv.kernel_size[0],
            stride=self.conv.stride[0],
        )
        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        return F.gelu(hidden_states), hidden_lengths


class HubertFeatureEncoder(nn.Module):
    """raw audio waveform から畳み込み特徴量を構築する。"""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        if config.feat_extract_norm == "group":
            conv_layers: list[nn.Module] = [
                HubertGroupNormConvLayer(config, layer_id=0)
            ]
            conv_layers.extend(
                HubertNoLayerNormConvLayer(config, layer_id=index + 1)
                for index in range(config.num_feat_extract_layers - 1)
            )
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                HubertLayerNormConvLayer(config, layer_id=index)
                for index in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"未対応の feat_extract_norm です: {config.feat_extract_norm}"
            )
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(
        self,
        input_values: Tensor,
        input_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """畳み込み特徴量を返す。"""
        hidden_states = input_values[:, None]
        hidden_lengths = input_lengths
        for conv_layer in self.conv_layers:
            hidden_states, hidden_lengths = conv_layer(hidden_states, hidden_lengths)
        return hidden_states, hidden_lengths


class HubertFeatureProjection(nn.Module):
    """畳み込み特徴量を Transformer 入力次元へ射影する。"""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(
                config.conv_dim[-1], eps=config.layer_norm_eps
            )
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """射影後の特徴量を返す。"""
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        return self.dropout(hidden_states)


class HubertWeightNormConv1d(nn.Module):
    """weight_norm 付き Conv1d を checkpoint 互換のパラメータ名で表す。"""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        kernel_size = config.num_conv_pos_embeddings
        channels_per_group = config.hidden_size // config.num_conv_pos_embedding_groups
        self.padding = kernel_size // 2
        self.groups = config.num_conv_pos_embedding_groups
        self.weight_g = nn.Parameter(torch.empty(1, 1, kernel_size))
        self.weight_v = nn.Parameter(
            torch.empty(config.hidden_size, channels_per_group, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """位置畳み込み結果を返す。"""
        weight_norm = torch.linalg.vector_norm(
            self.weight_v,
            ord=2,
            dim=(0, 1),
            keepdim=True,
        )
        if bool(torch.any(weight_norm == 0).item()):
            raise ValueError("位置畳み込みの weight_v のノルムが0です")
        weight = self.weight_v * (self.weight_g / weight_norm)
        return F.conv1d(
            hidden_states,
            weight,
            self.bias,
            padding=self.padding,
            groups=self.groups,
        )


class HubertSamePadLayer(nn.Module):
    """偶数カーネルの余分な右端 padding を取り除く。"""

    def __init__(self, num_conv_pos_embeddings: int) -> None:
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states: Tensor) -> Tensor:
        """padding 調整後の特徴量を返す。"""  # noqa: D401
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class HubertPositionalConvEmbedding(nn.Module):
    """Transformer 入力に加算する畳み込み位置埋め込みを表す。"""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        self.conv = HubertWeightNormConv1d(config)
        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """位置埋め込みを返す。"""
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states.transpose(1, 2)


class HubertAttention(nn.Module):
    """Multi-headed attention を表す。"""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size は num_attention_heads で割り切れる必要があります"
            )
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """attention 出力を返す。"""
        batch_size, frame_count, hidden_size = hidden_states.shape
        query_states = self._shape(self.q_proj(hidden_states), batch_size, frame_count)
        key_states = self._shape(self.k_proj(hidden_states), batch_size, frame_count)
        value_states = self._shape(self.v_proj(hidden_states), batch_size, frame_count)

        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attention_weights = attention_weights * self.scaling
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_probabilities = F.dropout(
            attention_weights,
            p=self.dropout,
            training=self.training,
        )
        attention_output = torch.matmul(attention_probabilities, value_states)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size,
            frame_count,
            hidden_size,
        )
        return self.out_proj(attention_output)

    def _shape(self, states: Tensor, batch_size: int, frame_count: int) -> Tensor:
        return states.view(
            batch_size,
            frame_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)


class HubertFeedForward(nn.Module):
    """Transformer layer の FFN を表す。"""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """FFN 出力を返す。"""
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return self.output_dropout(hidden_states)


class HubertEncoderLayer(nn.Module):
    """HuBERT encoder の1層を表す。"""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        self.attention = HubertAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """encoder layer の出力を返す。"""  # noqa: D401
        attention_residual = hidden_states
        hidden_states = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attention_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        return self.final_layer_norm(hidden_states)


class HubertEncoder(nn.Module):
    """HuBERT encoder を表す。"""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        self.layerdrop: float = config.layerdrop
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [HubertEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> list[Tensor]:
        """各層後の hidden state のリストを返す。"""
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.unsqueeze(-1).expand(
                -1,
                -1,
                hidden_states.shape[2],
            )
            hidden_states = hidden_states.masked_fill(~expanded_attention_mask, 0.0)

        additive_attention_mask = _create_bidirectional_attention_mask(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.device)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states: list[Tensor] = []
        for layer in self.layers:
            all_hidden_states.append(hidden_states)

            dropout_probability = torch.rand([], device=hidden_states.device)
            skip = self.training and dropout_probability < self.layerdrop
            if not skip:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=additive_attention_mask,
                )

        all_hidden_states.append(hidden_states)
        return all_hidden_states


class HubertModel(nn.Module):
    """HuBERT encoder-only model を表す。"""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__()
        self.num_hidden_layers: int = config.num_hidden_layers
        self.hidden_size: int = config.hidden_size
        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = HubertFeatureProjection(config)
        self.encoder = HubertEncoder(config)

    def extract_hidden_layers(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
    ) -> list[Tensor]:
        """Transformer 各層後の hidden state を返す。"""  # noqa: D401
        if input_values.ndim != 2:
            raise ValueError("入力波形は2次元にしてください")

        input_lengths: Tensor | None = None
        if attention_mask is not None:
            input_lengths = self._get_input_lengths(
                input_values=input_values,
                attention_mask=attention_mask,
            )

        extract_features, output_lengths = self.feature_extractor(
            input_values=input_values,
            input_lengths=input_lengths,
        )
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            if output_lengths is None:
                raise ValueError("feature extractor の出力長がありません")
            attention_mask = self._get_feature_vector_attention_mask(
                feature_vector_length=extract_features.shape[1],
                output_lengths=output_lengths,
            )

        hidden_states = self.feature_projection(extract_features)
        all_hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        return all_hidden_states[1:]

    def _get_input_lengths(
        self,
        input_values: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if bool(torch.any((attention_mask != 0) & (attention_mask != 1)).item()):
            raise ValueError("attention_mask は0または1だけにしてください")

        if attention_mask.shape[1] >= 2:
            mask_increases = (attention_mask[:, 1:] == 1) & (
                attention_mask[:, :-1] == 0
            )
            if bool(torch.any(mask_increases).item()):
                raise ValueError(
                    "attention_mask は左詰めの padding mask にしてください"
                )

        input_lengths = attention_mask.to(torch.long).sum(dim=-1)
        if bool(torch.any(input_lengths <= 0).item()):
            raise ValueError("入力長が0以下です")
        return input_lengths

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        output_lengths: Tensor,
    ) -> Tensor:
        """feature-level attention mask を返す。"""
        output_lengths = output_lengths.to(torch.long)
        if bool(torch.any(output_lengths <= 0).item()):
            raise ValueError("畳み込み後の長さが0以下です")
        if bool(torch.any(output_lengths > feature_vector_length).item()):
            raise ValueError("畳み込み後の長さが feature_vector_length を超えています")

        batch_size = output_lengths.shape[0]
        feature_attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=torch.bool,
            device=output_lengths.device,
        )
        feature_attention_mask[
            torch.arange(batch_size, device=output_lengths.device),
            output_lengths - 1,
        ] = True
        feature_attention_mask = feature_attention_mask.flip([-1]).cumsum(-1).flip([-1])
        return feature_attention_mask > 0


def _create_bidirectional_attention_mask(
    hidden_states: Tensor,
    attention_mask: Tensor | None,
) -> Tensor | None:
    """bidirectional attention 用の加算 mask を返す。"""
    if attention_mask is None:
        return None
    additive_attention_mask = attention_mask[:, None, None, :]
    additive_attention_mask = additive_attention_mask.to(dtype=hidden_states.dtype)
    additive_attention_mask = 1.0 - additive_attention_mask
    additive_attention_mask = (
        additive_attention_mask * -3.4028235e38  # NOTE: float32 の最小値
    )
    return additive_attention_mask


def load_japanese_hubert_model(model_path: UPath, device: torch.device) -> HubertModel:
    """日本語 HuBERT の重みを読み込む。"""
    state, checkpoint_kind = _load_checkpoint_state(model_path)
    config = create_base_hubert_config()
    model = HubertModel(config)
    if checkpoint_kind == "fairseq":
        converted_state = _convert_fairseq_state(state, config)
    elif checkpoint_kind == "huggingface":
        converted_state = _convert_huggingface_state(state, config)
    else:
        raise ValueError(f"未対応のチェックポイント形式です: {checkpoint_kind}")
    model.load_state_dict(converted_state, strict=True)
    model.eval()
    model.to(device)
    return model


def load_contentvec_model(model_path: UPath, device: torch.device) -> HubertModel:
    """ContentVec の重みを読み込む。"""
    state, checkpoint_kind = _load_checkpoint_state(model_path)
    if checkpoint_kind != "fairseq":
        raise ValueError(f"ContentVec は fairseq 形式だけ対応しています: {model_path}")
    config = create_base_hubert_config()
    model = HubertModel(config)
    converted_state = _convert_fairseq_state(state, config)
    model.load_state_dict(converted_state, strict=True)
    model.eval()
    model.to(device)
    return model


def create_base_hubert_config() -> HubertConfig:
    """HuBERT base の構成を返す。"""
    return HubertConfig(
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        num_feat_extract_layers=7,
        feat_extract_activation="gelu",
        feat_extract_norm="group",
        feat_proj_dropout=0.0,
        feat_proj_layer_norm=True,
        hidden_dropout=0.1,
        activation_dropout=0.0,
        attention_dropout=0.1,
        layerdrop=0.0,
        layer_norm_eps=1e-5,
    )


_FairseqDictionary = type(
    "Dictionary",
    (),
    {"__module__": "fairseq.data.dictionary"},
)


def _load_checkpoint_state(
    model_path: UPath,
) -> tuple[dict[str, Tensor], CheckpointKind]:
    _install_fairseq_dictionary_stub()
    local_model_path = to_local_path(model_path)
    checkpoint = torch.load(
        local_model_path,
        map_location="cpu",
        weights_only=True,
    )
    state = _extract_tensor_state(checkpoint)
    if _has_key_prefix(state, "feature_extractor.conv_layers.0.0."):
        return state, "fairseq"
    if _has_key_prefix(state, "feature_extractor.conv_layers.0.conv."):
        return _strip_prefix(state, "hubert."), "huggingface"
    if _has_key_prefix(state, "hubert.feature_extractor.conv_layers.0.conv."):
        return _strip_prefix(state, "hubert."), "huggingface"
    raise ValueError(f"チェックポイント形式を判別できません: {model_path}")


def _install_fairseq_dictionary_stub() -> None:
    if "fairseq.data.dictionary" in sys.modules:
        dictionary_module = sys.modules["fairseq.data.dictionary"]
        if not hasattr(dictionary_module, "Dictionary"):
            raise ValueError("fairseq.data.dictionary.Dictionary が見つかりません")
        return

    fairseq_module = sys.modules.get("fairseq", types.ModuleType("fairseq"))
    fairseq_data_module = sys.modules.get(
        "fairseq.data", types.ModuleType("fairseq.data")
    )
    fairseq_dictionary_module = types.ModuleType("fairseq.data.dictionary")
    fairseq_dictionary_module.Dictionary = _FairseqDictionary
    fairseq_module.data = fairseq_data_module
    fairseq_data_module.dictionary = fairseq_dictionary_module
    sys.modules["fairseq"] = fairseq_module
    sys.modules["fairseq.data"] = fairseq_data_module
    sys.modules["fairseq.data.dictionary"] = fairseq_dictionary_module
    torch.serialization.add_safe_globals([_FairseqDictionary])


def _extract_tensor_state(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"チェックポイントが dict ではありません: {type(checkpoint)}")
    if "model" in checkpoint:
        model_state = checkpoint["model"]
        if not isinstance(model_state, Mapping):
            raise ValueError(f"model が dict ではありません: {type(model_state)}")
        return _validate_tensor_state(model_state)
    return _validate_tensor_state(checkpoint)


def _validate_tensor_state(state: Mapping[object, object]) -> dict[str, Tensor]:
    tensor_state: dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError(f"重みキーが文字列ではありません: {key}")
        if not isinstance(value, Tensor):
            continue
        tensor_state[key] = value
    if len(tensor_state) == 0:
        raise ValueError("テンソル重みがありません")
    return tensor_state


def _has_key_prefix(state: Mapping[str, Tensor], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state)


def _strip_prefix(state: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    stripped_state: dict[str, Tensor] = {}
    for key, value in state.items():
        if key.startswith(prefix):
            stripped_state[key.removeprefix(prefix)] = value
        else:
            stripped_state[key] = value
    return stripped_state


def _convert_fairseq_state(
    state: Mapping[str, Tensor],
    config: HubertConfig,
) -> dict[str, Tensor]:
    converted_state: dict[str, Tensor] = {}
    for index in range(len(config.conv_dim)):
        converted_state[f"feature_extractor.conv_layers.{index}.conv.weight"] = (
            _require_tensor(state, f"feature_extractor.conv_layers.{index}.0.weight")
        )
    converted_state["feature_extractor.conv_layers.0.layer_norm._group_norm.weight"] = (
        _require_tensor(state, "feature_extractor.conv_layers.0.2.weight")
    )
    converted_state["feature_extractor.conv_layers.0.layer_norm._group_norm.bias"] = (
        _require_tensor(state, "feature_extractor.conv_layers.0.2.bias")
    )
    converted_state["feature_projection.layer_norm.weight"] = _require_tensor(
        state, "layer_norm.weight"
    )
    converted_state["feature_projection.layer_norm.bias"] = _require_tensor(
        state, "layer_norm.bias"
    )
    converted_state["feature_projection.projection.weight"] = _require_tensor(
        state, "post_extract_proj.weight"
    )
    converted_state["feature_projection.projection.bias"] = _require_tensor(
        state, "post_extract_proj.bias"
    )
    converted_state.update(_convert_fairseq_encoder_state(state, config))
    return converted_state


def _convert_fairseq_encoder_state(
    state: Mapping[str, Tensor],
    config: HubertConfig,
) -> dict[str, Tensor]:
    converted_state: dict[str, Tensor] = {
        "encoder.pos_conv_embed.conv.weight_g": _require_tensor(
            state, "encoder.pos_conv.0.weight_g"
        ),
        "encoder.pos_conv_embed.conv.weight_v": _require_tensor(
            state, "encoder.pos_conv.0.weight_v"
        ),
        "encoder.pos_conv_embed.conv.bias": _require_tensor(
            state, "encoder.pos_conv.0.bias"
        ),
        "encoder.layer_norm.weight": _require_tensor(
            state, "encoder.layer_norm.weight"
        ),
        "encoder.layer_norm.bias": _require_tensor(state, "encoder.layer_norm.bias"),
    }
    for index in range(config.num_hidden_layers):
        src = f"encoder.layers.{index}"
        dst = f"encoder.layers.{index}"
        converted_state.update(
            {
                f"{dst}.attention.k_proj.weight": _require_tensor(
                    state, f"{src}.self_attn.k_proj.weight"
                ),
                f"{dst}.attention.k_proj.bias": _require_tensor(
                    state, f"{src}.self_attn.k_proj.bias"
                ),
                f"{dst}.attention.v_proj.weight": _require_tensor(
                    state, f"{src}.self_attn.v_proj.weight"
                ),
                f"{dst}.attention.v_proj.bias": _require_tensor(
                    state, f"{src}.self_attn.v_proj.bias"
                ),
                f"{dst}.attention.q_proj.weight": _require_tensor(
                    state, f"{src}.self_attn.q_proj.weight"
                ),
                f"{dst}.attention.q_proj.bias": _require_tensor(
                    state, f"{src}.self_attn.q_proj.bias"
                ),
                f"{dst}.attention.out_proj.weight": _require_tensor(
                    state, f"{src}.self_attn.out_proj.weight"
                ),
                f"{dst}.attention.out_proj.bias": _require_tensor(
                    state, f"{src}.self_attn.out_proj.bias"
                ),
                f"{dst}.layer_norm.weight": _require_tensor(
                    state, f"{src}.self_attn_layer_norm.weight"
                ),
                f"{dst}.layer_norm.bias": _require_tensor(
                    state, f"{src}.self_attn_layer_norm.bias"
                ),
                f"{dst}.feed_forward.intermediate_dense.weight": _require_tensor(
                    state, f"{src}.fc1.weight"
                ),
                f"{dst}.feed_forward.intermediate_dense.bias": _require_tensor(
                    state, f"{src}.fc1.bias"
                ),
                f"{dst}.feed_forward.output_dense.weight": _require_tensor(
                    state, f"{src}.fc2.weight"
                ),
                f"{dst}.feed_forward.output_dense.bias": _require_tensor(
                    state, f"{src}.fc2.bias"
                ),
                f"{dst}.final_layer_norm.weight": _require_tensor(
                    state, f"{src}.final_layer_norm.weight"
                ),
                f"{dst}.final_layer_norm.bias": _require_tensor(
                    state, f"{src}.final_layer_norm.bias"
                ),
            }
        )
    return converted_state


def _convert_huggingface_state(
    state: Mapping[str, Tensor],
    config: HubertConfig,
) -> dict[str, Tensor]:
    converted_state: dict[str, Tensor] = {}
    for index in range(len(config.conv_dim)):
        src = f"feature_extractor.conv_layers.{index}"
        converted_state[f"{src}.conv.weight"] = _require_tensor(
            state, f"{src}.conv.weight"
        )
    converted_state["feature_extractor.conv_layers.0.layer_norm._group_norm.weight"] = (
        _require_tensor(state, "feature_extractor.conv_layers.0.layer_norm.weight")
    )
    converted_state["feature_extractor.conv_layers.0.layer_norm._group_norm.bias"] = (
        _require_tensor(state, "feature_extractor.conv_layers.0.layer_norm.bias")
    )
    converted_state["feature_projection.layer_norm.weight"] = _require_tensor(
        state, "feature_projection.layer_norm.weight"
    )
    converted_state["feature_projection.layer_norm.bias"] = _require_tensor(
        state, "feature_projection.layer_norm.bias"
    )
    converted_state["feature_projection.projection.weight"] = _require_tensor(
        state, "feature_projection.projection.weight"
    )
    converted_state["feature_projection.projection.bias"] = _require_tensor(
        state, "feature_projection.projection.bias"
    )
    converted_state.update(_convert_huggingface_encoder_state(state, config))
    return converted_state


def _convert_huggingface_encoder_state(
    state: Mapping[str, Tensor],
    config: HubertConfig,
) -> dict[str, Tensor]:
    converted_state: dict[str, Tensor] = {
        "encoder.pos_conv_embed.conv.weight_g": _require_tensor(
            state, "encoder.pos_conv_embed.conv.weight_g"
        ),
        "encoder.pos_conv_embed.conv.weight_v": _require_tensor(
            state, "encoder.pos_conv_embed.conv.weight_v"
        ),
        "encoder.pos_conv_embed.conv.bias": _require_tensor(
            state, "encoder.pos_conv_embed.conv.bias"
        ),
        "encoder.layer_norm.weight": _require_tensor(
            state, "encoder.layer_norm.weight"
        ),
        "encoder.layer_norm.bias": _require_tensor(state, "encoder.layer_norm.bias"),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"encoder.layers.{index}"
        converted_state.update(
            {
                f"{prefix}.attention.k_proj.weight": _require_tensor(
                    state, f"{prefix}.attention.k_proj.weight"
                ),
                f"{prefix}.attention.k_proj.bias": _require_tensor(
                    state, f"{prefix}.attention.k_proj.bias"
                ),
                f"{prefix}.attention.v_proj.weight": _require_tensor(
                    state, f"{prefix}.attention.v_proj.weight"
                ),
                f"{prefix}.attention.v_proj.bias": _require_tensor(
                    state, f"{prefix}.attention.v_proj.bias"
                ),
                f"{prefix}.attention.q_proj.weight": _require_tensor(
                    state, f"{prefix}.attention.q_proj.weight"
                ),
                f"{prefix}.attention.q_proj.bias": _require_tensor(
                    state, f"{prefix}.attention.q_proj.bias"
                ),
                f"{prefix}.attention.out_proj.weight": _require_tensor(
                    state, f"{prefix}.attention.out_proj.weight"
                ),
                f"{prefix}.attention.out_proj.bias": _require_tensor(
                    state, f"{prefix}.attention.out_proj.bias"
                ),
                f"{prefix}.layer_norm.weight": _require_tensor(
                    state, f"{prefix}.layer_norm.weight"
                ),
                f"{prefix}.layer_norm.bias": _require_tensor(
                    state, f"{prefix}.layer_norm.bias"
                ),
                f"{prefix}.feed_forward.intermediate_dense.weight": _require_tensor(
                    state, f"{prefix}.feed_forward.intermediate_dense.weight"
                ),
                f"{prefix}.feed_forward.intermediate_dense.bias": _require_tensor(
                    state, f"{prefix}.feed_forward.intermediate_dense.bias"
                ),
                f"{prefix}.feed_forward.output_dense.weight": _require_tensor(
                    state, f"{prefix}.feed_forward.output_dense.weight"
                ),
                f"{prefix}.feed_forward.output_dense.bias": _require_tensor(
                    state, f"{prefix}.feed_forward.output_dense.bias"
                ),
                f"{prefix}.final_layer_norm.weight": _require_tensor(
                    state, f"{prefix}.final_layer_norm.weight"
                ),
                f"{prefix}.final_layer_norm.bias": _require_tensor(
                    state, f"{prefix}.final_layer_norm.bias"
                ),
            }
        )
    return converted_state


def _require_tensor(state: Mapping[str, Tensor], key: str) -> Tensor:
    value = state.get(key)
    if value is None:
        raise ValueError(f"必要な重みがありません: {key}")
    if not isinstance(value, Tensor):
        raise ValueError(f"重みが Tensor ではありません: {key}")
    return value


SslModelType = Literal["hubert", "contentvec"]


def compute_ssl_frame_length(wave_length: int, config: HubertConfig) -> int:
    """波形サンプル数からSSL特徴量のフレーム数を計算する。"""
    length = wave_length
    for kernel, stride in zip(config.conv_kernel, config.conv_stride, strict=True):
        length = (length - kernel) // stride + 1
    return length


def load_ssl_model(
    model_type: SslModelType, model_path: UPath, device: torch.device
) -> HubertModel:
    """SSLモデルをモデルタイプに応じて読み込む。"""
    if model_type == "hubert":
        return load_japanese_hubert_model(model_path=model_path, device=device)
    elif model_type == "contentvec":
        return load_contentvec_model(model_path=model_path, device=device)
    else:
        raise ValueError(f"未対応のSSLモデルタイプです: {model_type}")
