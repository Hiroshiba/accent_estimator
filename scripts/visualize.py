"""
機械学習データセットの可視化ツール

設定ファイルからDatasetCollectionを読み込み、データタイプごとにStreamlit UIで表示する。
各データタイプの表示形式は機械学習タスクに応じて_show_dataset_sectionや_show_inference_sectionをカスタマイズする。

起動方法:
    uv run -m streamlit run scripts/visualize.py -- --config_path <config> --predictor_path <predictor>
"""

import argparse

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from upath import UPath

from hiho_pytorch_base.batch import collate_dataset_output
from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data.data import OutputData, mora_phoneme_list
from hiho_pytorch_base.dataset import (
    DatasetCollection,
    DatasetType,
    LazyInputData,
    create_dataset,
)
from hiho_pytorch_base.generator import Generator, GeneratorOutput
from hiho_pytorch_base.utility.upath_utility import to_local_path

accent_channel_names = (
    "アクセント核開始",
    "アクセント核終了",
    "アクセント句開始",
    "アクセント句終了",
)


@st.cache_resource
def _load_config_and_dataset(config_path_str: str) -> tuple[Config, DatasetCollection]:
    """configパスからConfigとDatasetCollectionを読み込みキャッシュする"""
    config = Config.load(UPath(config_path_str))
    dataset_collection = create_dataset(config.dataset)
    return config, dataset_collection


@st.cache_resource
def _load_generator(config_path_str: str, predictor_path_str: str) -> Generator:
    """configパスとpredictorパスからGeneratorを読み込みキャッシュする"""
    config, _ = _load_config_and_dataset(config_path_str)
    return Generator(
        config=config,
        predictor=to_local_path(UPath(predictor_path_str)),
        use_gpu=False,
    )


def _run_inference(generator: Generator, output_data: OutputData) -> GeneratorOutput:
    """OutputDataから推論結果を生成する"""
    batch = collate_dataset_output([output_data])
    return generator(
        wave=batch.wave,
        phoneme_index=batch.phoneme_index,
        phoneme_id=batch.phoneme_id,
        vowel_index=batch.vowel_index,
        mora_f0=batch.mora_f0,
        wave_length=batch.wave_length,
        phoneme_length=batch.phoneme_length,
        mora_length=batch.mora_length,
    )


def _show_line_plot(data: np.ndarray, title: str, xaxis_title: str) -> None:
    """1次元データの折れ線グラフを表示する"""
    fig = go.Figure()
    fig.add_scatter(y=data, mode="lines")
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title="Value")
    st.plotly_chart(fig)


def _show_accent_heatmap(data: np.ndarray, title: str) -> None:
    """アクセント4種をモーラ単位のヒートマップで表示する"""
    fig = go.Figure(
        go.Heatmap(
            z=data.T,
            y=list(accent_channel_names),
            zmin=0,
            zmax=1,
            colorscale="Viridis",
        )
    )
    fig.update_layout(title=title, xaxis_title="Mora")
    st.plotly_chart(fig)


def _extract_mora_phoneme(lazy_data: LazyInputData) -> tuple[list[str], list[str]]:
    """各モーラの子音と母音を音素列から取得し、子音がなければ空文字にする"""
    phoneme_list = lazy_data.fetch().phoneme_list
    consonant: list[str] = []
    vowel: list[str] = []
    for i, p in enumerate(phoneme_list):
        if p.phoneme not in mora_phoneme_list:
            continue
        vowel.append(p.phoneme)
        prev = phoneme_list[i - 1].phoneme if i > 0 else None
        if prev is not None and prev not in mora_phoneme_list:
            consonant.append(prev)
        else:
            consonant.append("")
    return consonant, vowel


def _show_dataset_section(output_data: OutputData, lazy_data: LazyInputData) -> None:
    """データセットの入力データを可視化する"""
    wave_data = output_data.wave.cpu().numpy()
    mora_f0_data = output_data.mora_f0.cpu().numpy()
    accent_data = output_data.accent.cpu().numpy()
    speaker_id = int(output_data.speaker_id.item())
    consonant, vowel = _extract_mora_phoneme(lazy_data)

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        st.markdown("### 音声波形")
        _show_line_plot(wave_data, "音声波形", "Sample")
    with plot_col2:
        st.markdown("### モーラf0")
        _show_line_plot(mora_f0_data, "モーラf0", "Mora")

    st.markdown("### アクセント")
    _show_accent_heatmap(accent_data, "アクセント")

    mora_col1, mora_col2 = st.columns(2)
    with mora_col1:
        st.markdown("### モーラ列")
        mora_rows = [
            {"index": i, "子音": c, "母音": v}
            for i, (c, v) in enumerate(zip(consonant, vowel, strict=True))
        ]
        st.dataframe(mora_rows)
    with mora_col2:
        st.markdown("### その他の値")
        st.markdown(f"**話者ID**: {speaker_id}")


def _show_inference_section(generator_output: GeneratorOutput) -> None:
    """推論結果を可視化する"""
    accent_logit = generator_output.accent_logit[0]  # (max(mL), 2, 4)
    mora_length = int(generator_output.mora_length[0].item())
    accent_prob = (
        torch.softmax(accent_logit[:mora_length], dim=1)[:, 1, :].cpu().numpy()
    )  # (mL, 4)

    st.markdown("### 予測アクセント確率")
    _show_accent_heatmap(accent_prob, "予測アクセント確率")


def _show_details(
    config_path: UPath,
    output_data: OutputData,
    lazy_data: LazyInputData,
) -> None:
    """詳細情報を表示する"""
    st.markdown("### 詳細情報")
    st.code(
        f"""
設定ファイル: {config_path}

音声波形
パス: {lazy_data.wave_path}
shape: {tuple(output_data.wave.shape)}

音素列
パス: {lazy_data.phoneme_list_path}

母音位置列
shape: {tuple(output_data.vowel_index.shape)}

モーラf0
shape: {tuple(output_data.mora_f0.shape)}

アクセント核開始
パス: {lazy_data.accent_start_path}

アクセント核終了
パス: {lazy_data.accent_end_path}

アクセント句開始
パス: {lazy_data.accent_phrase_start_path}

アクセント句終了
パス: {lazy_data.accent_phrase_end_path}

shape: {tuple(output_data.accent.shape)}

話者ID: {output_data.speaker_id.item()}
""",
        language=None,
    )


def _to_optional_path(path_str: str) -> UPath | None:
    """テキスト入力をパスへ変換し、空なら指定なしとしてNoneを返す"""
    stripped = path_str.strip()
    if len(stripped) == 0:
        return None
    return UPath(stripped)


def visualize(
    config_path: UPath | None,
    dataset_type: DatasetType,
    predictor_path: UPath | None,
) -> None:
    """指定されたデータセットをStreamlit UIで可視化する"""
    st.set_page_config(page_title="データセット可視化", layout="wide")

    initial_config_path_str = str(config_path) if config_path is not None else ""
    initial_predictor_path_str = (
        str(predictor_path) if predictor_path is not None else ""
    )

    input_col1, input_col2 = st.columns(2)
    config_path_str = input_col1.text_input(
        "設定ファイルパス",
        value=initial_config_path_str,
        placeholder="/path/to/config.yaml",
    )
    predictor_path_str = input_col2.text_input(
        "モデルファイルパス",
        value=initial_predictor_path_str,
        placeholder="/path/to/predictor.pth",
    )

    selected_dataset_type = st.selectbox(
        "データセットタイプ",
        options=list(DatasetType),
        index=list(DatasetType).index(dataset_type),
        format_func=lambda t: t.value,
    )

    target_config_path = _to_optional_path(config_path_str)
    if target_config_path is None:
        st.info("設定ファイルパスを指定してください")
        return
    if not target_config_path.exists():
        st.error(f"設定ファイルが見つかりません: {target_config_path}")
        return

    _, dataset_collection = _load_config_and_dataset(str(target_config_path))
    dataset = dataset_collection.get(selected_dataset_type)
    if len(dataset) == 0:
        st.error("データセットが空です")
        return

    max_index = len(dataset) - 1
    index = st.slider(
        "サンプルインデックス",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1,
    )

    output_data = dataset[index]
    lazy_data = dataset.datas[index]

    _show_dataset_section(output_data, lazy_data)

    target_predictor_path = _to_optional_path(predictor_path_str)
    if target_predictor_path is not None:
        st.markdown("## 推論結果")
        if not target_predictor_path.exists():
            st.error(f"モデルファイルが見つかりません: {target_predictor_path}")
        else:
            generator = _load_generator(
                str(target_config_path), str(target_predictor_path)
            )
            generator_output = _run_inference(generator, output_data)
            _show_inference_section(generator_output)

    st.markdown("---")
    _show_details(target_config_path, output_data, lazy_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセットのビジュアライゼーション")
    parser.add_argument("--config_path", type=UPath, help="設定ファイルのパス")
    parser.add_argument(
        "--dataset_type",
        type=DatasetType,
        default=DatasetType.TRAIN,
        help="データセットタイプ",
    )
    parser.add_argument(
        "--predictor_path",
        type=UPath,
        help="推論結果を可視化する場合のpredictorモデルパス",
    )
    args = parser.parse_args()
    visualize(
        config_path=args.config_path,
        dataset_type=args.dataset_type,
        predictor_path=args.predictor_path,
    )
