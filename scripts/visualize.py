"""
機械学習データセットの可視化ツール

設定ファイルからDatasetCollectionを読み込み、データタイプごとにStreamlit UIで表示する。
各データタイプの表示形式は機械学習タスクに応じてカスタマイズする。
データタイプに応じた可視化ロジックを_show_line_plotや_create_details_textで調整する。

起動方法:
    uv run streamlit run scripts/visualize.py -- --config_path <config> --predictor_path <predictor>
"""

import argparse

import japanize_matplotlib  # noqa: F401 日本語フォントに必須
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from upath import UPath

from hiho_pytorch_base.batch import collate_dataset_output
from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data.data import OutputData
from hiho_pytorch_base.dataset import (
    DatasetCollection,
    DatasetType,
    LazyInputData,
    create_dataset,
)
from hiho_pytorch_base.generator import Generator, GeneratorOutput
from hiho_pytorch_base.utility.upath_utility import to_local_path


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
        feature_vector=batch.feature_vector,
        feature_variable=batch.feature_variable,
        speaker_id=batch.speaker_id,
        length=batch.length,
    )


def _show_line_plot(data: np.ndarray, title: str) -> None:
    """1次元データの折れ線グラフを表示する"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(data)), data)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)


def _show_pred_target_plot(pred_data: np.ndarray, target_data: np.ndarray) -> None:
    """可変長出力の予測と正解の折れ線グラフを表示する"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(pred_data)), pred_data, label="予測")
    ax.plot(range(len(target_data)), target_data, label="正解")
    ax.set_title("可変長出力の予測と正解")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)


def _create_details_text(
    config_path: UPath,
    output_data: OutputData,
    lazy_data: LazyInputData,
) -> str:
    """詳細情報テキストを作成する"""
    return f"""
設定ファイル: {config_path}

固定長特徴ベクトル
パス: {lazy_data.feature_vector_path}
shape: {tuple(output_data.feature_vector.shape)}

可変長特徴データ
パス: {lazy_data.feature_variable_path}
shape: {tuple(output_data.feature_variable.shape)}

サンプリングデータ
パス: {lazy_data.target_vector_path}
shape: {tuple(output_data.target_vector.shape)}

可変長ターゲット
パス: {lazy_data.target_variable_path}
shape: {tuple(output_data.target_variable.shape)}

回帰ターゲット
パス: {lazy_data.target_scalar_path}
shape: {tuple(output_data.target_scalar.shape)}

話者ID: {output_data.speaker_id.item()}
"""


def _to_config_path(config_path_str: str) -> UPath | None:
    """テキスト入力をconfigパスへ変換し、空なら指定なしとしてNoneを返す"""
    stripped = config_path_str.strip()
    if len(stripped) == 0:
        return None
    return UPath(stripped)


def _to_predictor_path(predictor_path_str: str) -> UPath | None:
    """テキスト入力をpredictorパスへ変換し、空なら指定なしとしてNoneを返す"""
    stripped = predictor_path_str.strip()
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

    target_config_path = _to_config_path(config_path_str)
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
    if max_index > 0:
        index = st.slider(
            "サンプルインデックス",
            min_value=0,
            max_value=max_index,
            value=0,
            step=1,
        )
    else:
        index = 0

    output_data = dataset[index]
    lazy_data = dataset.datas[index]

    feature_vector_data = output_data.feature_vector.cpu().numpy().flatten()
    feature_variable_data = output_data.feature_variable.cpu().numpy().flatten()
    target_variable_data = output_data.target_variable.cpu().numpy().flatten()
    target_vector = output_data.target_vector.cpu().numpy()
    target_scalar = float(output_data.target_scalar.item())
    speaker_id = int(output_data.speaker_id.item())

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        st.markdown("### 固定長特徴ベクトル")
        _show_line_plot(feature_vector_data, "固定長特徴ベクトル")
    with plot_col2:
        st.markdown("### 可変長特徴データ")
        _show_line_plot(feature_variable_data, "可変長特徴データ")

    data_col1, data_col2 = st.columns(2)
    with data_col1:
        st.markdown("### サンプリングデータ")
        st.dataframe(target_vector.reshape(1, -1))
    with data_col2:
        st.markdown("### その他の値")
        st.markdown(f"**回帰ターゲット**: {target_scalar:.6f}")
        st.markdown(f"**話者ID**: {speaker_id}")

    st.markdown("### 可変長ターゲット")
    _show_line_plot(target_variable_data, "可変長ターゲット")

    target_predictor_path = _to_predictor_path(predictor_path_str)
    if target_predictor_path is not None and not target_predictor_path.exists():
        st.markdown("## 推論結果")
        st.error(f"モデルファイルが見つかりません: {target_predictor_path}")
    elif target_predictor_path is not None:
        generator = _load_generator(str(target_config_path), str(target_predictor_path))
        generator_output = _run_inference(generator, output_data)
        vector_output = generator_output.vector_output[0].cpu().numpy()
        variable_output = generator_output.variable_output[0].cpu().numpy()
        scalar_output = float(generator_output.scalar_output[0].item())

        predicted_class = int(vector_output.argmax())
        target_class = int(output_data.target_vector.item())

        st.markdown("## 推論結果")
        infer_col1, infer_col2 = st.columns(2)
        with infer_col1:
            st.markdown("### 固定長ベクトル出力")
            st.dataframe(vector_output.reshape(1, -1))
            st.markdown(
                f"**クラス比較**: 予測クラス {predicted_class} / 正解クラス {target_class}"
            )
        with infer_col2:
            st.markdown("### スカラー出力")
            st.markdown(f"**予測 scalar_output**: {scalar_output:.6f}")
            st.markdown(f"**正解 target_scalar**: {target_scalar:.6f}")

        st.markdown("### 可変長出力")
        _show_pred_target_plot(variable_output.flatten(), target_variable_data)

    st.markdown("---")
    st.markdown("### 詳細情報")
    st.code(
        _create_details_text(target_config_path, output_data, lazy_data),
        language=None,
    )


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
