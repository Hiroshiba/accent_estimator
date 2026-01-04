"""
機械学習データセットの可視化ツール

設定ファイルからDatasetCollectionを読み込み、データタイプごとにGradio UIで表示する。
各データタイプの表示形式（プロット、テーブル等）は機械学習タスクに応じてカスタマイズする。
データタイプに応じた可視化ロジックを_setup_*_plotや_create_data_infoで調整する。
"""

import argparse
from dataclasses import dataclass
from typing import Any

import gradio as gr
import japanize_matplotlib  # noqa: F401 日本語フォントに必須
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data.data import OutputData
from hiho_pytorch_base.dataset import (
    DatasetCollection,
    DatasetType,
    LazyInputData,
    create_dataset,
)


@dataclass
class DataInfo:
    """データ情報"""

    target_vector: np.ndarray
    target_scalar: float
    speaker_id: int
    details: str


@dataclass
class FigureState:
    """図の状態"""

    feature_vector_fig: Figure | None = None
    feature_variable_fig: Figure | None = None
    feature_vector_line: Line2D | None = None
    feature_variable_line: Line2D | None = None


class VisualizationApp:
    """可視化アプリケーション"""

    def __init__(self, config_path: UPath, initial_dataset_type: DatasetType):
        self.config_path = config_path
        self.initial_dataset_type = initial_dataset_type

        self.dataset_collection = self._create_dataset()
        self.figure_state = FigureState()

    def _create_dataset(self) -> DatasetCollection:
        """データセットを作成"""
        config = Config.load(self.config_path)
        return create_dataset(config.dataset)

    def _get_output_data(self, index: int, dataset_type: DatasetType) -> OutputData:
        """前処理済みのOutputDataを取得"""
        dataset = self.dataset_collection.get(dataset_type)
        return dataset[index]

    def _get_lazy_data(self, index: int, dataset_type: DatasetType) -> LazyInputData:
        """遅延読み込み用のLazyInputDataを取得"""
        dataset = self.dataset_collection.get(dataset_type)
        return dataset.datas[index]

    def _create_details_text(
        self, output_data: OutputData, lazy_data: LazyInputData
    ) -> str:
        """詳細情報テキストを作成"""
        return f"""
設定ファイル: {self.config_path}

固定長特徴ベクトル
パス: {lazy_data.feature_vector_path}
shape: {tuple(output_data.feature_vector.shape)}

可変長特徴データ
パス: {lazy_data.feature_variable_path}
shape: {tuple(output_data.feature_variable.shape)}

サンプリングデータ
パス: {lazy_data.target_vector_path}
shape: {tuple(output_data.target_vector.shape)}

回帰ターゲット
パス: {lazy_data.target_scalar_path}
shape: {tuple(output_data.target_scalar.shape)}

話者ID: {output_data.speaker_id.item()}
"""

    def _setup_feature_vector_plot(self, data: np.ndarray) -> Figure:
        if (
            self.figure_state.feature_vector_fig is None
            or self.figure_state.feature_vector_line is None
        ):
            self.figure_state.feature_vector_fig, ax = plt.subplots(figsize=(10, 4))
            x_data = range(len(data))
            (self.figure_state.feature_vector_line,) = ax.plot(x_data, data)
            ax.set_title("固定長特徴ベクトル")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.grid(True)
        else:
            x_data = range(len(data))
            self.figure_state.feature_vector_line.set_data(x_data, data)
            ax = self.figure_state.feature_vector_fig.gca()
            ax.relim()
            ax.autoscale_view()
            self.figure_state.feature_vector_fig.canvas.draw()

        return self.figure_state.feature_vector_fig

    def _setup_feature_variable_plot(self, data: np.ndarray) -> Figure:
        if (
            self.figure_state.feature_variable_fig is None
            or self.figure_state.feature_variable_line is None
        ):
            self.figure_state.feature_variable_fig, ax = plt.subplots(figsize=(10, 4))
            x_data = range(len(data))
            (self.figure_state.feature_variable_line,) = ax.plot(x_data, data)
            ax.set_title("可変長特徴データ")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.grid(True)
        else:
            x_data = range(len(data))
            self.figure_state.feature_variable_line.set_data(x_data, data)
            ax = self.figure_state.feature_variable_fig.gca()
            ax.relim()
            ax.autoscale_view()
            self.figure_state.feature_variable_fig.canvas.draw()

        return self.figure_state.feature_variable_fig

    def _setup_plots(self, output_data: OutputData) -> tuple[Figure, Figure]:
        """プロットを作成または更新"""
        # データの取得と整形
        feature_vector_data = output_data.feature_vector.cpu().numpy().flatten()
        feature_variable_data = output_data.feature_variable.cpu().numpy().flatten()

        # figureの更新または作成
        feature_vector_plot = self._setup_feature_vector_plot(feature_vector_data)
        feature_variable_plot = self._setup_feature_variable_plot(feature_variable_data)

        return (feature_vector_plot, feature_variable_plot)

    def _create_data_info(
        self, output_data: OutputData, lazy_data: LazyInputData
    ) -> DataInfo:
        """データ情報を作成"""
        target_vector = output_data.target_vector.cpu().numpy()
        target_scalar = float(output_data.target_scalar.item())
        speaker_id = int(output_data.speaker_id.item())
        details = self._create_details_text(output_data, lazy_data)

        return DataInfo(
            target_vector=target_vector,
            target_scalar=target_scalar,
            speaker_id=speaker_id,
            details=details,
        )

    def launch(self) -> None:
        """Gradio UIを起動"""
        initial_dataset = self.dataset_collection.get(self.initial_dataset_type)
        initial_max_index = len(initial_dataset) - 1

        with gr.Blocks() as demo:
            # 状態管理
            current_index = gr.State(0)
            current_dataset_type = gr.State(self.initial_dataset_type)

            # UI コンポーネント
            with gr.Row():
                dataset_type_dropdown = gr.Dropdown(
                    choices=list(DatasetType),
                    value=self.initial_dataset_type,
                    label="データセットタイプ",
                    scale=1,
                )
                index_slider = gr.Slider(
                    minimum=0,
                    maximum=initial_max_index,
                    value=0,
                    step=1,
                    label="サンプルインデックス",
                    scale=3,
                )

            @gr.render(inputs=[current_index, current_dataset_type])
            def render_content(index: int, dataset_type: DatasetType) -> None:
                output_data = self._get_output_data(index, dataset_type)
                lazy_data = self._get_lazy_data(index, dataset_type)

                feature_vector_plot, feature_variable_plot = self._setup_plots(
                    output_data
                )
                data_info = self._create_data_info(output_data, lazy_data)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 固定長特徴ベクトル")
                        gr.Plot(value=feature_vector_plot, label="feature_vector")

                    with gr.Column():
                        gr.Markdown("### 可変長特徴データ")
                        gr.Plot(value=feature_variable_plot, label="feature_variable")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### サンプリングデータ")
                        gr.DataFrame(
                            value=pd.DataFrame(data_info.target_vector.reshape(1, -1)),
                            label="target_vector",
                        )

                    with gr.Column():
                        gr.Markdown("### その他の値")
                        gr.Textbox(
                            value=f"{data_info.target_scalar:.6f}",
                            label="回帰ターゲット",
                            interactive=False,
                        )
                        gr.Textbox(
                            value=str(data_info.speaker_id),
                            label="話者ID",
                            interactive=False,
                        )

                gr.Markdown("---")
                gr.Textbox(
                    value=data_info.details,
                    label="詳細情報",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                )

            # 状態変更によるUI同期
            def sync_slider_from_state(
                index: int, dataset_type: DatasetType
            ) -> tuple[int, Any]:
                dataset = self.dataset_collection.get(dataset_type)
                max_index = len(dataset) - 1

                return (
                    index,  # index_slider value
                    gr.update(maximum=max_index),  # index_slider max
                )

            current_index.change(
                sync_slider_from_state,
                inputs=[current_index, current_dataset_type],
                outputs=[index_slider, index_slider],
            )

            current_dataset_type.change(
                sync_slider_from_state,
                inputs=[current_index, current_dataset_type],
                outputs=[index_slider, index_slider],
            )

            # UI操作から状態への更新
            index_slider.change(
                lambda new_index: new_index,
                inputs=[index_slider],
                outputs=[current_index],
            )

            dataset_type_dropdown.change(
                lambda new_type: (0, new_type),
                inputs=[dataset_type_dropdown],
                outputs=[current_index, current_dataset_type],
            )

            # 初期化
            demo.load(
                lambda: (0, self.initial_dataset_type),
                outputs=[current_index, current_dataset_type],
            )

        demo.launch()


def visualize(config_path: UPath, dataset_type: DatasetType) -> None:
    """指定されたデータセットをGradio UIで可視化する"""
    app = VisualizationApp(config_path, dataset_type)
    app.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセットのビジュアライゼーション")
    parser.add_argument("config_path", type=UPath, help="設定ファイルのパス")
    parser.add_argument(
        "--dataset_type", type=DatasetType, required=True, help="データセットタイプ"
    )

    args = parser.parse_args()
    visualize(config_path=args.config_path, dataset_type=args.dataset_type)
