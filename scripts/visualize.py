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
    target_variable_fig: Figure | None = None
    target_variable_line: Line2D | None = None
    variable_output_fig: Figure | None = None
    variable_output_pred_line: Line2D | None = None
    variable_output_target_line: Line2D | None = None


class VisualizationApp:
    """可視化アプリケーション"""

    def __init__(
        self,
        initial_config_path: UPath | None,
        initial_dataset_type: DatasetType,
        initial_predictor_path: UPath | None,
    ):
        self.initial_config_path = initial_config_path
        self.initial_dataset_type = initial_dataset_type
        self.initial_predictor_path = initial_predictor_path
        self._config_dataset_cache: tuple[str, Config, DatasetCollection] | None = None
        self._generator_cache: tuple[str, str, Generator] | None = None
        self.figure_state = FigureState()

    def _get_config_and_dataset(
        self, config_path: UPath
    ) -> tuple[Config, DatasetCollection]:
        """configパスからConfigとDatasetCollectionを取得し、同じパスならキャッシュを再利用する"""
        cache_key = str(config_path)
        cache = self._config_dataset_cache
        if cache is not None and cache[0] == cache_key:
            return cache[1], cache[2]
        config = Config.load(config_path)
        dataset_collection = create_dataset(config.dataset)
        self._config_dataset_cache = (cache_key, config, dataset_collection)
        return config, dataset_collection

    def _create_generator(self, config: Config, predictor_path: UPath) -> Generator:
        """推論用のGeneratorを作成"""
        return Generator(
            config=config,
            predictor=to_local_path(predictor_path),
            use_gpu=False,
        )

    def _get_generator(
        self, config_path: UPath, predictor_path: UPath
    ) -> Generator:
        """configパスとpredictorパスからGeneratorを取得し、同じパスならキャッシュを再利用する"""
        config_key = str(config_path)
        predictor_key = str(predictor_path)
        cache = self._generator_cache
        if cache is not None and cache[0] == config_key and cache[1] == predictor_key:
            return cache[2]
        config, _ = self._get_config_and_dataset(config_path)
        generator = self._create_generator(config, predictor_path)
        self._generator_cache = (config_key, predictor_key, generator)
        return generator

    def _run_inference(
        self, generator: Generator, output_data: OutputData
    ) -> GeneratorOutput:
        """OutputDataから推論結果を生成"""
        batch = collate_dataset_output([output_data])
        return generator(
            feature_vector=batch.feature_vector,
            feature_variable=batch.feature_variable,
            speaker_id=batch.speaker_id,
            length=batch.length,
        )

    def _get_output_data(
        self,
        dataset_collection: DatasetCollection,
        index: int,
        dataset_type: DatasetType,
    ) -> OutputData:
        """前処理済みのOutputDataを取得"""
        dataset = dataset_collection.get(dataset_type)
        return dataset[index]

    def _get_lazy_data(
        self,
        dataset_collection: DatasetCollection,
        index: int,
        dataset_type: DatasetType,
    ) -> LazyInputData:
        """遅延読み込み用のLazyInputDataを取得"""
        dataset = dataset_collection.get(dataset_type)
        return dataset.datas[index]

    def _create_details_text(
        self,
        config_path: UPath,
        output_data: OutputData,
        lazy_data: LazyInputData,
    ) -> str:
        """詳細情報テキストを作成"""
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

    def _setup_target_variable_plot(self, data: np.ndarray) -> Figure:
        if (
            self.figure_state.target_variable_fig is None
            or self.figure_state.target_variable_line is None
        ):
            self.figure_state.target_variable_fig, ax = plt.subplots(figsize=(10, 4))
            x_data = range(len(data))
            (self.figure_state.target_variable_line,) = ax.plot(x_data, data)
            ax.set_title("可変長ターゲット")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.grid(True)
        else:
            x_data = range(len(data))
            self.figure_state.target_variable_line.set_data(x_data, data)
            ax = self.figure_state.target_variable_fig.gca()
            ax.relim()
            ax.autoscale_view()
            self.figure_state.target_variable_fig.canvas.draw()

        return self.figure_state.target_variable_fig

    def _setup_variable_output_plot(
        self, pred_data: np.ndarray, target_data: np.ndarray
    ) -> Figure:
        if (
            self.figure_state.variable_output_fig is None
            or self.figure_state.variable_output_pred_line is None
            or self.figure_state.variable_output_target_line is None
        ):
            self.figure_state.variable_output_fig, ax = plt.subplots(figsize=(10, 4))
            (self.figure_state.variable_output_pred_line,) = ax.plot(
                range(len(pred_data)), pred_data, label="予測"
            )
            (self.figure_state.variable_output_target_line,) = ax.plot(
                range(len(target_data)), target_data, label="正解"
            )
            ax.set_title("可変長出力の予測と正解")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
        else:
            self.figure_state.variable_output_pred_line.set_data(
                range(len(pred_data)), pred_data
            )
            self.figure_state.variable_output_target_line.set_data(
                range(len(target_data)), target_data
            )
            ax = self.figure_state.variable_output_fig.gca()
            ax.relim()
            ax.autoscale_view()
            self.figure_state.variable_output_fig.canvas.draw()

        return self.figure_state.variable_output_fig

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
        self,
        config_path: UPath,
        output_data: OutputData,
        lazy_data: LazyInputData,
    ) -> DataInfo:
        """データ情報を作成"""
        target_vector = output_data.target_vector.cpu().numpy()
        target_scalar = float(output_data.target_scalar.item())
        speaker_id = int(output_data.speaker_id.item())
        details = self._create_details_text(config_path, output_data, lazy_data)

        return DataInfo(
            target_vector=target_vector,
            target_scalar=target_scalar,
            speaker_id=speaker_id,
            details=details,
        )

    def launch(self) -> None:
        """Gradio UIを起動"""
        initial_config_path_str = (
            str(self.initial_config_path)
            if self.initial_config_path is not None
            else ""
        )
        initial_predictor_path_str = (
            str(self.initial_predictor_path)
            if self.initial_predictor_path is not None
            else ""
        )

        if self.initial_config_path is not None:
            _, initial_dataset_collection = self._get_config_and_dataset(
                self.initial_config_path
            )
            initial_dataset = initial_dataset_collection.get(self.initial_dataset_type)
            initial_max_index = len(initial_dataset) - 1
        else:
            initial_max_index = 0

        with gr.Blocks() as demo:
            # 状態管理
            current_index = gr.State(0)
            current_dataset_type = gr.State(self.initial_dataset_type)
            current_predictor_path = gr.State(initial_predictor_path_str)
            current_config_path = gr.State(initial_config_path_str)

            # UI コンポーネント
            with gr.Row():
                config_path_textbox = gr.Textbox(
                    value=initial_config_path_str,
                    label="設定ファイルパス",
                    placeholder="/path/to/config.yaml",
                )
                predictor_path_textbox = gr.Textbox(
                    value=initial_predictor_path_str,
                    label="モデルファイルパス",
                    placeholder="/path/to/predictor.pth",
                )

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

            @gr.render(
                inputs=[
                    current_index,
                    current_dataset_type,
                    current_predictor_path,
                    current_config_path,
                ]
            )
            def render_content(
                index: int,
                dataset_type: DatasetType,
                predictor_path_str: str,
                config_path_str: str,
            ) -> None:
                config_path = _to_config_path(config_path_str)
                if config_path is None:
                    gr.Markdown("設定ファイルパスを指定してください")
                    return
                if not config_path.exists():
                    gr.Markdown(f"設定ファイルが見つかりません: {config_path}")
                    return

                _, dataset_collection = self._get_config_and_dataset(config_path)
                output_data = self._get_output_data(
                    dataset_collection, index, dataset_type
                )
                lazy_data = self._get_lazy_data(dataset_collection, index, dataset_type)

                feature_vector_plot, feature_variable_plot = self._setup_plots(
                    output_data
                )
                target_variable_data = (
                    output_data.target_variable.cpu().numpy().flatten()
                )
                target_variable_plot = self._setup_target_variable_plot(
                    target_variable_data
                )
                data_info = self._create_data_info(config_path, output_data, lazy_data)

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

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 可変長ターゲット")
                        gr.Plot(value=target_variable_plot, label="target_variable")

                predictor_path = _to_predictor_path(predictor_path_str)
                if predictor_path is not None and not predictor_path.exists():
                    gr.Markdown("## 推論結果")
                    gr.Markdown(f"モデルファイルが見つかりません: {predictor_path}")
                elif predictor_path is not None:
                    generator = self._get_generator(config_path, predictor_path)
                    generator_output = self._run_inference(generator, output_data)
                    vector_output = generator_output.vector_output[0].cpu().numpy()
                    variable_output = generator_output.variable_output[0].cpu().numpy()
                    scalar_output = float(generator_output.scalar_output[0].item())

                    predicted_class = int(vector_output.argmax())
                    target_class = int(output_data.target_vector.item())
                    variable_output_plot = self._setup_variable_output_plot(
                        variable_output.flatten(), target_variable_data
                    )

                    gr.Markdown("## 推論結果")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 固定長ベクトル出力")
                            gr.DataFrame(
                                value=pd.DataFrame(vector_output.reshape(1, -1)),
                                label="vector_output",
                            )
                            gr.Textbox(
                                value=f"予測クラス {predicted_class} / 正解クラス {target_class}",
                                label="クラス比較",
                                interactive=False,
                            )
                        with gr.Column():
                            gr.Markdown("### スカラー出力")
                            gr.Textbox(
                                value=f"{scalar_output:.6f}",
                                label="予測 scalar_output",
                                interactive=False,
                            )
                            gr.Textbox(
                                value=f"{data_info.target_scalar:.6f}",
                                label="正解 target_scalar",
                                interactive=False,
                            )
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 可変長出力")
                            gr.Plot(value=variable_output_plot, label="variable_output")

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
                index: int, dataset_type: DatasetType, config_path_str: str
            ) -> tuple[int, Any]:
                config_path = _to_config_path(config_path_str)
                if config_path is None or not config_path.exists():
                    return (index, gr.update(maximum=0))
                _, dataset_collection = self._get_config_and_dataset(config_path)
                dataset = dataset_collection.get(dataset_type)
                max_index = len(dataset) - 1
                return (
                    index,
                    gr.update(maximum=max_index),
                )

            def on_config_path_change(
                config_path_str: str, dataset_type: DatasetType
            ) -> tuple[int, Any]:
                config_path = _to_config_path(config_path_str)
                if config_path is None or not config_path.exists():
                    return (0, gr.update(maximum=0, value=0))
                _, dataset_collection = self._get_config_and_dataset(config_path)
                dataset = dataset_collection.get(dataset_type)
                max_index = len(dataset) - 1
                return (0, gr.update(maximum=max_index, value=0))

            current_index.change(
                sync_slider_from_state,
                inputs=[current_index, current_dataset_type, current_config_path],
                outputs=[index_slider, index_slider],
            )

            current_dataset_type.change(
                sync_slider_from_state,
                inputs=[current_index, current_dataset_type, current_config_path],
                outputs=[index_slider, index_slider],
            )

            current_config_path.change(
                on_config_path_change,
                inputs=[current_config_path, current_dataset_type],
                outputs=[current_index, index_slider],
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

            predictor_path_textbox.submit(
                lambda new_path: new_path,
                inputs=[predictor_path_textbox],
                outputs=[current_predictor_path],
            )

            config_path_textbox.submit(
                lambda new_path: new_path,
                inputs=[config_path_textbox],
                outputs=[current_config_path],
            )

            # 初期化
            demo.load(
                lambda: (0, self.initial_dataset_type, initial_config_path_str),
                outputs=[current_index, current_dataset_type, current_config_path],
            )

        demo.launch()


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
    """指定されたデータセットをGradio UIで可視化する"""
    app = VisualizationApp(config_path, dataset_type, predictor_path)
    app.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセットのビジュアライゼーション")
    parser.add_argument(
        "--config_path",
        type=UPath,
        help="設定ファイルのパス",
    )
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
