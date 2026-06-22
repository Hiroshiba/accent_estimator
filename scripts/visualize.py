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
import torch
from matplotlib.figure import Figure
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


@dataclass
class DataInfo:
    """データ情報"""

    consonant: list[str]
    vowel: list[str]
    accent: np.ndarray
    speaker_id: int
    details: str


@dataclass
class FigureState:
    """図の状態"""

    feature_fig: Figure | None = None
    accent_fig: Figure | None = None
    accent_pred_fig: Figure | None = None


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

    def _get_generator(self, config_path: UPath, predictor_path: UPath) -> Generator:
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
            wave=batch.wave,
            phoneme_index=batch.phoneme_index,
            phoneme_id=batch.phoneme_id,
            vowel_index=batch.vowel_index,
            mora_f0=batch.mora_f0,
            wave_length=batch.wave_length,
            phoneme_length=batch.phoneme_length,
            mora_length=batch.mora_length,
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

音声波形
パス: {lazy_data.wave_path}
shape: {tuple(output_data.wave.shape)}

母音位置列
パス: {lazy_data.phoneme_list_path}
shape: {tuple(output_data.vowel_index.shape)}

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
"""

    def _setup_feature_plot(self, data: np.ndarray) -> Figure:
        if self.figure_state.feature_fig is not None:
            plt.close(self.figure_state.feature_fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(data.T, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title("フレーム特徴量")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Feature")
        fig.colorbar(im, ax=ax)

        self.figure_state.feature_fig = fig
        return fig

    def _setup_accent_plot(self, data: np.ndarray) -> Figure:
        if self.figure_state.accent_fig is not None:
            plt.close(self.figure_state.accent_fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title("アクセント")
        ax.set_xlabel("Mora")
        ax.set_yticks(range(len(accent_channel_names)))
        ax.set_yticklabels(accent_channel_names)
        fig.colorbar(im, ax=ax)

        self.figure_state.accent_fig = fig
        return fig

    def _setup_accent_pred_plot(self, data: np.ndarray) -> Figure:
        if self.figure_state.accent_pred_fig is not None:
            plt.close(self.figure_state.accent_pred_fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title("予測アクセント確率")
        ax.set_xlabel("Mora")
        ax.set_yticks(range(len(accent_channel_names)))
        ax.set_yticklabels(accent_channel_names)
        fig.colorbar(im, ax=ax)

        self.figure_state.accent_pred_fig = fig
        return fig

    def _setup_plots(self, output_data: OutputData) -> tuple[Figure, Figure]:
        """プロットを作成または更新"""
        feature_data = output_data.wave.cpu().numpy()
        accent_data = output_data.accent.cpu().numpy()

        feature_plot = self._setup_feature_plot(feature_data)
        accent_plot = self._setup_accent_plot(accent_data)

        return (feature_plot, accent_plot)

    def _extract_mora_phoneme(
        self, lazy_data: LazyInputData
    ) -> tuple[list[str], list[str]]:
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

    def _create_data_info(
        self,
        config_path: UPath,
        output_data: OutputData,
        lazy_data: LazyInputData,
    ) -> DataInfo:
        """データ情報を作成"""
        consonant, vowel = self._extract_mora_phoneme(lazy_data)
        accent = output_data.accent.cpu().numpy()
        speaker_id = int(output_data.speaker_id.item())
        details = self._create_details_text(config_path, output_data, lazy_data)

        return DataInfo(
            consonant=consonant,
            vowel=vowel,
            accent=accent,
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

                feature_plot, accent_plot = self._setup_plots(output_data)
                data_info = self._create_data_info(config_path, output_data, lazy_data)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### フレーム特徴量")
                        gr.Plot(value=feature_plot, label="feature")

                    with gr.Column():
                        gr.Markdown("### アクセント")
                        gr.Plot(value=accent_plot, label="accent")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### モーラ列")
                        mora_table = {"": ["子音", "母音"]}
                        for i, (c, v) in enumerate(
                            zip(data_info.consonant, data_info.vowel, strict=True)
                        ):
                            mora_table[str(i)] = [c, v]
                        gr.DataFrame(
                            value=pd.DataFrame(mora_table),
                            label="mora",
                        )

                    with gr.Column():
                        gr.Markdown("### その他の値")
                        gr.Textbox(
                            value=str(data_info.speaker_id),
                            label="話者ID",
                            interactive=False,
                        )

                predictor_path = _to_predictor_path(predictor_path_str)
                if predictor_path is not None and not predictor_path.exists():
                    gr.Markdown("## 推論結果")
                    gr.Markdown(f"モデルファイルが見つかりません: {predictor_path}")
                elif predictor_path is not None:
                    generator = self._get_generator(config_path, predictor_path)
                    generator_output = self._run_inference(generator, output_data)

                    accent_logit = generator_output.accent_logit[0]  # (max(mL), 2, 4)
                    mora_length = int(generator_output.mora_length[0].item())
                    accent_prob = (
                        torch.softmax(accent_logit[:mora_length], dim=1)[:, 1, :]
                        .cpu()
                        .numpy()
                    )  # (mL, 4)
                    accent_pred_plot = self._setup_accent_pred_plot(accent_prob)

                    gr.Markdown("## 推論結果")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 予測アクセント確率")
                            gr.Plot(value=accent_pred_plot, label="accent_pred")

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
