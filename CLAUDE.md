# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、音声からアクセントを予測するタスクを扱う機械学習プロジェクトです。フレームレベルの音声特徴量と音素列を入力とし、モーラ単位でアクセント情報を推定します。PyTorchベースで多話者学習に対応しています。

学習・推論の基盤として、汎用機械学習フレームワーク hiho-pytorch-base をフォークして利用しています。

## 主なコンポーネント

以下の主要コンポーネントがあります。
`hiho_pytorch_base`内部のモジュール同士は必ず相対インポートで参照します。

### 設定管理 (`hiho_pytorch_base/config.py`)
```python
DataFileConfig:     # ファイルパス設定
DatasetConfig:      # データセット分割設定
NetworkConfig:      # ネットワーク構造設定
ModelConfig:        # モデル設定
TrainConfig:        # 学習パラメータ設定
ProjectConfig:      # プロジェクト情報設定
```

### 学習システム (`scripts/train.py`)
- PyTorch独自実装の学習ループ
- TensorBoard/W&B統合
- torch.amp（Automatic Mixed Precision）対応
- エポックベーススケジューラー対応
- スナップショット保存・復旧機能

### データ処理 (`hiho_pytorch_base/dataset.py`)
- 遅延読み込みによるメモリ効率化
- dataclassベースの型安全なデータ構造
- train/test/eval/valid の4種類データセット対応
- pathlistファイル方式によるファイル管理
- stemベース対応付けで異なるデータタイプを自動関連付け
- 多話者学習対応（JSON形式の話者マッピング）

### ネットワーク (`hiho_pytorch_base/network/predictor.py`)
- フレーム特徴のモーラ単位集約
- 母音・話者埋め込みの付加
- アクセント4種の2クラス分類

### 推論・生成
- `hiho_pytorch_base/generator.py`: 推論ジェネレーター
- `scripts/generate.py`: 推論実行スクリプト

### テストシステム
- 自動テストデータ生成
- エンドツーエンドテスト
- 統合テスト

## 使用方法

### 学習実行
```bash
uv run -m scripts.train <config_yaml_path> <output_dir>
```

### 推論実行
```bash
uv run -m scripts.generate --model_dir <model_dir> --output_dir <output_dir> [--use_gpu] [--num_files N]
```

### データセットチェック
```bash
uv run -m scripts.check_dataset <config_yaml_path> [--trials 10]
```

### テスト実行
```bash
uv run pytest tests/ -sv
```

### 開発環境セットアップ
```bash
uv sync
```

### 静的解析とフォーマット
```bash
uv run pyright && uv run ruff check --fix && uv run ruff format
```

## 技術仕様

### 設定ファイル
- **形式**: YAML
- **管理**: Pydanticによる型安全な設定

### 主な依存関係
- **Python**: 3.12+
- **PyTorch**: 2.7.1+
- **NumPy**: 2.2.5+
- **Pydantic**: 2.11.7+
- **librosa**: 0.11.0+（音声処理）
- その他詳細は`pyproject.toml`を参照

### パッケージ管理
- **uv**による高速パッケージ管理
- **pyproject.toml**ベースの依存関係管理

## Docker設計思想

このプロジェクトのDockerfileは、実行環境の提供に特化した設計を採用しています：

- **環境のみ提供**: Dockerfileは依存関係とライブラリのインストールのみを行い、学習コードや推論コードは含みません
- **Git Clone前提**: 実際の利用時は、コンテナ内でGit cloneを実行してコードを取得することを想定しています
- **音声処理対応**: libsoundfile1-dev、libasound2-dev等の音声処理ライブラリの整備方法をコメント等で案内
- **uv使用**: pyproject.tomlベースの依存関係管理にuvを使用し、高速なパッケージインストールを実現

## フォーク時の使用方法

このフレームワークはフォークして別プロジェクト名でパッケージ化することを想定しています。

### ディレクトリ構造の維持

フォーク後も `hiho_pytorch_base/` ディレクトリ名はそのまま維持してください。
ライブラリ内部は相対インポートで実装されているため、ディレクトリ名を変更する必要はありません。

### 拡張例

このフレームワークを拡張する際の参考：

1. **新しいネットワークアーキテクチャ**: `network/`ディレクトリに追加
2. **カスタム損失関数**: `model.py`の拡張
3. **異なるデータ形式**: データローダーの拡張

**注意**: フォーク前からある汎用関数の関数名やdocstringは変更してはいけません。
追従するときにコンフリクトしてしまうためです。

### パッケージ名の変更方法

パッケージ名を`hiho_pytorch_base`から必ず変更します。
フォーク先で別のパッケージ名（例: `repository_name`）として配布する場合、`pyproject.toml` を以下のように変更します：

```toml
[project]
name = "repository_name"

[tool.hatch.build.targets.wheel.sources]
"hiho_pytorch_base" = "repository_name"
```

これら以外の変更は不要です。

---

@docs/設計.md
@docs/コーディング規約.md
