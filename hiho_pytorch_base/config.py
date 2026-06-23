"""機械学習プロジェクトの設定モジュール"""

from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field
from upath import UPath

from .utility.git_utility import get_branch_name, get_commit_id
from .utility.upath_utility import UPathField, to_local_path


class _Model(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DataFileConfig(_Model):
    """データファイルの設定"""

    feature_vector_pathlist_path: UPathField
    feature_variable_pathlist_path: UPathField
    target_vector_pathlist_path: UPathField
    target_variable_pathlist_path: UPathField
    target_scalar_pathlist_path: UPathField
    speaker_dict_path: UPathField
    root_dir: UPathField | None


class DatasetConfig(_Model):
    """データセット全体の設定"""

    train: DataFileConfig
    valid: DataFileConfig | None = None
    hdf5_cache_dir: UPathField | None = None
    train_num: int | None = None
    test_num: int
    valid_num: int | None = None
    eval_for_test: bool
    eval_times_num: int = 1
    seed: int = 0
    frame_rate: float
    frame_length: int


class NetworkConfig(_Model):
    """ニューラルネットワークの設定"""

    feature_vector_size: int
    feature_variable_size: int
    hidden_size: int
    target_vector_size: int
    conformer_block_num: int
    conformer_dropout_rate: float
    speaker_size: int
    speaker_embedding_size: int


class ModelConfig(_Model):
    """モデルの設定"""

    pass


class TrainConfig(_Model):
    """学習の設定"""

    batch_size: int
    gradient_accumulation: int = 1
    eval_batch_size: int
    log_epoch: int
    eval_epoch: int
    snapshot_epoch: int
    stop_epoch: int
    model_save_num: int
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None = None
    weight_initializer: str | None = None
    pretrained_predictor_path: UPathField | None = None
    prefetch_workers: int = 256
    preprocess_workers: int | None = None
    use_gpu: bool = True
    use_amp: bool = True


class ProjectConfig(_Model):
    """プロジェクトの設定"""

    name: str
    tags: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None


class Config(_Model):
    """機械学習の全設定"""

    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """辞書から設定オブジェクトを作成"""
        backward_compatible(d)
        return cls.model_validate(d)

    @staticmethod
    def load(config_path: UPath) -> "Config":
        """設定ファイルから読み込む"""
        return Config.from_dict(yaml.safe_load(to_local_path(config_path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return self.model_dump(mode="json")

    def save(self, config_path: UPath) -> None:
        """設定ファイルに保存する"""
        config_path.write_text(yaml.safe_dump(self.to_dict()))

    def validate_config(self) -> None:
        """設定の妥当性を検証"""
        assert self.train.eval_epoch % self.train.log_epoch == 0

    def add_git_info(self) -> None:
        """Git情報をプロジェクトタグに追加"""
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: dict[str, Any]) -> None:
    """設定の後方互換性を保つための変換"""
    pass
