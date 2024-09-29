import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from accent_estimator.utility import dataclass_utility
from accent_estimator.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetFileConfig:
    feature_glob: str
    phoneme_list_glob: str
    accent_start_glob: str
    accent_end_glob: str
    accent_phrase_start_glob: str
    accent_phrase_end_glob: str


@dataclass
class DatasetConfig:
    train_file: DatasetFileConfig
    # valid_file: DatasetFileConfig
    train_num: Optional[int]
    test_num: int
    seed: int = 0


@dataclass
class MMConformerConfig:
    hidden_size: int
    block_num: int
    dropout_rate: float
    positional_dropout_rate: float
    attention_head_size: int
    attention_dropout_rate: float
    use_conv_glu_module: bool
    conv_glu_module_kernel_size_a: int
    conv_glu_module_kernel_size_b: int
    feed_forward_hidden_size: int
    feed_forward_kernel_size_a: int
    feed_forward_kernel_size_b: int


@dataclass
class NetworkConfig:
    vowel_embedding_size: int
    feature_size: int
    hidden_size: int
    mm_conformer_config: MMConformerConfig


@dataclass
class ModelConfig:
    pass


@dataclass
class TrainConfig:
    batch_size: int
    eval_batch_size: int
    log_epoch: int
    eval_epoch: int
    snapshot_epoch: int
    stop_epoch: int
    model_save_num: int
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    weight_initializer: Optional[str] = None
    pretrained_predictor_path: Optional[Path] = None
    num_processes: int = 4
    use_gpu: bool = True
    use_amp: bool = True


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = copy.deepcopy(d)
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    pass
