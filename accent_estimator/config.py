from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from accent_estimator.utility import dataclass_utility
from accent_estimator.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetFileConfig:
    f0_glob: str
    phoneme_list_glob: str
    volume_glob: str
    accent_start_glob: str
    accent_end_glob: str
    accent_phrase_start_glob: str
    accent_phrase_end_glob: str


@dataclass
class DatasetConfig:
    train_file: DatasetFileConfig
    valid_file: DatasetFileConfig
    frame_rate: float
    test_num: int
    seed: int = 0


@dataclass
class NetworkConfig:
    phoneme_size: int
    phoneme_embedding_size: int
    hidden_size: int
    encoder_block_num: int
    attention_heads: int
    decoder_block_num: int
    post_layer_num: int


@dataclass
class ModelConfig:
    pass


@dataclass
class TrainConfig:
    batch_size: int
    eval_batch_size: Optional[int]
    log_epoch: int
    eval_epoch: int
    snapshot_epoch: int
    stop_epoch: int
    model_save_num: int
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    weight_initializer: Optional[str] = None
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
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    pass
