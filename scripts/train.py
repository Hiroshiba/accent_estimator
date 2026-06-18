"""機械学習モデルの学習メインスクリプト"""

import argparse
import os
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset, Sampler
from upath import UPath

from hiho_pytorch_base.batch import BatchOutput, collate_dataset_output
from hiho_pytorch_base.config import Config
from hiho_pytorch_base.dataset import DatasetType, create_dataset, prefetch_datas
from hiho_pytorch_base.evaluator import (
    Evaluator,
    EvaluatorOutput,
    calculate_value,
)
from hiho_pytorch_base.generator import Generator
from hiho_pytorch_base.model import Model, ModelOutput
from hiho_pytorch_base.network.predictor import create_predictor
from hiho_pytorch_base.utility.pytorch_utility import (
    init_weights,
    make_optimizer,
    make_scheduler,
)
from hiho_pytorch_base.utility.train_utility import (
    DataNumProtocol,
    Logger,
    SaveManager,
    reduce_result,
)


def _delete_data_num(output: DataNumProtocol) -> dict[str, Any]:
    if not hasattr(output, "data_num"):
        raise ValueError("Output does not have 'data_num' attribute")
    return {k: v for k, v in asdict(output).items() if k != "data_num"}


@dataclass
class TrainingResults:
    """学習結果"""

    train: ModelOutput

    def to_summary_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書を生成"""
        return {DatasetType.TRAIN.value: _delete_data_num(self.train)}


@dataclass
class EvaluationResults:
    """評価結果"""

    test: ModelOutput
    eval: EvaluatorOutput | None
    valid: EvaluatorOutput | None

    def to_summary_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書を生成"""
        summary = {
            DatasetType.TEST.value: _delete_data_num(self.test),
        }
        if self.eval is not None:
            summary[DatasetType.EVAL.value] = _delete_data_num(self.eval)
        if self.valid is not None:
            summary[DatasetType.VALID.value] = _delete_data_num(self.valid)
        return summary


@dataclass
class TrainingContext:
    """学習に必要な全てのオブジェクトをまとめる"""

    config: Config
    train_loader: DataLoader
    test_loader: DataLoader
    eval_loader: DataLoader | None
    valid_loader: DataLoader | None
    model: Model
    evaluator: Evaluator
    optimizer: torch.optim.Optimizer
    scaler: GradScaler
    logger: Logger
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    save_manager: SaveManager
    device: str
    epoch: int
    iteration: int
    snapshot_path: UPath
    close_prefetch: Callable[[], None]


class FirstEpochOrderedSampler(Sampler[int]):
    """初回エポックは指定順序、以降はランダムサンプリング。prefetchに有効。"""

    def __init__(self, first_indices: list[int]) -> None:
        self.first_indices = first_indices
        self.first_epoch = True

    def __iter__(self) -> Iterator[int]:  # noqa: D105
        if self.first_epoch:
            self.first_epoch = False
            return iter(self.first_indices)
        else:
            indices_tensor = torch.tensor(self.first_indices)
            return iter(indices_tensor[torch.randperm(len(indices_tensor))].tolist())

    def __len__(self) -> int:  # noqa: D105
        return len(self.first_indices)


def create_data_loader(
    config: Config,
    dataset: Dataset,
    for_train: bool,
    for_eval: bool,
    first_indices: list[int] | None,
) -> DataLoader:
    """DataLoaderを作成"""
    batch_size = config.train.eval_batch_size if for_eval else config.train.batch_size

    if first_indices is not None:
        sampler = FirstEpochOrderedSampler(first_indices)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    num_workers = config.train.preprocess_workers
    if num_workers is None:
        num_workers = os.cpu_count()
        if num_workers is None:
            raise ValueError("Failed to get CPU count")

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_dataset_output,
        pin_memory=config.train.use_gpu,
        drop_last=for_train,
        timeout=0 if num_workers == 0 else 30,
        persistent_workers=num_workers > 0,
    )


def setup_training_context(
    config_yaml_path: UPath, output_dir: UPath
) -> TrainingContext:
    """TrainingContextを作成"""
    # config
    config = Config.load(config_yaml_path)
    config.add_git_info()
    config.validate_config()

    # dataset
    datasets = create_dataset(config.dataset)

    # prefetch
    train_indices = torch.randperm(len(datasets.train)).tolist()
    close_prefetch = prefetch_datas(
        train_datas=datasets.train.datas,
        test_datas=datasets.test.datas,
        valid_datas=datasets.valid.datas if datasets.valid is not None else None,
        train_indices=train_indices,
        train_batch_size=config.train.batch_size,
        num_prefetch=config.train.prefetch_workers,
    )

    # data loader
    train_loader = create_data_loader(
        config,
        datasets.train,
        for_train=True,
        for_eval=False,
        first_indices=train_indices,
    )
    test_loader = create_data_loader(
        config, datasets.test, for_train=False, for_eval=False, first_indices=None
    )
    eval_loader = (
        create_data_loader(
            config, datasets.eval, for_train=False, for_eval=True, first_indices=None
        )
        if datasets.eval is not None
        else None
    )
    valid_loader = (
        create_data_loader(
            config, datasets.valid, for_train=False, for_eval=True, first_indices=None
        )
        if datasets.valid is not None
        else None
    )

    # predictor
    predictor = create_predictor(config.network)
    device = "cuda" if config.train.use_gpu else "cpu"
    if config.train.pretrained_predictor_path is not None:
        state_dict = torch.load(
            BytesIO(config.train.pretrained_predictor_path.read_bytes()),
            map_location=device,
        )
        predictor.load_state_dict(state_dict)
    print("predictor:", predictor)

    # model
    predictor_scripted: Predictor = torch.jit.script(predictor)  # type: ignore
    model = Model(model_config=config.model, predictor=predictor_scripted)
    if config.train.weight_initializer is not None:
        init_weights(model, name=config.train.weight_initializer)
    model.to(device)

    # evaluator
    generator = Generator(
        config=config, predictor=predictor_scripted, use_gpu=config.train.use_gpu
    )
    evaluator = Evaluator(generator=generator)

    # optimizer
    optimizer = make_optimizer(config_dict=config.train.optimizer, model=model)
    scaler = GradScaler(device, enabled=config.train.use_amp)

    # logger
    logger = Logger(
        config_dict=config.to_dict(),
        project_category=config.project.category,
        project_name=config.project.name,
        output_dir=output_dir,
    )

    # scheduler
    scheduler = None
    if config.train.scheduler is not None:
        scheduler = make_scheduler(
            config_dict=config.train.scheduler, optimizer=optimizer
        )

    # save
    save_manager = SaveManager(
        predictor=predictor,
        prefix="predictor_",
        output_dir=output_dir,
        top_num=config.train.model_save_num,
        last_num=config.train.model_save_num,
    )

    return TrainingContext(
        config=config,
        close_prefetch=close_prefetch,
        train_loader=train_loader,
        test_loader=test_loader,
        eval_loader=eval_loader,
        valid_loader=valid_loader,
        model=model,
        evaluator=evaluator,
        optimizer=optimizer,
        scaler=scaler,
        logger=logger,
        scheduler=scheduler,
        save_manager=save_manager,
        device=device,
        epoch=0,
        iteration=0,
        snapshot_path=output_dir / "snapshot.pth",
    )


def load_snapshot(context: TrainingContext) -> None:
    """学習状態を復元"""
    snapshot = torch.load(
        BytesIO(context.snapshot_path.read_bytes()), map_location=context.device
    )

    context.model.load_state_dict(snapshot["model"])
    context.optimizer.load_state_dict(snapshot["optimizer"])
    context.scaler.load_state_dict(snapshot["scaler"])
    context.logger.load_state_dict(snapshot["logger"])

    context.iteration = snapshot["iteration"]
    context.epoch = snapshot["epoch"]

    if context.scheduler is not None:
        context.scheduler.last_epoch = context.epoch


def train_one_epoch(context: TrainingContext) -> TrainingResults:
    """１エポックの学習処理"""
    context.model.train()
    if hasattr(context.optimizer, "train"):
        context.optimizer.train()  # type: ignore

    gradient_accumulation = context.config.train.gradient_accumulation
    context.optimizer.zero_grad()  # NOTE: 端数分の勾配を消す

    batch: BatchOutput
    train_results: list[ModelOutput] = []

    for batch_index, batch in enumerate(context.train_loader, start=1):
        with autocast(context.device, enabled=context.config.train.use_amp):
            batch = batch.to_device(context.device, non_blocking=True)
            result: ModelOutput = context.model(batch)

        loss = result.loss / gradient_accumulation
        if not loss.isfinite():
            raise ValueError("loss is NaN")

        context.scaler.scale(loss).backward()
        train_results.append(result.detach_cpu())

        if batch_index % gradient_accumulation == 0:
            context.scaler.step(context.optimizer)
            context.scaler.update()
            context.optimizer.zero_grad()
            context.iteration += 1

    if context.scheduler is not None:
        context.scheduler.step()

    return TrainingResults(train=reduce_result(train_results))


@torch.no_grad()
def evaluate(context: TrainingContext) -> EvaluationResults:
    """評価値を計算する"""
    context.model.eval()
    if hasattr(context.optimizer, "eval"):
        context.optimizer.eval()  # type: ignore

    batch: BatchOutput

    # test評価
    test_result_list: list[ModelOutput] = []
    for batch in context.test_loader:
        batch = batch.to_device(context.device, non_blocking=True)
        model_result: ModelOutput = context.model(batch)
        test_result_list.append(model_result.detach_cpu())
    test_result = reduce_result(test_result_list)

    # eval評価
    eval_result = None
    if context.eval_loader is not None:
        eval_result_list: list[EvaluatorOutput] = []
        for batch in context.eval_loader:
            batch = batch.to_device(context.device, non_blocking=True)
            evaluator_result: EvaluatorOutput = context.evaluator(batch)
            eval_result_list.append(evaluator_result.detach_cpu())
        eval_result = reduce_result(eval_result_list)

    # valid評価
    valid_result = None
    if context.valid_loader is not None:
        valid_result_list: list[EvaluatorOutput] = []
        for batch in context.valid_loader:
            batch = batch.to_device(context.device, non_blocking=True)
            evaluator_result: EvaluatorOutput = context.evaluator(batch)
            valid_result_list.append(evaluator_result.detach_cpu())
        valid_result = reduce_result(valid_result_list)

    return EvaluationResults(test=test_result, eval=eval_result, valid=valid_result)


def save_predictor(
    context: TrainingContext, evaluation_results: EvaluationResults
) -> None:
    """評価結果に基づいてPredictorを保存する"""
    if evaluation_results.valid is not None:
        evaluation_value = calculate_value(evaluation_results.valid).item()
    else:
        evaluation_value = 0

    context.save_manager.save(value=evaluation_value, step=context.epoch)


def save_snapshot(context: TrainingContext) -> None:
    """チェックポイント保存する"""
    torch.save(
        {
            "model": context.model.state_dict(),
            "optimizer": context.optimizer.state_dict(),
            "scaler": context.scaler.state_dict(),
            "logger": context.logger.state_dict(),
            "iteration": context.iteration,
            "epoch": context.epoch,
        },
        context.snapshot_path,
    )


def should_log_epoch(context: TrainingContext) -> bool:
    """ログ出力するかどうか判定する"""
    return context.epoch % context.config.train.log_epoch == 0


def should_eval_epoch(context: TrainingContext) -> bool:
    """評価実行するかどうか判定する"""
    return context.epoch % context.config.train.eval_epoch == 0


def should_snapshot_epoch(context: TrainingContext) -> bool:
    """スナップショット保存するかどうか判定する"""
    return context.epoch % context.config.train.snapshot_epoch == 0


def training_loop(context: TrainingContext) -> None:
    """学習ループ"""
    for _ in range(context.config.train.stop_epoch):
        context.epoch += 1
        if context.epoch > context.config.train.stop_epoch:
            break

        training_results = train_one_epoch(context)

        if should_log_epoch(context):
            summary = {
                "iteration": context.iteration,
                "epoch": context.epoch,
                "lr": context.optimizer.param_groups[0]["lr"],
            }
            summary.update(training_results.to_summary_dict())

            if should_eval_epoch(context):
                evaluation_results = evaluate(context)
                summary.update(evaluation_results.to_summary_dict())
                save_predictor(context, evaluation_results)

            context.logger.log(summary=summary, step=context.epoch)

        if should_snapshot_epoch(context):
            save_snapshot(context)


def train(config_yaml_path: UPath, output_dir: UPath) -> None:
    """機械学習モデルを学習する。スナップショットがあれば再開する。"""
    context = setup_training_context(config_yaml_path, output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    context.config.save(output_dir / "config.yaml")

    if context.snapshot_path.exists():
        load_snapshot(context)
    else:
        # NOTE: W&BのIDを固定するために保存
        save_snapshot(context)

    try:
        training_loop(context)
    finally:
        context.close_prefetch()
        context.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=UPath)
    parser.add_argument("output_dir", type=UPath)
    train(**vars(parser.parse_args()))
