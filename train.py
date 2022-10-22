import argparse
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from accent_estimator.config import Config
from accent_estimator.dataset import create_dataset
from accent_estimator.model import Model, ModelOutput
from accent_estimator.network.predictor import create_predictor
from accent_estimator.utility.pytorch_utility import (
    collate_list,
    detach_cpu,
    init_weights,
    make_optimizer,
    make_scheduler,
    to_device,
)
from accent_estimator.utility.train_utility import Logger, SaveManager


def train(config_yaml_path: Path, output_dir: Path):
    # config
    with config_yaml_path.open() as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_dict(config_dict)
    config.add_git_info()

    # model
    predictor = create_predictor(config.network)
    model = Model(model_config=config.model, predictor=predictor)
    if config.train.weight_initializer is not None:
        init_weights(model, name=config.train.weight_initializer)

    device = "cuda" if config.train.use_gpu else "cpu"
    model.to(device)
    model.train()

    # dataset
    datasets = create_dataset(config.dataset)

    num_workers = multiprocessing.cpu_count()
    if config.train.num_processes is not None:
        num_workers = config.train.num_processes

    def _create_loader(dataset, for_train: bool, for_eval: bool):
        batch_size = (
            config.train.eval_batch_size if for_eval else config.train.batch_size
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_list,
            pin_memory=config.train.use_gpu,
            drop_last=for_train,
            timeout=0 if num_workers == 0 else 60,
        )

    datasets = create_dataset(config.dataset)
    train_loader = _create_loader(datasets["train"], for_train=True, for_eval=False)
    test_loader = _create_loader(datasets["test"], for_train=False, for_eval=False)
    eval_loader = _create_loader(datasets["test"], for_train=False, for_eval=True)

    valid_loader = None
    if datasets["valid"] is not None:
        valid_loader = _create_loader(datasets["valid"], for_eval=True)

    # optimizer
    optimizer = make_optimizer(config_dict=config.train.optimizer, model=model)

    # logger
    logger = Logger(
        config_dict=config_dict,
        project_category=config.project.category,
        project_name=config.project.name,
        output_dir=output_dir,
    )

    # snapshot
    snapshot_path = output_dir / "snapshot.pth"
    if not snapshot_path.exists():
        iteration = -1
        epoch = -1
    else:
        snapshot = torch.load(snapshot_path, map_location=device)

        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        logger.load_state_dict(snapshot["logger"])

        iteration = snapshot["iteration"]
        epoch = snapshot["epoch"]
        print(f"Loaded snapshot from {snapshot_path} (epoch: {epoch})")

    # scheduler
    scheduler = None
    if config.train.scheduler is not None:
        scheduler = make_scheduler(
            config_dict=config.train.scheduler,
            optimizer=optimizer,
            last_epoch=iteration,
        )

    # save
    save_manager = SaveManager(
        predictor=predictor,
        prefix="predictor_",
        output_dir=output_dir,
        top_num=config.train.model_save_num,
        last_num=config.train.model_save_num,
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # loop
    assert config.train.eval_epoch % config.train.log_epoch == 0
    assert config.train.snapshot_epoch % config.train.eval_epoch == 0

    for _ in range(config.train.stop_epoch):
        epoch += 1
        if epoch == config.train.stop_epoch:
            break

        train_results: List[ModelOutput] = []
        for batch in train_loader:
            iteration += 1

            batch = to_device(batch, device, non_blocking=True)
            result: ModelOutput = model(batch)

            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()

            scheduler.step()

            train_results.append(detach_cpu(result))

        def reduce_result(results: List[ModelOutput]):
            result: Dict[str, Any] = {}
            sum_data_num = sum([r["data_num"] for r in results])
            for key in set(results[0].keys()) - {"data_num"}:
                values = [r[key] * r["data_num"] for r in results]
                if isinstance(values[0], Tensor):
                    result[key] = torch.stack(values).sum() / sum_data_num
                else:
                    result[key] = sum(values) / sum_data_num
            return result

        if epoch % config.train.log_epoch == 0:
            model.eval()

            test_results: List[ModelOutput] = []
            for batch in test_loader:
                batch = to_device(batch, device, non_blocking=True)
                with torch.inference_mode():
                    result = model(batch)
                test_results.append(detach_cpu(result))

            summary = {
                "train": reduce_result(train_results),
                "test": reduce_result(test_results),
                "iteration": iteration,
                "lr": optimizer.param_groups[0]["lr"],
            }
            logger.log(summary=summary, step=epoch)

            if epoch % config.train.snapshot_epoch == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "logger": logger.state_dict(),
                        "iteration": iteration,
                        "epoch": epoch,
                    },
                    snapshot_path,
                )

                save_manager.save(
                    value=summary["test"]["loss"], step=epoch, judge="min"
                )

            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    train(**vars(parser.parse_args()))
