"""データセットをチェックする"""

import argparse
import multiprocessing
import os
import traceback
from functools import partial

from torch.utils.data import DataLoader
from tqdm import tqdm
from upath import UPath

from hiho_pytorch_base.batch import collate_dataset_output
from hiho_pytorch_base.config import Config
from hiho_pytorch_base.dataset import Dataset, create_dataset


def check_dataset(config_yaml_path: UPath, trials: int, break_on_error: bool) -> None:
    """データセットの整合性をチェックする"""
    config = Config.load(config_yaml_path)

    preprocess_workers = config.train.preprocess_workers
    batch_size = config.train.batch_size
    pin_memory = config.train.use_gpu

    datasets = create_dataset(config.dataset)

    wrapper = partial(
        _check,
        preprocess_workers=preprocess_workers,
        batch_size=batch_size,
        pin_memory=pin_memory,
        break_on_error=break_on_error,
    )

    for i in range(trials):
        print(f"try {i}")
        wrapper(datasets.train, desc="train", drop_last=True)
        wrapper(datasets.test, desc="test", drop_last=False)
        if datasets.eval is not None:
            wrapper(datasets.eval, desc="eval", drop_last=False)
        if datasets.valid is not None:
            wrapper(datasets.valid, desc="valid", drop_last=False)


def _check(
    dataset,
    desc: str,
    preprocess_workers: int | None,
    batch_size: int,
    pin_memory: bool,
    drop_last: bool,
    break_on_error: bool,
) -> None:
    wrapper = partial(_wrapper, dataset=dataset)

    num_workers = preprocess_workers
    if num_workers is None:
        num_workers = os.cpu_count()
        if num_workers is None:
            raise ValueError("Failed to get CPU count")

    pool_processes = None if num_workers == 0 else num_workers
    with multiprocessing.Pool(processes=pool_processes) as pool:
        it = pool.imap_unordered(wrapper, range(len(dataset)), chunksize=2**8)
        for i, error in tqdm(it, desc=desc, total=len(dataset)):
            if error is not None:
                print(f"error at {i}")
                traceback.print_exception(type(error), error, error.__traceback__)
                if break_on_error:
                    breakpoint()
                raise error

    it = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_dataset_output,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=0 if num_workers == 0 else 15,
    )
    for _, _ in tqdm(enumerate(it), desc=desc, total=len(dataset) // batch_size):
        pass


def _wrapper(index: int, dataset: Dataset):
    try:
        dataset[index]
        return index, None
    except Exception as e:
        return index, e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=UPath)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--break_on_error", action="store_true")
    check_dataset(**vars(parser.parse_args()))
