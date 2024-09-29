from copy import deepcopy
from typing import Any, Callable, Dict

import numpy
import torch
import torch_optimizer
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


def init_weights(model: torch.nn.Module, name: str):
    def _init_weights(layer: nn.Module):
        initializer: Callable
        if name == "uniform":
            initializer = torch.nn.init.uniform_
        elif name == "normal":
            initializer = torch.nn.init.normal_
        elif name == "xavier_uniform":
            initializer = torch.nn.init.xavier_uniform_
        elif name == "xavier_normal":
            initializer = torch.nn.init.xavier_normal_
        elif name == "kaiming_uniform":
            initializer = torch.nn.init.kaiming_uniform_
        elif name == "kaiming_normal":
            initializer = torch.nn.init.kaiming_normal_
        elif name == "orthogonal":
            initializer = torch.nn.init.orthogonal_
        elif name == "sparse":
            initializer = torch.nn.init.sparse_
        else:
            raise ValueError(name)

        for key, param in layer.named_parameters():
            if "weight" in key:
                try:
                    initializer(param)
                except:
                    pass

    model.apply(_init_weights)


def make_optimizer(config_dict: Dict[str, Any], model: nn.Module):
    cp: Dict[str, Any] = deepcopy(config_dict)
    n = cp.pop("name").lower()

    optimizer: Optimizer
    if n == "adam":
        optimizer = optim.Adam(model.parameters(), **cp)
    elif n == "radam":
        optimizer = torch_optimizer.RAdam(model.parameters(), **cp)
    elif n == "ranger":
        optimizer = torch_optimizer.Ranger(model.parameters(), **cp)
    elif n == "sgd":
        optimizer = optim.SGD(model.parameters(), **cp)
    elif n == "true_adamw":
        cp["weight_decay"] /= cp["lr"]
        optimizer = optim.AdamW(model.parameters(), **cp)
    else:
        raise ValueError(n)

    return optimizer


class WarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps**0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]


def make_scheduler(config_dict: Dict[str, Any], optimizer: Optimizer, last_epoch: int):
    cp: Dict[str, Any] = deepcopy(config_dict)
    n = cp.pop("name").lower()

    scheduler: optim.lr_scheduler._LRScheduler
    if n == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **cp)
    elif n == "warmup":
        scheduler = WarmupLR(optimizer, last_epoch=last_epoch, **cp)
    else:
        raise ValueError(n)

    return scheduler


def detach_cpu(data):
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    elif isinstance(data, numpy.ndarray):
        return torch.as_tensor(data)
    elif isinstance(data, dict):
        try:
            return elem_type({key: detach_cpu(data[key]) for key in data})
        except TypeError:
            return {key: detach_cpu(data[key]) for key in data}
    elif isinstance(data, (list, tuple)):
        try:
            return elem_type([detach_cpu(d) for d in data])
        except TypeError:
            return [detach_cpu(d) for d in data]
    else:
        return data


def to_device(batch, device, non_blocking=False):
    if isinstance(batch, dict):
        return {
            key: to_device(value, device, non_blocking) for key, value in batch.items()
        }
    elif isinstance(batch, (list, tuple)):
        return type(batch)(to_device(value, device, non_blocking) for value in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    else:
        return batch


def collate_list(batch):
    if not batch:
        raise ValueError("batch is empty")

    first_elem = batch[0]

    if isinstance(first_elem, dict):
        result = {}
        for key in first_elem:
            result[key] = [example[key] for example in batch]
        return result

    else:
        raise ValueError(type(first_elem))
