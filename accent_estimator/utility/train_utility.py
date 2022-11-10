from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from typing_extensions import Literal


def _flatten_dict(dd, separator="/", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in _flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


class Logger(object):
    def __init__(
        self,
        config_dict: Dict[str, Any],
        project_category: Optional[str],
        project_name: str,
        output_dir: Path,
    ):
        self.config_dict = config_dict
        self.project_category = project_category
        self.project_name = project_name
        self.output_dir = output_dir

        self.wandb_id = None

        self.wandb = None
        self.tensorboard = None

    def _initialize(self):
        import wandb
        from torch.utils.tensorboard import SummaryWriter

        if self.wandb_id is None:
            self.wandb_id = wandb.util.generate_id()

        self.wandb = wandb.init(
            id=self.wandb_id,
            project=self.project_category,
            name=self.project_name,
            dir=self.output_dir,
            resume="allow",
        )
        self.wandb.config.update(_flatten_dict(self.config_dict), allow_val_change=True)

        self.tensorboard = SummaryWriter(log_dir=self.output_dir)

    def log(self, summary: Dict[str, Any], step: int):
        if self.wandb is None or self.tensorboard is None:
            self._initialize()

        assert self.wandb is not None
        assert self.tensorboard is not None

        flattern_summary = _flatten_dict(summary)

        self.wandb.log(flattern_summary, step=step)

        for key, value in flattern_summary.items():
            self.tensorboard.add_scalar(key, value, step)

        print(f"Step: {step}, {flattern_summary}")

    def state_dict(self):
        state_dict = {"wandb_id": self.wandb_id}
        return state_dict

    def load_state_dict(self, state_dict):
        self.wandb_id = state_dict["wandb_id"]


class SaveManager(object):
    def __init__(
        self,
        predictor: torch.nn.Module,
        prefix: str,
        output_dir: Path,
        top_num: int,
        last_num: int,
    ):
        self.predictor = predictor
        self.prefix = prefix
        self.output_dir = output_dir
        self.top_num = top_num
        self.last_num = last_num

        self.last_steps: List[int] = []
        self.top_step_values: List[Tuple[int, float]] = []

    def save(self, value: float, step: int, judge: Literal["min", "max"]):
        delete_steps: Set[int] = set()
        judged = False

        # top N
        if (
            len(self.top_step_values) < self.top_num
            or (judge == "min" and value < self.top_step_values[-1][1])
            or (judge == "max" and value > self.top_step_values[-1][1])
        ):
            self.top_step_values.append((step, value))
            self.top_step_values.sort(key=lambda x: x[1], reverse=judge == "max")
            judged = True

        if len(self.top_step_values) > self.top_num:
            delete_steps.add(self.top_step_values.pop(-1)[0])

        # last N
        if len(self.last_steps) < self.last_num:
            self.last_steps.append(step)
            judged = True
        else:
            delete_steps.add(self.last_steps.pop(0))
            self.last_steps.append(step)

        # save and delete
        if judged:
            output_path = self.output_dir / f"{self.prefix}{step}.pth"
            tmp_output_path = self.output_dir / f"{self.prefix}{step}.pth.tmp"
            torch.save(self.predictor.state_dict(), tmp_output_path)
            tmp_output_path.rename(output_path)

        delete_steps = delete_steps - (
            set([x[0] for x in self.top_step_values]) | set(self.last_steps)
        )
        for delete_step in delete_steps:
            delete_path = self.output_dir / f"{self.prefix}{delete_step}.pth"
            if delete_path.exists():
                delete_path.unlink()

    def state_dict(self):
        state_dict = {
            "last_steps": self.last_steps,
            "top_step_values": self.top_step_values,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.last_steps = state_dict["last_steps"]
        self.top_step_values = state_dict["top_step_values"]
