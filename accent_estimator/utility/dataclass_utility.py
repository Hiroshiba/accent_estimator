import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional


def convert_to_dict(data) -> Dict[str, Any]:
    if dataclasses.is_dataclass(data):
        data = dataclasses.asdict(data)
    for key, val in data.items():
        if isinstance(val, Path):
            data[key] = str(val)
        if isinstance(val, dict):
            data[key] = convert_to_dict(val)
    return data


def convert_from_dict(cls, data):
    if data is None:
        data = {}

    for key, val in data.items():
        child_class = cls.__dataclass_fields__[key].type
        if child_class == Path:
            data[key] = Path(val)
        if child_class == Optional[Path]:
            data[key] = Path(val) if val is not None else None
        if dataclasses.is_dataclass(child_class) and val is not None:
            data[key] = convert_from_dict(child_class, val)
    return cls(**data)
