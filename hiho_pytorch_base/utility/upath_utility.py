"""型ユーティリティ"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fsspec.implementations.local import LocalFileSystem
from pydantic import BeforeValidator, PlainSerializer
from upath import UPath


def _to_upath(v: str):
    return UPath(v)


def _ser_upath(v: UPath | None):
    return None if v is None else str(v)


UPathField = Annotated[
    UPath,
    BeforeValidator(_to_upath),
    PlainSerializer(_ser_upath, return_type=str),
]


def ensure_path(p: UPath) -> Path:
    """ローカルならそのパスを返す。リモートなら例外を投げる"""
    if not isinstance(p.fs, LocalFileSystem):
        raise ValueError(f"ローカルパスである必要があります: {p}")
    return Path(str(p))


def to_local_path(p: UPath) -> Path:
    """リモートならキャッシュを作ってそのパスを、ローカルならそのままそのパスを返す"""
    if isinstance(p.fs, LocalFileSystem):
        return Path(str(p))

    cache_dir = Path("./hiho_cache/")
    local_path = cache_dir / hashlib.sha256(p.path.encode()).hexdigest()
    if local_path.exists():
        return local_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False, suffix=".tmp") as f:
        tmp_path = Path(f.name)
    try:
        p.copy(tmp_path)
        os.replace(tmp_path, local_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return local_path
