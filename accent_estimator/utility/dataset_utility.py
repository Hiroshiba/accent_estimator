import filecmp
from glob import glob
from os import PathLike, replace
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import Dict, List


def _stem_to_path(g: str):
    return {Path(p).stem: Path(p) for p in glob(g)}


def get_stem_to_paths(*globs: str):
    """
    globからstemをキーとしたPathの辞書を取得する。
    stemが一致しない場合はエラー。
    """
    stem_to_paths: List[Dict[str, Path]] = []

    first_paths = _stem_to_path(globs[0])
    fn_list = sorted(first_paths.keys())
    assert len(fn_list) > 0, f"ファイルが存在しません: {globs[0]}"

    print(f"glob {globs[0]}: {len(first_paths)}")
    stem_to_paths.append(first_paths)

    for g in globs[1:]:
        paths = _stem_to_path(g)
        assert set(fn_list) == set(paths.keys()), f"ファイルが一致しません: {g}"

        print(f"glob {g}: {len(paths)}")
        stem_to_paths.append(paths)

    return fn_list, *stem_to_paths


class CachePath(PathLike):
    """
    ファイルキャッシュを作る。
    `open()`か`str()`か`create_cache()`を呼び出すとキャッシュが作られる。
    同じファイルが存在する場合はタイムスタンプを更新する。
    """

    def __init__(
        self,
        src_path: PathLike,
        dst_path: Path = None,
        cache_dir: Path = Path(gettempdir()),
    ):
        src_path = Path(src_path)
        if dst_path is None:
            dst_path = cache_dir.joinpath(*src_path.absolute().parts[1:])

        self.src_path = src_path
        self.dst_path = dst_path

    def create_cache(self):
        if not self.src_path.exists() or self.src_path.is_dir():
            raise FileNotFoundError(f"ファイルが存在しません: {self.src_path}")

        if self.dst_path.exists() and filecmp.cmp(self.src_path, self.dst_path):
            self.dst_path.touch()
            return

        self.dst_path.parent.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile(dir=str(self.dst_path.parent), delete=False) as f:
            f.write(self.src_path.read_bytes())

        replace(f.name, str(self.dst_path))

    def __str__(self):
        self.create_cache()
        return str(self.dst_path)

    def __fspath__(self):
        return str(self)
