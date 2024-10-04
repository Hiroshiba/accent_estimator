import filecmp
from glob import glob
from os import PathLike, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy


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
        cache_dir: Path = Path(
            "/mnt/disks/local-ssd"
        ),  # FIXME: 環境変数で指定可能にする
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


class Hdf5Path:
    """HDF5ファイル用のパスっぽいもの。HDF5ファイルへのパスと、ファイル内のパスを持つ。"""

    def __init__(self, hdf5_path: Path, internal_path: str):
        self.hdf5_path = hdf5_path
        self.internal_path = internal_path

    def __str__(self):
        return f"{self.hdf5_path}/{self.internal_path}"

    def __repr__(self):
        return f"<Hdf5Path '{self.hdf5_path}/{self.internal_path}'>"


HPath = Path | CachePath | Hdf5Path  # HDF5用パスか通常のパス


def _hdf5_to_path(p: Path):
    with h5py.File(p, "r") as f:
        return {Path(k).stem: Hdf5Path(p, k) for k in f.keys()}


def _stem_to_path(g: str) -> dict[str, HPath]:
    d = {}
    for p in map(Path, glob(g)):
        if p.suffix == ".hdf5":
            d.update(_hdf5_to_path(p))
        else:
            d[Path(p).stem] = p
    return d


def get_stem_to_paths(*globs: str):
    """
    globからstemをキーとしたPathの辞書を取得する。
    HDF5ファイルの場合はroot直下の名前をstemとして扱い、HDF5ファイルのパスをPathとして扱う。
    stemが一致しない場合はエラー。
    """
    stem_to_paths: list[dict[str, HPath]] = []

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


def read_text(path: HPath) -> str:
    if isinstance(path, (Path, CachePath)):
        return path.read_text()
    else:
        with h5py.File(path.hdf5_path, "r") as f:
            return f[path.internal_path][()].decode()


def load_numpy(path: HPath) -> numpy.ndarray:
    if isinstance(path, (Path, CachePath)):
        return numpy.load(path)
    else:
        with h5py.File(path.hdf5_path, "r") as f:
            return numpy.array(f[path.internal_path])  # TODO: 動くか検証する
