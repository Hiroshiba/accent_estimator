from glob import glob
from pathlib import Path
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
