from pathlib import Path

from setuptools import find_packages, setup

install_requires = [
    r if not r.startswith("git+") else f"{Path(r).stem.split('@')[0]} @ {r}"
    for r in Path("requirements.txt").read_text().split()
]
setup(
    name="accent_estimator",
    version="0.0.1",
    packages=find_packages(),
    author="Kazuyuki Hiroshiba",
    author_email="hihokaruta@gmail.com",
    install_requires=install_requires,
)
