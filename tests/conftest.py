"""Shared fixtures.

The analysis scripts live in code/ and are run as scripts, not installed as a
package, so the tests put that directory on the path the same way the scripts
find each other.
"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from recpy_config import Config, load  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def default_cfg() -> Config:
    return load(ROOT / "config" / "default.toml")


@pytest.fixture(scope="session")
def quick_cfg() -> Config:
    return load(ROOT / "config" / "quick.toml")


@pytest.fixture
def cfg_with(default_cfg):
    """A configuration with one thing changed, without touching the file."""
    def make(**changes) -> Config:
        raw = copy.deepcopy(default_cfg.raw)
        for dotted, value in changes.items():
            node = raw
            *path, leaf = dotted.split("__")
            for key in path:
                node = node[key]
            node[leaf] = value
        return Config(raw=raw, source=default_cfg.source)
    return make


@pytest.fixture
def dataset(tmp_path, default_cfg):
    """A private copy of the example data, optionally damaged, with a
    configuration pointing at it."""
    def make(mutate=None) -> Config:
        folder = tmp_path / "data"
        folder.mkdir(exist_ok=True)
        for f in default_cfg.data_dir.glob("*.csv"):
            shutil.copy(f, folder / f.name)
        if mutate is not None:
            mutate(folder)
        raw = copy.deepcopy(default_cfg.raw)
        raw["paths"]["data"] = str(folder)
        return Config(raw=raw, source=default_cfg.source)
    return make


@pytest.fixture
def read_csv():
    """Read and rewrite one file of a dataset copy, for damaging it."""
    def edit(folder: Path, name: str, change):
        path = folder / name
        df = pd.read_csv(path)
        change(df).to_csv(path, index=False)
    return edit
