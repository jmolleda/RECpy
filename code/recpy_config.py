"""
RECpy configuration.

Every analysis script imports its settings from here instead of hard-coding them,
so the pipeline can be pointed at a different energy community, a different
calendar year, or a reduced run without editing any script.

Selecting a configuration
-------------------------
    python code/generate_ml_cv.py                      # config/default.toml
    RECPY_CONFIG=config/quick.toml python code/...     # a different file
    RECPY_OUTPUT_DIR=/results python code/...          # redirect outputs only

The output root is overridable on its own because hosted reproducibility
platforms (e.g. Code Ocean) discard anything written under the code directory and
capture only a dedicated results directory.

Typical use
-----------
    from recpy_config import CONFIG

    for fold in CONFIG.folds:            # fold["name"], ["train_end"], ...
        for sp in CONFIG.supply_points:  # sp["id"], ["column"], ...
            ...
    out = CONFIG.output_dir("cross_validation", fold["name"])

Defaults reproduce the configuration used for the published results, so an
unmodified checkout behaves exactly as the archived v1.0.0 release did.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Repository root (this file lives in <root>/code/).
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = ROOT / "config" / "default.toml"

ENV_CONFIG = "RECPY_CONFIG"
ENV_OUTPUT = "RECPY_OUTPUT_DIR"


def _resolve(value: str | Path) -> Path:
    """Resolve a path from the config: absolute kept, relative taken from ROOT."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Config:
    """Settings for one run, loaded from a TOML file."""

    raw: dict[str, Any]
    source: Path

    # ---- identity -------------------------------------------------------
    @property
    def name(self) -> str:
        return self.raw.get("name", self.source.stem)

    @property
    def description(self) -> str:
        return self.raw.get("description", "")

    @property
    def year(self) -> int:
        return int(self.raw["year"])

    # ---- directories ----------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return _resolve(self.raw["paths"]["data"])

    @property
    def output_root(self) -> Path:
        """Root for generated output; RECPY_OUTPUT_DIR overrides the file."""
        env = os.environ.get(ENV_OUTPUT)
        return _resolve(env) if env else _resolve(self.raw["paths"]["output"])

    def output_dir(self, *parts: str) -> Path:
        """Return (and create) an output directory under the output root."""
        d = self.output_root.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def data_file(self, key: str) -> Path:
        """Path to a named dataset file, e.g. data_file("prices")."""
        return self.data_dir / self.raw["data_files"][key]

    # ---- experiment design ---------------------------------------------
    @property
    def folds(self) -> list[dict[str, str]]:
        """Folds with month-day boundaries expanded against the configured year."""
        y = self.year
        out = []
        for f in self.raw["folds"]:
            out.append({
                "name": f["name"],
                "label": f["label"],
                "train_end": f"{y}-{f['train_end']}",
                "test_start": f"{y}-{f['test_start']}",
                "test_end": f"{y}-{f['test_end']}",
                # convenience aliases used by the optimization scripts
                "dir": f["name"],
                "suffix": f["name"],
                "month": f"{y}-{f['test_start'][:2]}",
            })
        return out

    @property
    def supply_points(self) -> list[dict[str, str]]:
        """Supply points, with data paths resolved."""
        out = []
        for sp in self.raw["supply_points"]:
            out.append({
                "id": sp["id"],
                "name": sp["id"],
                "folder": sp["id"],
                "column": sp["column"],
                "history_col": sp["column"],
                "history_csv": str(self.data_dir / sp["history_csv"]),
                "deployed_model": sp.get("deployed_model"),
            })
        return out

    @property
    def supply_point_ids(self) -> list[str]:
        return [sp["id"] for sp in self.raw["supply_points"]]

    @property
    def deployed_models(self) -> dict[str, str]:
        """Column name -> statistically selected model (the Wilcoxon winner)."""
        return {sp["column"]: sp["deployed_model"]
                for sp in self.raw["supply_points"] if sp.get("deployed_model")}

    @property
    def model_families(self) -> list[str]:
        return list(self.raw["models"]["families"])

    def runs(self, family: str) -> bool:
        """True if this configuration includes the given model family."""
        return family in self.model_families

    # ---- physical system ------------------------------------------------
    @property
    def battery(self) -> dict[str, float]:
        return dict(self.raw["battery"])

    @property
    def initial_soc(self) -> float:
        b = self.raw["battery"]
        return b["capacity_kwh"] * b["initial_soc_fraction"]

    @property
    def grid_export_limit(self) -> float:
        return float(self.raw["grid"]["export_limit_kw"])

    # ---- forecasting ----------------------------------------------------
    @property
    def lookback_hours(self) -> int:
        return int(self.raw["forecasting"]["lookback_hours"])

    @property
    def horizon_hours(self) -> int:
        return int(self.raw["forecasting"]["horizon_hours"])

    @property
    def random_seed(self) -> int:
        return int(self.raw["forecasting"]["random_seed"])

    # ---- reporting ------------------------------------------------------
    def summary(self) -> str:
        return (
            f"RECpy configuration '{self.name}' ({self.source.name})\n"
            f"  year            : {self.year}\n"
            f"  supply points   : {', '.join(self.supply_point_ids)}\n"
            f"  folds           : {', '.join(f['name'] for f in self.folds)}\n"
            f"  model families  : {', '.join(self.model_families)}\n"
            f"  data            : {self.data_dir}\n"
            f"  output          : {self.output_root}"
        )


def load(path: str | Path | None = None) -> Config:
    """Load a configuration. Order: argument, then $RECPY_CONFIG, then the default."""
    chosen = path or os.environ.get(ENV_CONFIG) or DEFAULT_CONFIG
    p = _resolve(chosen)
    if not p.is_file():
        raise FileNotFoundError(
            f"RECpy configuration not found: {p}\n"
            f"Set {ENV_CONFIG} to a TOML file, or restore {DEFAULT_CONFIG}."
        )
    return Config(raw=tomllib.loads(p.read_text(encoding="utf-8")), source=p)


#: Configuration for the current run, imported by the analysis scripts.
CONFIG = load()


if __name__ == "__main__":
    print(CONFIG.summary())
