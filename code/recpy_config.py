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

    # ---- column names in the data files ----------------------------------
    @property
    def columns(self) -> dict[str, Any]:
        """Column names inside the CSV files, with this study's as defaults."""
        c = dict(self.raw.get("columns", {}))
        c.setdefault("timestamp_consumption", "date_time")
        c.setdefault("timestamp_market", "datetime")
        c.setdefault("import_price", "PVPC")
        c.setdefault("export_price", "Compensation")
        c.setdefault("generation", ["PV to inverter A",
                                    "PV to inverter B"])
        c.setdefault("battery_power", "Battery Power")
        return c

    @property
    def time_col(self) -> str:
        """Timestamp column of the consumption and weather files."""
        return self.columns["timestamp_consumption"]

    @property
    def market_time_col(self) -> str:
        """Timestamp column of the price and generation files."""
        return self.columns["timestamp_market"]

    @property
    def import_price_col(self) -> str:
        return self.columns["import_price"]

    @property
    def export_price_col(self) -> str:
        return self.columns["export_price"]

    @property
    def generation_cols(self) -> list[str]:
        """Generation columns, summed to give total production."""
        g = self.columns["generation"]
        return [g] if isinstance(g, str) else list(g)

    @property
    def battery_power_col(self) -> str:
        return self.columns["battery_power"]

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

    def dl_params(self, sp_id: str, arch: str) -> tuple:
        """DL hyperparameters for one supply point, in the tuple order the model
        builders expect: lstm/gru -> (n_input, nodes, epochs, batch, dropout);
        tcn -> (n_input, filters, dilations, epochs, batch, dropout)."""
        d = self.raw["dl"][sp_id][arch]
        if arch == "tcn":
            return (d["n_input"], d["filters"], list(d["dilations"]),
                    d["epochs"], d["batch"], d["dropout"])
        return (d["n_input"], d["nodes"], d["epochs"], d["batch"], d["dropout"])

    @property
    def consumption_columns(self) -> list[str]:
        """Every consumption column in the multivariate dataset, including supply
        points this configuration does not analyse.

        The deep-learning models take one consumption series as their target and
        the calendar and weather columns as features, so every *other* household's
        consumption must be removed first. Deriving that list from the configured
        supply points alone is wrong whenever a configuration analyses a subset:
        the unconfigured households would stay in the frame and the first column
        -- the training target -- would be the wrong household's.
        """
        cols = self.raw.get("data", {}).get("consumption_columns")
        if cols:
            return list(cols)
        return [sp["column"] for sp in self.raw["supply_points"]]

    def supply_points_dl(self) -> list[dict[str, Any]]:
        """Supply points enriched with DL hyperparameters and drop_cols, as the
        deep-learning scripts consume them."""
        cols = self.consumption_columns
        out = []
        for sp in self.supply_points:
            entry = dict(sp)
            entry["drop_cols"] = [c for c in cols if c != sp["column"]]
            for arch in ("lstm", "gru", "tcn"):
                entry[arch] = self.dl_params(sp["id"], arch)
            out.append(entry)
        return out

    def supply_points_arima(self) -> list[dict[str, Any]]:
        """Supply points enriched with the per-supply-point ARIMA and SARIMAX
        orders, as the statistical scripts consume them. Orders are tuples
        because statsmodels requires them in that form."""
        out = []
        for sp in self.supply_points:
            a = self.raw["arima"][sp["id"]]
            entry = dict(sp)
            entry["col"] = sp["column"]
            entry["arima_order"] = tuple(a["order"])
            entry["sarimax_order"] = tuple(a["sarimax_order"])
            entry["sarimax_seasonal"] = tuple(a["sarimax_seasonal"])
            entry["sarimax_exog"] = list(a["sarimax_exog"])
            out.append(entry)
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

    @property
    def ml_subset(self) -> list[str]:
        """Names of the ML regressors to run; empty means all of them."""
        return list(self.raw["models"].get("ml_subset", []))

    @property
    def max_test_days(self) -> int:
        """Cap on test days per fold; 0 means the whole configured window."""
        return int(self.raw["models"].get("max_test_days", 0))

    def select_models(self, models: dict) -> dict:
        """Filter a name -> estimator mapping through ml_subset."""
        keep = self.ml_subset
        if not keep:
            return models
        missing = [k for k in keep if k not in models]
        if missing:
            raise KeyError(f"ml_subset names unknown models: {missing}")
        return {k: models[k] for k in keep}

    def limit_test(self, hourly):
        """Truncate an hourly test slice (Series, DataFrame or array) to the
        first max_test_days days; 0 leaves it untouched."""
        n = self.max_test_days
        if not n:
            return hourly
        rows = n * 24
        return hourly.iloc[:rows] if hasattr(hourly, "iloc") else hourly[:rows]

    def runs(self, family: str) -> bool:
        """True if this configuration includes the given model family."""
        return family in self.model_families

    # ---- where results live ----------------------------------------------
    def column_of(self, consumer) -> str:
        """The consumption column of a consumer, given the consumer, its id or
        the column itself."""
        if isinstance(consumer, dict):
            return consumer["column"]
        by_id = {sp["id"]: sp["column"] for sp in self.raw["supply_points"]}
        return by_id.get(consumer, consumer)

    def prediction_path(self, consumer, period) -> Path:
        """Where one consumer's forecasts for one period live.

        Every stage has to agree on this: the machine-learning stage creates the
        file, the statistical and deep-learning stages append their columns to
        it, and model selection and scheduling read it. It was previously built
        in ten places from two different parts of the configuration -- the
        consumer's id in some, its column in others -- which agree only while
        every column happens to be "C_" plus the id. A community naming its
        meters anything else had the forecasting stages write one filename and
        the scheduler look for another.

        The column names the file, because the column identifies the series.
        """
        name = period["name"] if isinstance(period, dict) else str(period)
        return (self.output_root / "cross_validation" / name
                / f"Predictions_{self.column_of(consumer)}_{name}.csv")

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
    def train_start(self) -> str:
        """First day of the training history, as YYYY-MM-DD."""
        return f"{self.year}-{self.raw['forecasting']['train_start']}"

    @property
    def lookback_hours(self) -> int:
        return int(self.raw["forecasting"]["lookback_hours"])

    @property
    def horizon_hours(self) -> int:
        return int(self.raw["forecasting"]["horizon_hours"])

    @property
    def random_seed(self) -> int:
        return int(self.raw["forecasting"]["random_seed"])

    # ---- consistency -----------------------------------------------------
    #: Families code/run_pipeline.py knows how to run.
    KNOWN_FAMILIES = ("ml", "arima", "dl")

    def validate(self) -> list[str]:
        """Problems that would make a run fail late or silently; empty if sound.

        Checked before anything expensive starts, because the failures this
        catches surface hours in: a model family nobody runs, or a supply point
        whose deployed model was left out of the regressor subset, which only
        breaks once the scheduling stage looks for its forecasts.

        The structural checks come first and return on their own. Everything
        after them dereferences the keys they look for, so reporting a missing
        key is only possible before that point -- which matters because the
        front end lets a user add a fold or a supply point a field at a time.
        """
        problems = []

        if not self.raw.get("folds"):
            problems.append("no folds defined")
        for i, f in enumerate(self.raw.get("folds", [])):
            missing = [k for k in ("name", "label", "train_end",
                                   "test_start", "test_end") if not f.get(k)]
            if missing:
                problems.append(
                    f"fold {f.get('name') or i + 1}: missing {', '.join(missing)}")

        if not self.raw.get("supply_points"):
            problems.append("no supply points defined")
        for i, sp in enumerate(self.raw.get("supply_points", [])):
            missing = [k for k in ("id", "history_csv", "column")
                       if not sp.get(k)]
            if missing:
                problems.append(
                    f"supply point {sp.get('id') or i + 1}: "
                    f"missing {', '.join(missing)}")

        if problems:
            return problems

        unknown = [f for f in self.model_families if f not in self.KNOWN_FAMILIES]
        if unknown:
            problems.append(
                f"models.families contains unknown {unknown}; "
                f"known families are {list(self.KNOWN_FAMILIES)}")

        keep = self.ml_subset
        if keep:
            for sp in self.raw["supply_points"]:
                dep = sp.get("deployed_model")
                if dep and dep not in keep:
                    problems.append(
                        f"supply point {sp['id']}: deployed_model {dep!r} is not in "
                        f"models.ml_subset, so the scheduling stage would find no "
                        f"forecasts for it")

        known = self.consumption_columns
        for sp in self.raw["supply_points"]:
            if sp["column"] not in known:
                problems.append(
                    f"supply point {sp['id']}: column {sp['column']!r} is not in "
                    f"data.consumption_columns, so the deep-learning stage would "
                    f"train on the wrong column")

        for sp in self.raw["supply_points"]:
            if not (self.data_dir / sp["history_csv"]).is_file():
                problems.append(
                    f"supply point {sp['id']}: history file not found "
                    f"({self.data_dir / sp['history_csv']})")

        if self.runs("dl"):
            for sp_id in self.supply_point_ids:
                if sp_id not in self.raw.get("dl", {}):
                    problems.append(f"no [dl.{sp_id}] section, but 'dl' is enabled")
        if self.runs("arima"):
            for sp_id in self.supply_point_ids:
                if sp_id not in self.raw.get("arima", {}):
                    problems.append(f"no [arima.{sp_id}] section, but 'arima' is enabled")

        return problems

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
    for problem in CONFIG.validate():
        print(f"  PROBLEM: {problem}")
