"""
Check that a dataset is the shape the pipeline expects, before running anything.

Pointing the framework at another community's data is where reuse actually
fails, and it used to fail badly: a missing column or a half-hourly series
surfaced as a pandas error thrown from somewhere deep in a forecasting script,
hours into a run. This reports the same problems in advance, in terms of the
files and columns you supplied.

    python code/check_data.py                       # the active configuration
    RECPY_CONFIG=config/mine.toml python code/check_data.py

Exits 0 when the data is usable, 1 when something would stop a run. Warnings do
not fail the check: they are things that are survivable but worth knowing, such
as a period your files only partly cover.

The checks are also importable, so the front end can run them on demand:

    from check_data import check
    for finding in check(CONFIG):
        ...
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import argparse
import os


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a dataset is the shape the pipeline expects.",
        epilog="The same settings can be given as the environment variables "
               "RECPY_CONFIG and RECPY_OUTPUT_DIR.")
    parser.add_argument("-c", "--config", metavar="FILE",
                        help="configuration to use (default: config/default.toml)")
    parser.add_argument("-o", "--output", metavar="DIR",
                        help="where to write results (default: from the configuration)")
    return parser.parse_args()


# Applied before the configuration is imported below, since it resolves itself
# from the environment as soon as it is loaded.
if __name__ == "__main__":
    _args = _arguments()
    if _args.config:
        os.environ["RECPY_CONFIG"] = _args.config
    if _args.output:
        os.environ["RECPY_OUTPUT_DIR"] = _args.output

from recpy_config import CONFIG, Config

#: A day is 24 rows everywhere in the pipeline: series are reshaped into whole
#: days before any model sees them.
HOURS_PER_DAY = 24


@dataclass(frozen=True)
class Finding:
    level: str          # "error", "warning" or "ok"
    where: str          # the file or setting the finding is about
    message: str

    def __str__(self) -> str:
        mark = {"error": "ERROR  ", "warning": "warning", "ok": "ok     "}
        return f"  {mark[self.level]}  {self.where:<34} {self.message}"


def _read(path: Path, time_col: str) -> tuple[pd.DataFrame | None, list[Finding]]:
    """Read one CSV indexed by its timestamp column, reporting why if it fails."""
    where = path.name
    if not path.is_file():
        return None, [Finding("error", where, f"file not found: {path}")]
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception as exc:
        return None, [Finding("error", where, f"cannot be read: {exc}")]
    if time_col not in head.columns:
        return None, [Finding(
            "error", where,
            f"no column named {time_col!r}. It has: "
            f"{', '.join(map(str, head.columns[:6]))}"
            f"{' ...' if len(head.columns) > 6 else ''}. "
            f"Set the timestamp name under [columns] to match.")]
    try:
        df = pd.read_csv(path, parse_dates=[time_col], index_col=time_col)
    except Exception as exc:
        return None, [Finding("error", where,
                              f"{time_col!r} does not parse as dates: {exc}")]
    return df, []


def _check_index(df: pd.DataFrame, where: str) -> list[Finding]:
    """The timeline itself: ordered, unique, hourly, whole days."""
    out = []
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return [Finding("error", where, "the timestamp column is not a date")]
    if idx.has_duplicates:
        n = int(idx.duplicated().sum())
        out.append(Finding("error", where, f"{n} duplicated timestamps"))
    if not idx.is_monotonic_increasing:
        out.append(Finding("error", where, "timestamps are not in order"))

    gaps = idx.to_series().diff().dropna()
    if not gaps.empty:
        common = gaps.mode().iloc[0]
        if common != pd.Timedelta(hours=1):
            out.append(Finding(
                "error", where,
                f"the usual step between rows is {common}, not one hour. The "
                f"pipeline models hourly data; resample before using it."))
        odd = int((gaps != common).sum())
        if odd:
            first = gaps[gaps != common].index[0]
            out.append(Finding(
                "error", where,
                f"{odd} gaps differ from {common}, the first at {first}. "
                f"Fill or interpolate the missing hours."))

    if len(df) % HOURS_PER_DAY:
        out.append(Finding(
            "error", where,
            f"{len(df):,} rows is not a whole number of days "
            f"({len(df) % HOURS_PER_DAY} left over). The pipeline forecasts "
            f"whole days at a time."))
    return out


def _check_columns(df: pd.DataFrame, where: str, needed: dict) -> list[Finding]:
    out = []
    for label, col in needed.items():
        for name in ([col] if isinstance(col, str) else col):
            if name not in df.columns:
                out.append(Finding(
                    "error", where,
                    f"no column named {name!r} for {label}. It has: "
                    f"{', '.join(map(str, df.columns[:6]))}"
                    f"{' ...' if len(df.columns) > 6 else ''}"))
    return out


def _check_values(df: pd.DataFrame, where: str, cols: list[str],
                  non_negative: bool) -> list[Finding]:
    out = []
    for name in cols:
        if name not in df.columns:
            continue
        s = pd.to_numeric(df[name], errors="coerce")
        if s.isna().any():
            out.append(Finding("error", where,
                               f"{name!r} has {int(s.isna().sum()):,} missing "
                               f"or non-numeric values"))
        if non_negative and (s.dropna() < 0).any():
            out.append(Finding("warning", where,
                               f"{name!r} goes negative "
                               f"({s.min():.3f} at the lowest)"))
        if s.dropna().empty:
            continue
        if s.dropna().std() == 0:
            out.append(Finding("warning", where,
                               f"{name!r} never changes (constant {s.iloc[0]})"))
    return out


def _summary(df: pd.DataFrame, where: str) -> Finding:
    """One line per file, so a clean run still shows what was looked at."""
    if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
        return Finding("ok", where,
                       f"{len(df):,} rows, {df.index.min().date()} to "
                       f"{df.index.max().date()}, {len(df.columns)} columns")
    return Finding("ok", where, f"{len(df):,} rows, {len(df.columns)} columns")


def _check_coverage(df: pd.DataFrame, where: str, cfg: Config) -> list[Finding]:
    """Does the file span the periods the configuration asks to run?"""
    out = []
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return out
    first, last = df.index.min(), df.index.max()
    wanted_from = pd.Timestamp(cfg.train_start)
    if first > wanted_from:
        out.append(Finding("warning", where,
                           f"starts {first.date()}, after the configured "
                           f"history start {wanted_from.date()}"))
    for fold in cfg.folds:
        end = pd.Timestamp(fold["test_end"]) + pd.Timedelta(hours=23)
        if last < end:
            out.append(Finding(
                "error", where,
                f"ends {last.date()}, before \"{fold['label']}\" finishes "
                f"{pd.Timestamp(fold['test_end']).date()}"))
    return out


def check(cfg: Config = CONFIG) -> list[Finding]:
    """Every check, over every file the configuration names."""
    findings: list[Finding] = []

    if not cfg.data_dir.is_dir():
        return [Finding("error", "data folder",
                        f"not a folder: {cfg.data_dir}")]
    findings.append(Finding("ok", "data folder", str(cfg.data_dir)))

    # --- prices ----------------------------------------------------------
    df, errs = _read(cfg.data_file("prices"), cfg.market_time_col)
    findings += errs
    if df is not None:
        where = cfg.data_file("prices").name
        findings += _check_index(df, where)
        findings += _check_columns(df, where, {
            "the price paid for imports": cfg.import_price_col,
            "the price received for exports": cfg.export_price_col})
        findings += _check_values(df, where,
                                  [cfg.import_price_col, cfg.export_price_col],
                                  non_negative=False)
        findings += _check_coverage(df, where, cfg)
        findings.append(_summary(df, where))

    # --- generation ------------------------------------------------------
    df, errs = _read(cfg.data_file("solar"), cfg.market_time_col)
    findings += errs
    if df is not None:
        where = cfg.data_file("solar").name
        findings += _check_index(df, where)
        findings += _check_columns(df, where, {
            "generation": cfg.generation_cols,
            "the metered battery flow": cfg.battery_power_col})
        findings += _check_values(df, where, cfg.generation_cols,
                                  non_negative=True)
        findings += _check_coverage(df, where, cfg)
        findings.append(_summary(df, where))

    # --- weather, calendar and every consumer's column --------------------
    df, errs = _read(cfg.data_file("meteo_calendar"), cfg.time_col)
    findings += errs
    if df is not None:
        where = cfg.data_file("meteo_calendar").name
        findings += _check_index(df, where)
        findings += _check_columns(df, where, {
            "consumption": cfg.consumption_columns})
        findings += _check_values(df, where,
                                  [c for c in cfg.consumption_columns
                                   if c in df.columns], non_negative=True)
        findings += _check_coverage(df, where, cfg)
        findings.append(_summary(df, where))
        for sp in cfg.supply_points:
            if sp["column"] not in df.columns:
                findings.append(Finding(
                    "error", where,
                    f"consumer {sp['id']} reads column {sp['column']!r}, "
                    f"which is not in this file"))

    # --- one consumption file per consumer --------------------------------
    for sp in cfg.supply_points:
        path = Path(sp["history_csv"])
        df, errs = _read(path, cfg.time_col)
        findings += errs
        if df is None:
            continue
        where = f"{sp['id']}: {path.name}"
        findings += _check_index(df, where)
        findings += _check_columns(df, where, {"its consumption": sp["column"]})
        findings += _check_values(df, where, [sp["column"]], non_negative=True)
        findings += _check_coverage(df, where, cfg)
        findings.append(_summary(df, where))

    return findings


def main() -> int:
    print(f"Checking the data named by {CONFIG.source.name}")
    print()
    findings = check(CONFIG)
    for f in findings:
        print(f)

    errors = [f for f in findings if f.level == "error"]
    warnings = [f for f in findings if f.level == "warning"]
    print()
    if errors:
        print(f"{len(errors)} problem(s) would stop a run"
              f"{f', and {len(warnings)} worth knowing about' if warnings else ''}.")
        return 1
    if warnings:
        print(f"Usable, with {len(warnings)} thing(s) worth knowing about.")
        return 0
    print("The data is the shape the pipeline expects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
