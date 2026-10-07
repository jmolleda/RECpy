"""The data checker: it must report what would otherwise fail mid-run.

Each test damages a private copy of the example dataset in one specific way and
asserts the checker names it. The point is not that an error is raised but that
the message identifies the file and the remedy, because the person reading it is
adapting the framework to their own data.
"""
from __future__ import annotations

import pandas as pd
import pytest

from check_data import check


def problems(cfg, level="error"):
    return [f"{f.where}: {f.message}" for f in check(cfg) if f.level == level]


def test_the_example_dataset_passes(default_cfg):
    assert problems(default_cfg) == []


def test_a_clean_run_still_reports_what_it_looked_at(default_cfg):
    oks = [f for f in check(default_cfg) if f.level == "ok"]
    # the folder, three shared files and one per consumer
    assert len(oks) == 1 + 3 + len(default_cfg.supply_points)


def test_missing_file(dataset):
    cfg = dataset(lambda d: (d / "solar_generation.csv").unlink())
    assert any("file not found" in p for p in problems(cfg))


def test_missing_price_column(dataset, read_csv):
    cfg = dataset(lambda d: read_csv(d, "prices_cv_folds.csv",
                                     lambda df: df.drop(columns=["PVPC"])))
    found = problems(cfg)
    assert any("PVPC" in p for p in found), found


def test_renamed_timestamp_column_suggests_the_setting(dataset, read_csv):
    cfg = dataset(lambda d: read_csv(d, "consumption_SP3.csv",
                                     lambda df: df.rename(
                                         columns={"date_time": "fecha"})))
    found = problems(cfg)
    assert any("no column named 'date_time'" in p for p in found), found
    assert any("[columns]" in p for p in found), "it should say how to fix it"


def test_wrong_sampling_rate(dataset, read_csv):
    def half_hourly(df):
        df["datetime"] = pd.date_range(df["datetime"].iloc[0], periods=len(df),
                                       freq="30min")
        return df
    cfg = dataset(lambda d: read_csv(d, "solar_generation.csv", half_hourly))
    found = problems(cfg)
    assert any("not one hour" in p for p in found), found


def test_a_gap_in_the_hours(dataset, read_csv):
    cfg = dataset(lambda d: read_csv(d, "consumption_SP2.csv",
                                     lambda df: df.drop(index=range(100, 124))))
    found = problems(cfg)
    assert any("gaps differ" in p for p in found), found


def test_a_partial_day(dataset, read_csv):
    cfg = dataset(lambda d: read_csv(d, "consumption_SP4.csv",
                                     lambda df: df.iloc[:-5]))
    found = problems(cfg)
    assert any("whole number of days" in p for p in found), found


def test_missing_values(dataset, read_csv):
    def blank(df):
        df.loc[10:20, "C_SP5"] = None
        return df
    cfg = dataset(lambda d: read_csv(d, "consumption_SP5.csv", blank))
    found = problems(cfg)
    assert any("missing or non-numeric" in p for p in found), found


def test_data_ending_before_a_configured_period(dataset, read_csv):
    def truncate(df):
        return df[pd.to_datetime(df["date_time"]) < "2025-06-01"]
    cfg = dataset(lambda d: read_csv(d, "consumption_SP6.csv", truncate))
    found = problems(cfg)
    assert any("before" in p and "Fold" in p for p in found), found


def test_negative_consumption_is_a_warning_not_an_error(dataset, read_csv):
    def negate(df):
        df.loc[5:9, "C_SP1"] = -1.5
        return df
    cfg = dataset(lambda d: read_csv(d, "consumption_SP1.csv", negate))
    assert problems(cfg) == [], "a sign oddity should not block a run"
    assert any("negative" in p for p in problems(cfg, "warning"))


def test_a_missing_data_folder_stops_at_the_first_finding(cfg_with):
    cfg = cfg_with(paths={"data": "no/such/folder", "output": "code"})
    found = check(cfg)
    assert len(found) == 1 and found[0].level == "error"


@pytest.mark.parametrize("name", ["prices", "solar", "meteo_calendar"])
def test_each_shared_file_is_actually_examined(default_cfg, name):
    where = default_cfg.data_file(name).name
    assert any(f.where == where for f in check(default_cfg))
