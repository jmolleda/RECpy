"""The pipeline, end to end and against the archived release.

The equivalence check is the important one: it asserts that the configuration
layer still resolves to the values the published v1.0.0 scripts held as
literals. The end-to-end run is deliberately tiny -- one consumer, one period,
one day, one regressor -- so that something actually executes on every commit
without the suite taking minutes.
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys

import pytest
import tomli_w

from recpy_config import load


def test_configuration_matches_the_archived_release(capsys):
    """All 108 settings the v1.0.0 scripts held as literals."""
    import test_config_equivalence as equivalence
    assert equivalence.main() == 0, capsys.readouterr().out


def test_the_driver_lists_its_stages_in_order():
    from run_pipeline import STAGES
    assert [label for _, label, _ in STAGES] == [
        "Machine-learning forecasts",
        "Statistical forecasts",
        "Deep-learning forecasts",
        "Wilcoxon model selection",
        "Visualization",
        "Day-ahead scheduling",
    ]
    assert [fam for _, _, fam in STAGES][:3] == ["ml", "arima", "dl"]


def test_stages_are_skipped_when_their_family_is_not_configured(quick_cfg,
                                                               cfg_with):
    from run_pipeline import STAGES
    cfg = cfg_with(models={**quick_cfg.raw["models"], "families": ["ml"]})
    running = [label for _, label, fam in STAGES if not fam or cfg.runs(fam)]
    assert "Statistical forecasts" not in running
    assert "Deep-learning forecasts" not in running
    assert len(running) == 4


@pytest.mark.slow
def test_a_whole_run_produces_every_output(root, default_cfg, tmp_path):
    """One consumer, one day, one regressor: every stage runs and writes."""
    raw = copy.deepcopy(default_cfg.raw)
    raw["name"] = "test-smoke"
    raw["models"]["families"] = ["ml"]
    raw["models"]["ml_subset"] = ["Huber"]
    raw["models"]["max_test_days"] = 1
    raw["supply_points"] = [sp for sp in raw["supply_points"]
                            if sp["id"] == "SP3"]
    raw["folds"] = raw["folds"][:1]
    raw["dl"] = {"SP3": raw["dl"]["SP3"]}
    raw["arima"] = {"SP3": raw["arima"]["SP3"]}

    cfg_path = tmp_path / "smoke.toml"
    cfg_path.write_bytes(tomli_w.dumps(raw).encode("utf-8"))
    out = tmp_path / "out"

    env = dict(os.environ)
    env["RECPY_CONFIG"] = str(cfg_path)
    env["RECPY_OUTPUT_DIR"] = str(out)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(root / "code" / "run_pipeline.py")],
        cwd=str(root / "code"), env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:]

    fold = load(cfg_path).folds[0]["name"]
    predictions = out / "cross_validation" / fold / "Predictions_C_SP3_{}.csv".format(fold)
    assert predictions.is_file(), "no forecasts were written"
    assert (out / "cross_validation" / "wilcoxon_cv_aggregated.csv").is_file()
    assert (out / "optimization_results" / "optimization_cv_summary.csv").is_file()
    assert list((out / "cross_validation" / "figures").glob("*.png")), "no figures"

    import pandas as pd
    d = pd.read_csv(predictions, index_col=0)
    assert len(d) == 24, "one day is 24 rows"
    assert "Observed" in d.columns and "Huber" in d.columns
    assert not d.isna().any().any(), "a forecast column has gaps"


@pytest.mark.slow
def test_a_run_on_foreign_column_names(root, default_cfg, tmp_path):
    """The whole point of the [columns] section: someone else's data, not
    renamed to match this study."""
    import pandas as pd

    folder = tmp_path / "theirs"
    folder.mkdir()
    rename = {default_cfg.time_col: "Timestamp",
              default_cfg.market_time_col: "Timestamp",
              default_cfg.import_price_col: "buy",
              default_cfg.export_price_col: "sell",
              default_cfg.battery_power_col: "flow",
              default_cfg.generation_cols[0]: "roof",
              default_cfg.generation_cols[1]: "barn"}
    for f in default_cfg.data_dir.glob("*.csv"):
        df = pd.read_csv(f)
        df.rename(columns={k: v for k, v in rename.items()
                           if k in df.columns}).to_csv(folder / f.name,
                                                       index=False)

    raw = copy.deepcopy(default_cfg.raw)
    raw["paths"]["data"] = str(folder)
    raw["columns"] = {"timestamp_consumption": "Timestamp",
                      "timestamp_market": "Timestamp",
                      "import_price": "buy", "export_price": "sell",
                      "generation": ["roof", "barn"], "battery_power": "flow"}
    raw["models"]["families"] = ["ml"]
    raw["models"]["ml_subset"] = ["Huber"]
    raw["models"]["max_test_days"] = 1
    raw["supply_points"] = [sp for sp in raw["supply_points"]
                            if sp["id"] == "SP3"]
    raw["folds"] = raw["folds"][:1]
    raw["dl"] = {"SP3": raw["dl"]["SP3"]}
    raw["arima"] = {"SP3": raw["arima"]["SP3"]}

    cfg_path = tmp_path / "theirs.toml"
    cfg_path.write_bytes(tomli_w.dumps(raw).encode("utf-8"))

    from check_data import check
    assert [f for f in check(load(cfg_path)) if f.level == "error"] == []

    env = dict(os.environ)
    env["RECPY_CONFIG"] = str(cfg_path)
    env["RECPY_OUTPUT_DIR"] = str(tmp_path / "out")
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(root / "code" / "run_pipeline.py")],
        cwd=str(root / "code"), env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert (tmp_path / "out" / "optimization_results"
            / "optimization_cv_summary.csv").is_file()
