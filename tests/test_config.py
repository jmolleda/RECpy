"""The configuration layer: what it resolves, and what it refuses.

Three settings were silently lost while the scripts were being migrated onto
this layer, and two of them left code that still ran. These tests exist so that
class of loss is caught by a test rather than by a reviewer.
"""
from __future__ import annotations

import pytest

from recpy_config import Config, load


# --- what the shipped configurations resolve to ----------------------------
def test_shipped_configurations_are_consistent(default_cfg, quick_cfg):
    assert default_cfg.validate() == []
    assert quick_cfg.validate() == []


def test_default_runs_the_whole_study(default_cfg):
    assert default_cfg.model_families == ["ml", "arima", "dl"]
    assert default_cfg.ml_subset == [], "empty means all fifteen regressors"
    assert default_cfg.max_test_days == 0, "zero means the whole test month"
    assert len(default_cfg.supply_points) == 6
    assert len(default_cfg.folds) == 4


def test_fold_labels_keep_their_en_dash(default_cfg):
    """The labels are keys in the optimization result files. A hyphen here
    would silently change those keys and break comparisons with earlier runs."""
    for fold in default_cfg.folds:
        assert "–" in fold["label"], fold["label"]


def test_folds_combine_month_day_with_the_year(cfg_with):
    cfg = cfg_with(year=1999)
    assert cfg.folds[0]["test_start"].startswith("1999-")
    assert cfg.train_start.startswith("1999-")


def test_paths_resolve_under_the_repository(default_cfg, root):
    assert default_cfg.data_dir == root / "data"
    assert default_cfg.data_file("prices").name.endswith(".csv")


def test_output_root_follows_the_environment(default_cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("RECPY_OUTPUT_DIR", str(tmp_path))
    assert default_cfg.output_root == tmp_path


# --- per-entity settings that a blanket edit could drop ---------------------
def test_every_supply_point_has_deep_learning_hyperparameters(default_cfg):
    for sp in default_cfg.supply_points_dl():
        for arch in ("lstm", "gru", "tcn"):
            assert sp[arch], f"{sp['id']} has no {arch} parameters"
        assert len(sp[arch := "tcn"]) == 6, "tcn carries its dilations"


def test_drop_cols_names_every_other_consumer(default_cfg):
    """Built from the dataset's consumption columns, not from the configured
    supply points: a configuration analysing a subset would otherwise leave the
    others in the frame and train on the wrong meter."""
    for sp in default_cfg.supply_points_dl():
        assert sp["column"] not in sp["drop_cols"]
        assert set(sp["drop_cols"]) == set(default_cfg.consumption_columns) - {sp["column"]}


def test_drop_cols_is_right_for_a_subset(cfg_with, default_cfg):
    cfg = cfg_with(supply_points=[sp for sp in default_cfg.raw["supply_points"]
                                  if sp["id"] in ("SP2", "SP3")])
    for sp in cfg.supply_points_dl():
        assert len(sp["drop_cols"]) == 5, "all five other meters are dropped"


def test_every_supply_point_has_arima_orders(default_cfg):
    for sp in default_cfg.supply_points_arima():
        assert len(sp["arima_order"]) == 3
        assert len(sp["sarimax_order"]) == 3
        assert len(sp["sarimax_seasonal"]) == 4
        assert sp["sarimax_exog"], f"{sp['id']} has no exogenous variables"
        assert sp["col"] == sp["column"]


def test_column_names_are_configurable(cfg_with):
    cfg = cfg_with(columns={"timestamp_consumption": "Timestamp",
                            "timestamp_market": "Timestamp",
                            "import_price": "buy", "export_price": "sell",
                            "generation": ["roof", "barn"],
                            "battery_power": "flow"})
    assert cfg.time_col == "Timestamp"
    assert cfg.import_price_col == "buy"
    assert cfg.generation_cols == ["roof", "barn"]
    assert cfg.battery_power_col == "flow"


def test_column_names_fall_back_to_this_study(cfg_with):
    cfg = cfg_with(columns={})
    assert cfg.time_col == "date_time"
    assert cfg.import_price_col == "PVPC"
    assert len(cfg.generation_cols) == 2


def test_a_single_generation_column_is_accepted(cfg_with):
    assert cfg_with(columns={"generation": "one_array"}).generation_cols == ["one_array"]


# --- the run levers ---------------------------------------------------------
def test_model_subset_filters_and_keeps_registry_order(default_cfg, cfg_with):
    models = {"a": 1, "b": 2, "c": 3}
    assert default_cfg.select_models(models) == models, "empty subset means all"
    cfg = cfg_with(models={**default_cfg.raw["models"], "ml_subset": ["c", "a"]})
    assert list(cfg.select_models(models)) == ["c", "a"]


def test_unknown_model_name_is_refused(cfg_with, default_cfg):
    cfg = cfg_with(models={**default_cfg.raw["models"], "ml_subset": ["nope"]})
    with pytest.raises(KeyError, match="nope"):
        cfg.select_models({"a": 1})


def test_test_window_cap(cfg_with, default_cfg):
    import pandas as pd
    hours = pd.Series(range(240))
    assert len(default_cfg.limit_test(hours)) == 240, "zero means no cap"
    cfg = cfg_with(models={**default_cfg.raw["models"], "max_test_days": 3})
    assert len(cfg.limit_test(hours)) == 72


# --- what validate() refuses ------------------------------------------------
@pytest.mark.parametrize("changes, expected", [
    ({"models__families": ["ml", "transformer"]}, "unknown"),
    ({"models__ml_subset": ["Huber"]}, "not in models.ml_subset"),
    ({"folds": []}, "no folds defined"),
    ({"supply_points": []}, "no supply points defined"),
])
def test_validate_reports(cfg_with, changes, expected):
    problems = cfg_with(**changes).validate()
    assert problems, f"expected a problem for {changes}"
    assert any(expected in p for p in problems), problems


def test_validate_reports_a_missing_fold_field(cfg_with, default_cfg):
    folds = [dict(f) for f in default_cfg.raw["folds"]]
    folds[0].pop("label")
    assert any("missing label" in p for p in cfg_with(folds=folds).validate())


def test_validate_reports_a_missing_history_file(cfg_with, default_cfg):
    sps = [dict(sp) for sp in default_cfg.raw["supply_points"]]
    sps[0]["history_csv"] = "nowhere.csv"
    assert any("history file not found" in p
               for p in cfg_with(supply_points=sps).validate())


def test_validate_reports_a_consumer_outside_the_dataset_columns(cfg_with,
                                                                 default_cfg):
    sps = [dict(sp) for sp in default_cfg.raw["supply_points"]]
    sps[0]["column"] = "C_NOT_THERE"
    problems = cfg_with(supply_points=sps).validate()
    assert any("consumption_columns" in p for p in problems), problems


def test_a_missing_configuration_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="RECPY_CONFIG"):
        load(tmp_path / "absent.toml")


# --- where results live -----------------------------------------------------
def test_prediction_path_is_the_published_one(default_cfg, monkeypatch, tmp_path):
    """The archived release named these files after "C_" plus the consumer id,
    which for this community is also its column. The layout must not change."""
    monkeypatch.setenv("RECPY_OUTPUT_DIR", str(tmp_path))
    for sp in default_cfg.supply_points:
        for fold in default_cfg.folds:
            expected = (tmp_path / "cross_validation" / fold["name"]
                        / f"Predictions_C_{sp['id']}_{fold['name']}.csv")
            assert default_cfg.prediction_path(sp, fold) == expected


def test_prediction_path_takes_a_consumer_an_id_or_a_column(default_cfg):
    """Stages hold different things: the writers have the consumer, the
    scheduler has the column. All three must reach the same file."""
    sp, fold = default_cfg.supply_points[0], default_cfg.folds[0]
    paths = {default_cfg.prediction_path(k, fold)
             for k in (sp, sp["id"], sp["column"])}
    assert len(paths) == 1, paths


def test_prediction_path_follows_the_column_not_the_id(cfg_with, default_cfg):
    """A community whose columns are not "C_" plus the id used to have the
    forecasting stages write one filename and the scheduler look for another,
    which failed at the scheduling stage after all the expensive work."""
    sps = [dict(sp) for sp in default_cfg.raw["supply_points"]]
    sps[0] = {**sps[0], "id": "meter_01", "column": "meter_01"}
    cfg = cfg_with(supply_points=sps,
                   data={**default_cfg.raw["data"],
                         "consumption_columns":
                             ["meter_01"] + default_cfg.consumption_columns[1:]})
    fold = cfg.folds[0]
    by_id = cfg.prediction_path("meter_01", fold)
    by_column = cfg.prediction_path(cfg.supply_points[0], fold)
    assert by_id == by_column
    assert by_id.name == f"Predictions_meter_01_{fold['name']}.csv"
