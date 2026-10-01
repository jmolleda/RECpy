"""
Regression gate, part 1: the default configuration resolves to the values that
v1.0.0 held as literals.

Every setting the configuration layer took out of the scripts is listed here as
it appeared in the archived v1.0.0 release -- the version that produced the
published results -- and checked against what config/default.toml now resolves
to. The values below are transcribed from v1.0.0 and must not be edited to make
a test pass: if a check fails, the configuration is wrong.

This exists because three separate settings were silently lost while the scripts
were being migrated (the deep-learning hyperparameters, the ARIMA and SARIMAX
orders, and each supply point's list of columns to drop). Two of them produced
code that still ran. A static check over every supply point and fold is a better
guard against that than any single pipeline run, which exercises only the
entities it happens to touch.

    python code/test_config_equivalence.py

Exits non-zero on the first difference, and prints every check otherwise.
"""

from __future__ import annotations

import sys

from recpy_config import load

YEAR = 2025

# ---------------------------------------------------------------------------
# v1.0.0 literals, transcribed from the archived release
# ---------------------------------------------------------------------------
V1_FOLDS = [
    {"name": "fold1_apr", "train_end": f"{YEAR}-03-31",
     "test_start": f"{YEAR}-04-01", "test_end": f"{YEAR}-04-30"},
    {"name": "fold2_jul", "train_end": f"{YEAR}-06-30",
     "test_start": f"{YEAR}-07-01", "test_end": f"{YEAR}-07-31"},
    {"name": "fold3_oct", "train_end": f"{YEAR}-09-30",
     "test_start": f"{YEAR}-10-01", "test_end": f"{YEAR}-10-31"},
    {"name": "fold4_dec", "train_end": f"{YEAR}-11-30",
     "test_start": f"{YEAR}-12-01", "test_end": f"{YEAR}-12-31"},
]

# Fold labels as the optimization scripts wrote them. These are keys in the
# result CSVs, and the dash is an en dash (U+2013), not a hyphen.
V1_FOLD_LABELS = ["Fold 1 – Apr", "Fold 2 – Jul",
                  "Fold 3 – Oct", "Fold 4 – Dec"]

V1_SUPPLY_POINTS = [
    {"id": "SP1", "history_csv": "data/consumption_SP1.csv", "history_col": "C_SP1"},
    {"id": "SP2", "history_csv": "data/consumption_SP2.csv", "history_col": "C_SP2"},
    {"id": "SP3", "history_csv": "data/consumption_SP3.csv", "history_col": "C_SP3"},
    {"id": "SP4", "history_csv": "data/consumption_SP4.csv", "history_col": "C_SP4"},
    {"id": "SP5", "history_csv": "data/consumption_SP5.csv", "history_col": "C_SP5"},
    {"id": "SP6", "history_csv": "data/consumption_SP6.csv", "history_col": "C_SP6"},
]

# run_optimization_cv.py: the Wilcoxon tournament winner per supply point
V1_SELECTED_MODEL = {
    "C_SP1": "Support Vector",
    "C_SP2": "k-Nearest Neighbors",
    "C_SP3": "Huber",
    "C_SP4": "Huber",
    "C_SP5": "Huber",
    "C_SP6": "Extra Trees",
}

# generate_dl_cv.py: (n_input, nodes, epochs, batch, dropout) for lstm and gru,
# (n_input, filters, dilations, epochs, batch, dropout) for tcn
V1_DL = {
    "SP1": {"lstm": (72, 30, 50, 32, 0.0), "gru": (192, 25, 20, 32, 0.0),
            "tcn": (144, 18, [1, 2, 4], 20, 32, 0.0)},
    "SP2": {"lstm": (72, 18, 20, 64, 0.0), "gru": (72, 25, 50, 32, 0.2),
            "tcn": (72, 25, [1, 2, 4], 50, 32, 0.0)},
    "SP3": {"lstm": (144, 25, 50, 64, 0.0), "gru": (72, 25, 50, 32, 0.0),
            "tcn": (72, 25, [1, 2, 4], 50, 32, 0.0)},
    "SP4": {"lstm": (72, 30, 50, 32, 0.0), "gru": (144, 25, 50, 64, 0.0),
            "tcn": (192, 30, [1, 2, 4, 8], 50, 64, 0.2)},
    "SP5": {"lstm": (72, 25, 50, 32, 0.2), "gru": (144, 25, 50, 32, 0.0),
            "tcn": (72, 18, [1, 2, 4], 20, 32, 0.0)},
    "SP6": {"lstm": (72, 25, 20, 32, 0.0), "gru": (72, 18, 20, 32, 0.0),
            "tcn": (144, 25, [1, 2, 4, 8], 50, 32, 0.0)},
}

# generate_dl_cv.py: every other household's consumption column
V1_DROP_COLS = {
    "SP1": ["C_SP2", "C_SP3", "C_SP4", "C_SP5", "C_SP6"],
    "SP2": ["C_SP1", "C_SP3", "C_SP4", "C_SP5", "C_SP6"],
    "SP3": ["C_SP1", "C_SP2", "C_SP4", "C_SP5", "C_SP6"],
    "SP4": ["C_SP1", "C_SP2", "C_SP3", "C_SP5", "C_SP6"],
    "SP5": ["C_SP1", "C_SP2", "C_SP3", "C_SP4", "C_SP6"],
    "SP6": ["C_SP1", "C_SP2", "C_SP3", "C_SP4", "C_SP5"],
}

DEG = "°"   # degree sign
SUP2 = "²"  # superscript two

# generate_arima_cv.py
V1_ARIMA = {
    "SP1": {"order": (7, 0, 2), "sarimax_order": (1, 0, 1),
            "sarimax_seasonal": (1, 0, 1, 24),
            "sarimax_exog": ["Hour", f"diffuse_radiation_instant (W/m{SUP2})"]},
    "SP2": {"order": (5, 0, 3), "sarimax_order": (1, 1, 1),
            "sarimax_seasonal": (1, 0, 1, 24),
            "sarimax_exog": [f"temperature_2m ({DEG}C)"]},
    "SP3": {"order": (2, 0, 3), "sarimax_order": (0, 0, 1),
            "sarimax_seasonal": (1, 0, 1, 24),
            "sarimax_exog": [f"diffuse_radiation_instant (W/m{SUP2})"]},
    "SP4": {"order": (3, 0, 3), "sarimax_order": (1, 1, 1),
            "sarimax_seasonal": (0, 0, 1, 24),
            "sarimax_exog": ["Hour"]},
    "SP5": {"order": (1, 1, 3), "sarimax_order": (1, 0, 1),
            "sarimax_seasonal": (1, 0, 1, 24),
            "sarimax_exog": ["Hour"]},
    "SP6": {"order": (3, 0, 2), "sarimax_order": (1, 0, 1),
            "sarimax_seasonal": (1, 1, 1, 24),
            "sarimax_exog": ["DoW", "Holidays",
                             f"diffuse_radiation_instant (W/m{SUP2})"]},
}

# opt_core.py / run_optimization_cv.py
V1_BATTERY = {
    "capacity_kwh": 50.16,
    "min_soc_kwh": 5.0,
    "efficiency": 0.96,
    "power_limit_kw": 12.0,
    "degradation_eur_per_kwh": 0.033,
}
V1_INITIAL_SOC = 50.16 * 0.50
V1_GRID_EXPORT_LIMIT = 32.0

V1_SCALARS = {
    "year": YEAR,
    "train_start": f"{YEAR}-01-02",
    "random_seed": 123,
    "lookback_hours": 192,
    "horizon_hours": 24,
}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def eq(self, label: str, got, want) -> None:
        self.checks += 1
        if got != want:
            self.failures.append(f"{label}\n      v1.0.0: {want!r}\n      now:    {got!r}")

    def report(self, title: str) -> None:
        mark = "ok" if not self.failures else "FAILED"
        print(f"  [{mark}] {title} ({self.checks} values)")
        for f in self.failures:
            print(f"    - {f}")


def main() -> int:
    cfg = load("config/default.toml")
    print(f"Comparing {cfg.source.name} against the v1.0.0 literals")
    print()

    groups = []

    # --- scalars ----------------------------------------------------------
    c = Checker()
    for key, want in V1_SCALARS.items():
        c.eq(f"{key}", getattr(cfg, key), want)
    groups.append(("scalars", c))

    # --- folds ------------------------------------------------------------
    c = Checker()
    folds = cfg.folds
    c.eq("number of folds", len(folds), len(V1_FOLDS))
    for got, want in zip(folds, V1_FOLDS):
        for key in ("name", "train_end", "test_start", "test_end"):
            c.eq(f"fold {want['name']}.{key}", got[key], want[key])
    c.eq("fold labels (CSV keys, en dash)",
         [f["label"] for f in folds], V1_FOLD_LABELS)
    groups.append(("folds", c))

    # --- supply points ----------------------------------------------------
    c = Checker()
    sps = cfg.supply_points
    c.eq("number of supply points", len(sps), len(V1_SUPPLY_POINTS))
    for got, want in zip(sps, V1_SUPPLY_POINTS):
        c.eq(f"{want['id']}.id", got["id"], want["id"])
        c.eq(f"{want['id']}.history_col", got["history_col"], want["history_col"])
        # v1.0.0 held a repository-relative path; the config resolves it
        c.eq(f"{want['id']}.history_csv",
             got["history_csv"].replace("\\", "/").split("/")[-1],
             want["history_csv"].split("/")[-1])
    c.eq("deployed models (Wilcoxon winners)", cfg.deployed_models, V1_SELECTED_MODEL)
    groups.append(("supply points", c))

    # --- deep learning ----------------------------------------------------
    c = Checker()
    for sp in cfg.supply_points_dl():
        want = V1_DL[sp["id"]]
        for arch in ("lstm", "gru", "tcn"):
            c.eq(f"{sp['id']}.{arch}", sp[arch], want[arch])
        c.eq(f"{sp['id']}.drop_cols", sp["drop_cols"], V1_DROP_COLS[sp["id"]])
    groups.append(("deep-learning hyperparameters", c))

    # --- statistical models ----------------------------------------------
    c = Checker()
    for sp in cfg.supply_points_arima():
        want = V1_ARIMA[sp["id"]]
        c.eq(f"{sp['id']}.col", sp["col"], f"C_{sp['id']}")
        c.eq(f"{sp['id']}.arima_order", sp["arima_order"], want["order"])
        c.eq(f"{sp['id']}.sarimax_order", sp["sarimax_order"], want["sarimax_order"])
        c.eq(f"{sp['id']}.sarimax_seasonal", sp["sarimax_seasonal"],
             want["sarimax_seasonal"])
        c.eq(f"{sp['id']}.sarimax_exog", sp["sarimax_exog"], want["sarimax_exog"])
    groups.append(("ARIMA / SARIMAX orders", c))

    # --- physical system --------------------------------------------------
    c = Checker()
    for key, want in V1_BATTERY.items():
        c.eq(f"battery.{key}", cfg.battery[key], want)
    c.eq("initial state of charge", cfg.initial_soc, V1_INITIAL_SOC)
    c.eq("grid export limit", cfg.grid_export_limit, V1_GRID_EXPORT_LIMIT)
    groups.append(("battery and grid", c))

    # --- the default configuration must run everything --------------------
    c = Checker()
    c.eq("model families", cfg.model_families, ["ml", "arima", "dl"])
    c.eq("regressor subset (empty = all 15)", cfg.ml_subset, [])
    c.eq("test-window cap (0 = whole month)", cfg.max_test_days, 0)
    c.eq("configuration problems", cfg.validate(), [])
    groups.append(("default configuration runs the full study", c))

    for title, checker in groups:
        checker.report(title)

    total = sum(ch.checks for _, ch in groups)
    failed = sum(len(ch.failures) for _, ch in groups)
    print()
    if failed:
        print(f"{failed} of {total} values differ from v1.0.0.")
        return 1
    print(f"All {total} values match the archived v1.0.0 release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
