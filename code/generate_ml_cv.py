"""
Blocked time-series cross-validation — ML predictions.

Four expanding-window folds; each uses one full calendar month as the test set:
  Fold 1: Train 01-02 – 03-31  |  Test Apr  (720 h)
  Fold 2: Train 01-02 – 06-30  |  Test Jul  (744 h)
  Fold 3: Train 01-02 – 09-30  |  Test Oct  (744 h)
  Fold 4: Train 01-02 – 11-30  |  Test Dec  (744 h)

Replicates the exact walk-forward methodology of generate_ml_predictions.py
(univariate input, n_input=192, single-output recursive forecasting).

Output: cross_validation/fold{N}_{month}/Predictions_C_{SP}_fold{N}.csv
        Columns: Observed, <15 ML models>, Naive-168
"""

import numpy as np
import pandas as pd
from pathlib import Path
from warnings import filterwarnings

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, HuberRegressor, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
    RandomForestRegressor, GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from recpy_config import CONFIG

# All settings come from the active configuration (config/default.toml unless
# RECPY_CONFIG points elsewhere); see code/recpy_config.py.
YEAR = CONFIG.year

filterwarnings("ignore")

BASE    = Path(__file__).parent.parent
RS      = CONFIG.random_seed
N_INPUT = CONFIG.lookback_hours

# ---------------------------------------------------------------------------
# Fold definitions
# ---------------------------------------------------------------------------
FOLDS = [
    {k: f[k] for k in ("name", "train_end", "test_start", "test_end")}
    for f in CONFIG.folds
]

TRAIN_START = f"{YEAR}-01-02"

# ---------------------------------------------------------------------------
# Supply points
# ---------------------------------------------------------------------------
SUPPLY_POINTS = CONFIG.supply_points

# ---------------------------------------------------------------------------
# ML model registry
# ---------------------------------------------------------------------------
def get_models() -> dict:
    return {
        "Linear Regression":            LinearRegression(),
        "Huber":                        HuberRegressor(),
        "Stochastics Gradient Descent": SGDRegressor(random_state=RS),
        "k-Nearest Neighbors":          KNeighborsRegressor(),
        "Decision Tree":                DecisionTreeRegressor(random_state=RS),
        "Extra Tree":                   ExtraTreeRegressor(random_state=RS),
        "Support Vector":               SVR(),
        "MLP":                          MLPRegressor(random_state=RS),
        "AdaBoost":                     AdaBoostRegressor(random_state=RS),
        "Bagged Decision Trees":        BaggingRegressor(random_state=RS),
        "Random Forest":                RandomForestRegressor(n_jobs=-1, random_state=RS),
        "Extra Trees":                  ExtraTreesRegressor(n_jobs=-1, random_state=RS),
        "Gradient Boosting":            GradientBoostingRegressor(random_state=RS),
        "XGBoost":                      XGBRegressor(n_jobs=-1, random_state=RS, verbosity=0),
        "LightGBM":                     LGBMRegressor(n_jobs=-1, random_state=RS, verbose=-1),
    }

# ---------------------------------------------------------------------------
# Walk-forward helpers (mirrors notebook: re-fit on each step)
# ---------------------------------------------------------------------------
def to_supervised(data: np.ndarray, n_input: int):
    X = np.lib.stride_tricks.sliding_window_view(data, n_input)[:-1]
    y = data[n_input:]
    return X.astype(float), y.astype(float)


def forecast_day(pipeline, last_window: np.ndarray, n_input: int) -> np.ndarray:
    history = list(last_window.astype(float))
    predictions = []
    for _ in range(24):
        x = np.array(history[-n_input:], dtype=float).reshape(1, -1)
        yhat = float(pipeline.predict(x)[0])
        predictions.append(yhat)
        history.append(yhat)
    return np.array(predictions)


def walk_forward(model, train_windows: np.ndarray, test_windows: np.ndarray,
                 n_input: int) -> np.ndarray:
    # History grows after each day: re-fit model on every step, exactly as
    # the notebook does with sklearn_predict called inside the evaluation loop.
    history = list(train_windows.flatten().astype(float))
    predictions = []
    for day_obs in test_windows:
        X_train, y_train = to_supervised(np.array(history), n_input)
        pipeline = make_pipeline(model)
        pipeline.fit(X_train, y_train)
        yhat = forecast_day(pipeline, np.array(history[-n_input:]), n_input)
        predictions.append(yhat)
        history.extend(day_obs.astype(float).tolist())
    return np.concatenate(predictions)


def compute_naive168(test_index: pd.DatetimeIndex, history: pd.Series) -> np.ndarray:
    lagged = test_index - pd.Timedelta(hours=168)
    return history.reindex(lagged).values


# ---------------------------------------------------------------------------
# Per fold × supply point runner
# ---------------------------------------------------------------------------
def run_fold_sp(fold: dict, sp: dict) -> pd.DataFrame:
    series = pd.read_csv(
        BASE / sp["history_csv"],
        parse_dates=["date_time"], index_col="date_time"
    )[sp["history_col"]]

    s_train = series.loc[TRAIN_START:fold["train_end"]]
    s_test  = series.loc[fold["test_start"]:fold["test_end"]]

    w_train = np.array(np.split(s_train.values, len(s_train) // 24))
    w_test  = np.array(np.split(s_test.values,  len(s_test)  // 24))

    test_index = pd.date_range(fold["test_start"], periods=len(s_test), freq="h")
    result = pd.DataFrame({"Observed": s_test.values}, index=test_index)

    for name, model in get_models().items():
        print(f"    [{sp['id']}] {name} ...", end=" ", flush=True)
        try:
            preds = walk_forward(model, w_train, w_test, N_INPUT)
            result[name] = preds
            mae  = np.mean(np.abs(s_test.values - preds))
            rmse = np.sqrt(np.mean((s_test.values - preds) ** 2))
            print(f"MAE={mae:.4f}  RMSE={rmse:.4f}")
        except Exception as exc:
            print(f"FAILED ({exc})")

    result["Naive-168"] = compute_naive168(test_index, series)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out_base = CONFIG.output_dir("cross_validation")

    for fold in FOLDS:
        print(f"\n{'='*65}")
        print(f"  {fold['name'].upper()}  |  Test: {fold['test_start']} – {fold['test_end']}")
        print(f"{'='*65}")
        fold_dir = out_base / fold["name"]
        fold_dir.mkdir(parents=True, exist_ok=True)

        for sp in SUPPLY_POINTS:
            print(f"\n  --- {sp['name']} ({sp['id']}) ---")
            result = run_fold_sp(fold, sp)
            out_path = fold_dir / f"Predictions_C_{sp['id']}_{fold['name']}.csv"
            result.to_csv(out_path)
            n_models = len(result.columns) - 1
            print(f"  Saved -> {out_path.name}  ({n_models} models, {len(result)} rows)")

    print("\n\nDone.")
