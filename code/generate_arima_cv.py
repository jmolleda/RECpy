"""
Blocked time-series cross-validation — ARIMA and SARIMAX predictions.

Per-supply-point model orders were selected by AIC grid search. 
Four expanding-window folds as generate_ml_cv.py are used.

Walk-forward protocol:
  - ARIMA: full refit at every test step on the expanding history (fast, ~1-5 s/fit).
  - SARIMAX: refit at fold start; subsequent steps use statsmodels .append()
    with refit=False to propagate the Kalman filter state without re-estimating
    parameters. This is the standard online-update procedure for state-space models
    and is orders of magnitude faster than daily full refitting on large datasets.

Output: appends ARIMA and SARIMAX columns to the existing fold CSVs produced by
        generate_ml_cv.py:
        cross_validation/fold{N}_{month}/Predictions_C_{SP}_fold{N}.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from warnings import filterwarnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_MODEL

from recpy_config import CONFIG

# All settings come from the active configuration (config/default.toml unless
# RECPY_CONFIG points elsewhere); see code/recpy_config.py.
YEAR = CONFIG.year

filterwarnings("ignore")

BASE        = Path(__file__).parent.parent
METEO_CSV   = "data/consumption_meteo_calendar.csv"
TRAIN_START = CONFIG.train_start

# ---------------------------------------------------------------------------
# Fold definitions (identical to generate_ml_cv.py)
# ---------------------------------------------------------------------------
FOLDS = [
    {k: f[k] for k in ("name", "train_end", "test_start", "test_end")}
    for f in CONFIG.folds
]

# ---------------------------------------------------------------------------
# Supply-point registry — orders from AIC grid search in exploratory notebooks
# ---------------------------------------------------------------------------
SUPPLY_POINTS = CONFIG.supply_points_arima()

# ---------------------------------------------------------------------------
# Walk-forward helpers
# ---------------------------------------------------------------------------
def walk_forward_arima(train_series: pd.Series, test_series: pd.Series,
                       order: tuple) -> np.ndarray:
    """Full daily refit on expanding history."""
    history = list(train_series.values)
    preds   = []
    n_days  = len(test_series) // 24

    for day_idx in range(n_days):
        day_obs = test_series.iloc[day_idx * 24 : (day_idx + 1) * 24].values
        try:
            fit  = ARIMA(history, order=order).fit()
            yhat = fit.forecast(steps=24)
        except Exception:
            # Fallback: persist last 24 h
            yhat = np.array(history[-24:])
        yhat = np.maximum(yhat, 0.0)
        preds.extend(yhat)
        history.extend(day_obs)

        if (day_idx + 1) % 5 == 0 or (day_idx + 1) == n_days:
            print(f"      step {day_idx + 1}/{n_days}", end="\r", flush=True)

    print()
    return np.array(preds)


def walk_forward_sarimax(train_series: pd.Series, test_series: pd.Series,
                         train_exog: pd.DataFrame, test_exog: pd.DataFrame,
                         order: tuple, seasonal_order: tuple) -> np.ndarray:
    """
    Refit once at fold start; use .append(refit=False) for daily state updates.
    Exogenous variables for the forecast period are taken from the known meteo data.
    """
    preds  = []
    n_days = len(test_series) // 24

    # Initial fit on training data
    try:
        model  = SARIMAX_MODEL(train_series, exog=train_exog,
                               order=order, seasonal_order=seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
        result = model.fit(disp=False, maxiter=200)
    except Exception as exc:
        print(f"      SARIMAX initial fit FAILED ({exc}) — skipping")
        return np.full(len(test_series), np.nan)

    for day_idx in range(n_days):
        obs_slice  = test_series.iloc[day_idx * 24 : (day_idx + 1) * 24]
        fcast_exog = test_exog.iloc[(day_idx + 1) * 24 : (day_idx + 2) * 24]

        # Forecast next 24 h using exog for forecast period
        if day_idx < n_days - 1:
            try:
                yhat = result.forecast(steps=24, exog=fcast_exog)
            except Exception:
                yhat = np.full(24, np.nan)
        else:
            # Last day: no future exog needed (not used further)
            try:
                yhat = result.forecast(steps=24, exog=fcast_exog if len(fcast_exog) == 24 else test_exog.iloc[-24:])
            except Exception:
                yhat = np.full(24, np.nan)

        yhat = np.maximum(np.array(yhat, dtype=float), 0.0)
        preds.extend(yhat)

        # Update state with observed day (no parameter re-estimation)
        obs_exog = test_exog.iloc[day_idx * 24 : (day_idx + 1) * 24]
        try:
            result = result.append(obs_slice, exog=obs_exog, refit=False)
        except Exception:
            pass  # keep previous result state

        if (day_idx + 1) % 5 == 0 or (day_idx + 1) == n_days:
            print(f"      step {day_idx + 1}/{n_days}", end="\r", flush=True)

    print()
    return np.array(preds)


# ---------------------------------------------------------------------------
# Per fold × supply point runner
# ---------------------------------------------------------------------------
def run_fold_sp(fold: dict, sp: dict, data: pd.DataFrame, out_dir: Path) -> None:
    sp_id = sp["id"]
    col   = sp["col"]

    train_series = data.loc[TRAIN_START:fold["train_end"], col]
    test_series  = CONFIG.limit_test(
        data.loc[fold["test_start"]:fold["test_end"], col])

    csv_path = out_dir / f"Predictions_C_{sp_id}_{fold['name']}.csv"
    all_df   = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # ---- ARIMA ----
    col_arima = "ARIMA"
    if col_arima in all_df.columns:
        print(f"    [{sp_id}] ARIMA already present — skipping")
    else:
        print(f"    [{sp_id}] ARIMA{sp['arima_order']} ({len(test_series)//24} steps) ...")
        preds = walk_forward_arima(train_series, test_series, sp["arima_order"])
        mae  = mean_absolute_error(test_series.values, preds)
        rmse = np.sqrt(mean_squared_error(test_series.values, preds))
        print(f"      MAE={mae:.4f}  RMSE={rmse:.4f}")
        all_df[col_arima] = preds

    # ---- SARIMAX ----
    col_sarimax = "SARIMAX"
    if col_sarimax in all_df.columns:
        print(f"    [{sp_id}] SARIMAX already present — skipping")
    else:
        exog_cols = sp["sarimax_exog"]
        train_exog = data.loc[TRAIN_START:fold["train_end"], exog_cols]
        test_exog  = CONFIG.limit_test(
            data.loc[fold["test_start"]:fold["test_end"], exog_cols])

        print(f"    [{sp_id}] SARIMAX{sp['sarimax_order']}x{sp['sarimax_seasonal']} ({len(test_series)//24} steps) ...")
        preds = walk_forward_sarimax(train_series, test_series,
                                     train_exog, test_exog,
                                     sp["sarimax_order"], sp["sarimax_seasonal"])
        if not np.all(np.isnan(preds)):
            mae  = mean_absolute_error(test_series.values[~np.isnan(preds)],
                                       preds[~np.isnan(preds)])
            rmse = np.sqrt(mean_squared_error(test_series.values[~np.isnan(preds)],
                                              preds[~np.isnan(preds)]))
            print(f"      MAE={mae:.4f}  RMSE={rmse:.4f}")
        all_df[col_sarimax] = preds

    all_df.to_csv(csv_path)
    n_models = len(all_df.columns) - 1
    print(f"    [{sp_id}] Updated -> {csv_path.name}  ({n_models} models)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = pd.read_csv(
        BASE / METEO_CSV,
        sep=",", decimal=".",
        index_col=0, parse_dates=["date_time"]
    )

    out_base = CONFIG.output_dir("cross_validation")

    for fold in FOLDS:
        print(f"\n{'='*65}")
        print(f"  {fold['name'].upper()}  |  Test: {fold['test_start']} – {fold['test_end']}")
        print(f"{'='*65}")
        fold_dir = out_base / fold["name"]

        for sp in SUPPLY_POINTS:
            print(f"\n  --- {sp['id']} ---")
            run_fold_sp(fold, sp, data, fold_dir)

    print("\n\nDone.")
