"""
Wilcoxon Signed-Rank Test: Best forecasting model vs. Seasonal Naive-168 baseline.

For each supply point, this test formally evaluates whether the best forecasting
model produces significantly lower absolute errors than the Seasonal Naive-168
baseline (prediction = same hour 7 days prior).

  H0: median(|e_model| - |e_naive168|) = 0   (no significant difference)
  SP1: median(|e_model| - |e_naive168|) < 0   (model is significantly better)

One-sided Wilcoxon Signed-Rank test (scipy.stats.wilcoxon, alternative='less').
Significance level: alpha = 0.05.

Prediction files used: Predictions_C_*_RW.csv (rolling-window validation,
development branch), checked out into each supply point's models/ subfolder.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(__file__).parent.parent

SUPPLY_POINTS = [
    {
        "name": "SP1",
        "predictions_csv": "models/SP1/Predictions_C_SP1_RW.csv",
        "history_csv": "data/consumption_SP1.csv",
        "history_col": "C_SP1",
        "best_model": "LSTM",
    },
    {
        "name": "SP2",
        "predictions_csv": "models/SP2/Predictions_C_SP2_RW.csv",
        "history_csv": "data/consumption_SP2.csv",
        "history_col": "C_SP2",
        "best_model": "Huber",
    },
    {
        "name": "SP3",
        "predictions_csv": "models/SP3/Predictions_C_SP3_RW.csv",
        "history_csv": "data/consumption_SP3.csv",
        "history_col": "C_SP3",
        "best_model": "LSTM",
    },
    {
        "name": "SP4",
        "predictions_csv": "models/SP4/Predictions_C_SP4_RW.csv",
        "history_csv": "data/consumption_SP4.csv",
        "history_col": "C_SP4",
        "best_model": "GRU",
    },
    {
        "name": "SP5",
        "predictions_csv": "models/SP5/Predictions_C_SP5_RW.csv",
        "history_csv": "data/consumption_SP5.csv",
        "history_col": "C_SP5",
        "best_model": "TCN",
    },
{
        "name": "SP6",
        "predictions_csv": "models/SP6/Predictions_C_SP6_RW.csv",
        "history_csv": "data/consumption_SP6.csv",
        "history_col": "C_SP6",
        "best_model": "GRU",
    },
]

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> pd.DataFrame:
    """Load a predictions CSV into a DatetimeIndex DataFrame."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "datetime"
    return df


def load_history(path: Path, col: str) -> pd.Series:
    """Load the historical consumption CSV, returning a single named Series."""
    df = pd.read_csv(path, parse_dates=["date_time"], index_col="date_time")
    return df[col]


def compute_naive168(index: pd.DatetimeIndex, history: pd.Series) -> pd.Series:
    """
    For each timestamp t in *index*, return history[t - 168 h].
    Missing lookups (NaN) are preserved so the caller can drop them.
    """
    lagged = index - pd.Timedelta(hours=168)
    values = history.reindex(lagged).values
    return pd.Series(values, index=index, name="Naive-168")


# ---------------------------------------------------------------------------
# Core analysis per supply point
# ---------------------------------------------------------------------------

def run_wilcoxon(sp: dict) -> dict:
    pred_path = BASE / "estudio" / sp["predictions_csv"]
    hist_path = BASE / sp["history_csv"]

    pred_df = load_predictions(pred_path)
    history = load_history(hist_path, sp["history_col"])

    observed = pred_df["Observed"]
    model_preds = pred_df[sp["best_model"]]

    naive168 = compute_naive168(observed.index, history)

    # Only keep rows where the Naive-168 lookup succeeded
    valid = naive168.notna()
    n = int(valid.sum())

    obs_v    = observed[valid].values
    model_v  = model_preds[valid].values
    naive_v  = naive168[valid].values

    ae_model = np.abs(obs_v - model_v)
    ae_naive = np.abs(obs_v - naive_v)

    # Wilcoxon on paired differences of absolute errors
    diff = ae_model - ae_naive
    stat, p_value = stats.wilcoxon(diff, alternative="less")

    mae_model = float(np.mean(ae_model))
    mae_naive = float(np.mean(ae_naive))
    reduction = (mae_naive - mae_model) / mae_naive * 100

    return {
        "Supply point":      sp["name"],
        "Best model":        sp["best_model"],
        "n":                 n,
        "MAE (model)":       round(mae_model, 4),
        "MAE (Naive-168)":   round(mae_naive, 4),
        "MAE reduction (%)": round(reduction, 1),
        "W statistic":       round(stat, 1),
        "p-value":           round(p_value, 4),
        f"Significant (alpha={ALPHA})": p_value < ALPHA,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = [run_wilcoxon(sp) for sp in SUPPLY_POINTS]
    df = pd.DataFrame(results)

    display_cols = [
        "Supply point", "Best model", "n",
        "MAE (model)", "MAE (Naive-168)", "MAE reduction (%)",
        "W statistic", "p-value", f"Significant (alpha={ALPHA})",
    ]
    print("=== Wilcoxon Signed-Rank Test: Best Model vs. Seasonal Naive-168 ===")
    print(f"H0: no difference  |  SP1: model errors < Naive-168 errors  |  alpha = {ALPHA}")
    print()
    print(df[display_cols].to_string(index=False))

    out_path = Path(__file__).parent / "wilcoxon_results.csv"
    df[display_cols].to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
