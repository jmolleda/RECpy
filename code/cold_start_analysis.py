"""
Cold-start (limited-history) analysis.

Note on feasibility: the forecasting feature set uses an n_input = 192 h (8-day)
lookback, so a training window must exceed 8 days to yield any supervised sample.

Output: cross_validation/robustness/cold_start.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from generate_ml_cv import (
    get_models, SUPPLY_POINTS, walk_forward, N_INPUT, TRAIN_START, BASE,
)

# Calendar year of the dataset (placeholder;
# set this to the actual year of your data files).
YEAR = 2025

OUT = Path(__file__).parent / "cross_validation" / "robustness" / "cold_start.csv"

DEPLOYED = {
    "SP1": "Support Vector", "SP2": "k-Nearest Neighbors", "SP3": "Huber",
    "SP4": "Huber", "SP5": "Huber", "SP6": "Extra Trees",
}

TEST_START, TEST_END = f"{YEAR}-04-01", f"{YEAR}-04-30"
WEEK_WINDOWS = [2, 4, 8]           # + full history (Jan 2 -> Mar 31)


def train_start_for(weeks):
    return (pd.Timestamp(TEST_START) - pd.Timedelta(weeks=weeks)).strftime("%Y-%m-%d")


def evaluate(series, model, train_start):
    s_train = series.loc[train_start:f"{YEAR}-03-31"]
    s_test = series.loc[TEST_START:TEST_END]
    w_train = np.array(np.split(s_train.values, len(s_train) // 24))
    w_test = np.array(np.split(s_test.values, len(s_test) // 24))
    preds = walk_forward(model, w_train, w_test, N_INPUT)
    obs = s_test.values
    ae = np.abs(obs - preds)
    mae = float(np.mean(ae))
    mae_wk1 = float(np.mean(ae[:168]))     # first 7 days = early cold start
    train_days = len(s_train) // 24
    return mae, mae_wk1, train_days


def main():
    models = get_models()
    rows = []
    for sp in SUPPLY_POINTS:
        name = DEPLOYED[sp["id"]]
        series = pd.read_csv(BASE / sp["history_csv"],
                             parse_dates=["date_time"], index_col="date_time")[sp["history_col"]]
        print(f"\n--- {sp['name']} ({sp['id']}) : {name} ---")
        windows = [(f"{w}w", train_start_for(w)) for w in WEEK_WINDOWS]
        windows.append(("full", TRAIN_START))
        base_mae = None
        for label, tstart in windows:
            mae, mae_wk1, tdays = evaluate(series, models[name], tstart)
            if label == "full":
                base_mae = mae
            rows.append({
                "supply_point": sp["name"], "model": name, "window": label,
                "train_days": tdays, "MAE": round(mae, 4),
                "MAE_week1": round(mae_wk1, 4),
            })
            print(f"   {label:<5} train_days={tdays:<3} MAE={mae:.4f}  MAE(wk1)={mae_wk1:.4f}")
        # penalty of smallest window vs full
        two_wk = next(r for r in rows if r["supply_point"] == sp["name"] and r["window"] == "2w")
        pen = (two_wk["MAE"] - base_mae) / base_mae * 100
        print(f"   2w vs full MAE penalty: {pen:+.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n=== Cold-start MAE penalty vs full history ===")
    piv = df.pivot(index="supply_point", columns="window", values="MAE")
    piv = piv[["2w", "4w", "8w", "full"]]
    piv["2w_pct"] = ((piv["2w"] - piv["full"]) / piv["full"] * 100).round(1)
    piv["4w_pct"] = ((piv["4w"] - piv["full"]) / piv["full"] * 100).round(1)
    print(piv.to_string())
    print(f"\nMean MAE penalty: 2w {piv['2w_pct'].mean():+.1f}%,  4w {piv['4w_pct'].mean():+.1f}%")
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
