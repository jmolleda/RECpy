"""
Forecast-combination analysis.

For each supply point we compare the single Wilcoxon-selected model (top-1 by
mean net wins) against three forecast-combination strategies built from the
top-k models by mean net wins:
  * top-3 equal-weight average
  * top-5 equal-weight average
  * top-5 stacking meta-learner (non-negative least squares), trained
    leave-one-fold-out so the weights only ever see other folds.

Metric: MAE / RMSE averaged over the four seasonal folds (hourly errors).

Output: cross_validation/robustness/ensemble_comparison.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import nnls

from recpy_config import CONFIG

BASE_CV = CONFIG.output_root / "cross_validation"
AGG = BASE_CV / "wilcoxon_cv_aggregated.csv"
OUT = BASE_CV / "robustness" / "ensemble_comparison.csv"

FOLDS = [f["name"] for f in CONFIG.folds]
SP_IDS = ["SP1", "SP2", "SP3", "SP4", "SP5", "SP6"]
SP_NAMES = {"SP1": "SP1", "SP2": "SP2", "SP3": "SP3",
            "SP4": "SP4", "SP5": "SP5", "SP6": "SP6"}


def load_fold(sp_id, fold):
    df = pd.read_csv(BASE_CV / fold / f"Predictions_C_{sp_id}_{fold}.csv",
                     index_col=0, parse_dates=True)
    return df


def mae_rmse(pred, obs):
    e = pred - obs
    return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))


def main():
    agg = pd.read_csv(AGG)
    rows = []

    for sp_id in SP_IDS:
        sp = SP_NAMES[sp_id]
        ranking = (agg[agg.supply_point == sp]
                   .sort_values("mean_net_wins", ascending=False)["model"].tolist())

        folds_data = {f: load_fold(sp_id, f) for f in FOLDS}
        # keep only models finite in every fold (drops e.g. SGD divergence)
        finite = [m for m in ranking
                  if all(m in folds_data[f].columns and
                         np.isfinite(folds_data[f][m].to_numpy()).all() for f in FOLDS)]
        winner = finite[0]
        top3, top5 = finite[:3], finite[:5]

        def eval_equal(models):
            maes, rmses = [], []
            for f in FOLDS:
                df = folds_data[f]
                pred = df[models].mean(axis=1).to_numpy()
                m, r = mae_rmse(pred, df["Observed"].to_numpy())
                maes.append(m); rmses.append(r)
            return np.mean(maes), np.mean(rmses)

        def eval_stack(models):
            maes, rmses = [], []
            for held in FOLDS:
                train = [f for f in FOLDS if f != held]
                X = np.vstack([folds_data[f][models].to_numpy() for f in train])
                y = np.concatenate([folds_data[f]["Observed"].to_numpy() for f in train])
                w, _ = nnls(X, y)
                dfh = folds_data[held]
                pred = dfh[models].to_numpy() @ w
                m, r = mae_rmse(pred, dfh["Observed"].to_numpy())
                maes.append(m); rmses.append(r)
            return np.mean(maes), np.mean(rmses)

        win_mae, win_rmse = eval_equal([winner])
        t3_mae, t3_rmse = eval_equal(top3)
        t5_mae, t5_rmse = eval_equal(top5)
        st_mae, st_rmse = eval_stack(top5)

        rows.append({
            "supply_point": sp,
            "winner": winner,
            "winner_MAE": round(win_mae, 4),
            "top3avg_MAE": round(t3_mae, 4),
            "top5avg_MAE": round(t5_mae, 4),
            "stack5_MAE": round(st_mae, 4),
            "best_MAE": min([("winner", win_mae), ("top3avg", t3_mae),
                             ("top5avg", t5_mae), ("stack5", st_mae)],
                            key=lambda x: x[1])[0],
            "winner_RMSE": round(win_rmse, 4),
            "top3avg_RMSE": round(t3_rmse, 4),
            "top5avg_RMSE": round(t5_rmse, 4),
            "stack5_RMSE": round(st_rmse, 4),
        })
        print(f"{sp:<11} winner={winner:<20} "
              f"MAE win={win_mae:.4f} t3={t3_mae:.4f} t5={t5_mae:.4f} stack={st_mae:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # how often does each strategy give the lowest MAE?
    print("\nLowest-MAE strategy per SP:")
    print(df["best_MAE"].value_counts().to_string())
    # mean MAE improvement of best combination over the single winner
    df["best_combo_MAE"] = df[["top3avg_MAE", "top5avg_MAE", "stack5_MAE"]].min(axis=1)
    impr = (df["winner_MAE"] - df["best_combo_MAE"]) / df["winner_MAE"] * 100
    print(f"\nBest-combination vs winner MAE change (mean over SPs): "
          f"{impr.mean():+.1f}%  (positive = combination better)")
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
