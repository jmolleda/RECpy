"""
Blocked CV — Wilcoxon pairwise model selection.

Reads the prediction CSVs produced by generate_ml_cv.py + generate_dl_cv.py,
runs the same pairwise one-sided Wilcoxon test as wilcoxon_model_selection.py,
and aggregates results across the four folds.

Per-fold output:
  cross_validation/wilcoxon_cv_per_fold.csv
  Columns: fold, supply_point, model, MAE, RMSE, wins, losses, net_wins

Aggregated output:
  cross_validation/wilcoxon_cv_aggregated.csv
  Columns: supply_point, model, mean_net_wins, std_net_wins, mean_MAE,
           mean_RMSE, folds_as_best (count of folds where model has top net wins)
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import permutations
from pathlib import Path

from recpy_config import CONFIG

# Calendar year of the dataset (anonymized placeholder;
# set this to the actual year of your data files).
YEAR = CONFIG.year

BASE_CV = CONFIG.output_root / "cross_validation"
BASE_CV.mkdir(parents=True, exist_ok=True)
ALPHA   = 0.05

FOLDS = [
    {"name": "fold1_apr", "test_start": f"{YEAR}-04-01", "test_end": f"{YEAR}-04-30"},
    {"name": "fold2_jul", "test_start": f"{YEAR}-07-01", "test_end": f"{YEAR}-07-31"},
    {"name": "fold3_oct", "test_start": f"{YEAR}-10-01", "test_end": f"{YEAR}-10-31"},
    {"name": "fold4_dec", "test_start": f"{YEAR}-12-01", "test_end": f"{YEAR}-12-31"},
]

SP_IDS = ["SP1", "SP2", "SP3", "SP4", "SP5", "SP6"]
SP_NAMES = {"SP1": "SP1", "SP2": "SP2", "SP3": "SP3",
            "SP4": "SP4", "SP5": "SP5", "SP6": "SP6"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pairwise_wilcoxon(errors: pd.DataFrame) -> pd.DataFrame:
    models = errors.columns.tolist()
    wins   = pd.DataFrame(False, index=models, columns=models)
    for a, b in permutations(models, 2):
        diff = errors[a] - errors[b]
        if (diff == 0).all():
            continue
        _, p = stats.wilcoxon(diff, alternative="less")
        wins.loc[a, b] = p < ALPHA
    return wins


def analyse_fold_sp(fold_name: str, sp_id: str) -> pd.DataFrame | None:
    csv_path = BASE_CV / fold_name / f"Predictions_C_{sp_id}_{fold_name}.csv"
    if not csv_path.exists():
        print(f"  MISSING: {csv_path.name}")
        return None

    df       = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    observed = df["Observed"]
    models   = df.drop(columns="Observed")

    ae = pd.DataFrame(
        {col: np.abs(observed.values - models[col].values) for col in models.columns}
    )
    wins_matrix = pairwise_wilcoxon(ae)

    summary = pd.DataFrame({
        "MAE":    ae.mean().round(4),
        "RMSE":   ae.pow(2).mean().pipe(np.sqrt).round(4),
        "Wins":   wins_matrix.sum(axis=1).astype(int),
        "Losses": wins_matrix.sum(axis=0).astype(int),
    })
    summary["Net wins"] = summary["Wins"] - summary["Losses"]

    rows = []
    for model, row in summary.iterrows():
        rows.append({
            "fold":         fold_name,
            "supply_point": SP_NAMES[sp_id],
            "sp_id":        sp_id,
            "model":        model,
            "MAE":          row["MAE"],
            "RMSE":         row["RMSE"],
            "wins":         row["Wins"],
            "losses":       row["Losses"],
            "net_wins":     row["Net wins"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_rows = []

    for fold in FOLDS:
        print(f"\n{'='*55}  {fold['name'].upper()}")
        for sp_id in SP_IDS:
            result = analyse_fold_sp(fold["name"], sp_id)
            if result is not None:
                all_rows.append(result)
                best = result.loc[result["net_wins"].idxmax(), "model"]
                print(f"  {SP_NAMES[sp_id]:<12} stat-best: {best}")

    per_fold = pd.concat(all_rows, ignore_index=True)
    per_fold.to_csv(BASE_CV / "wilcoxon_cv_per_fold.csv", index=False)
    print(f"\nPer-fold results saved.")

    # --- Aggregation ---
    agg = (
        per_fold
        .groupby(["supply_point", "model"])
        .agg(
            mean_net_wins=("net_wins", "mean"),
            std_net_wins =("net_wins", "std"),
            mean_MAE     =("MAE",      "mean"),
            mean_RMSE    =("RMSE",     "mean"),
        )
        .round(3)
        .reset_index()
    )

    # Count folds in which each model has the highest net wins per SP
    best_per_fold = (
        per_fold
        .loc[per_fold.groupby(["fold", "supply_point"])["net_wins"].idxmax()]
        [["fold", "supply_point", "model"]]
    )
    folds_as_best = (
        best_per_fold
        .groupby(["supply_point", "model"])
        .size()
        .reset_index(name="folds_as_best")
    )
    agg = agg.merge(folds_as_best, on=["supply_point", "model"], how="left")
    agg["folds_as_best"] = agg["folds_as_best"].fillna(0).astype(int)
    agg = agg.sort_values(["supply_point", "mean_net_wins"], ascending=[True, False])

    agg.to_csv(BASE_CV / "wilcoxon_cv_aggregated.csv", index=False)
    print("Aggregated results saved.")

    # --- Console summary ---
    print(f"\n{'='*65}")
    print("  CROSS-VALIDATION SUMMARY — Statistical model selection")
    print(f"{'='*65}")
    for sp_name in agg["supply_point"].unique():
        sp_agg = agg[agg["supply_point"] == sp_name]
        top = sp_agg.iloc[0]
        print(f"\n  {sp_name}")
        print(f"    Stat-best (avg net wins): {top['model']}  "
              f"[mean={top['mean_net_wins']:.1f}, std={top['std_net_wins']:.1f}, "
              f"best in {top['folds_as_best']}/4 folds]")
        print(f"    {'Model':<35} {'AvgNet':>7} {'Std':>6} {'AvgMAE':>8} {'Folds#':>7}")
        print(f"    {'-'*65}")
        for _, row in sp_agg.head(5).iterrows():
            print(f"    {row['model']:<35} {row['mean_net_wins']:>7.1f} "
                  f"{row['std_net_wins']:>6.1f} {row['mean_MAE']:>8.4f} "
                  f"{row['folds_as_best']:>7}")
