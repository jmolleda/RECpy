"""
Pairwise Wilcoxon Signed-Rank Test for statistical model selection.

For supply points that have multiple candidate models in their _RW prediction
file, this script runs all pairwise one-sided Wilcoxon tests and builds a
dominance matrix. Naive-168 (same hour, 7 days prior) is included as a
reference baseline in every comparison.

For each ordered pair (A, B):
  d_i = |e_A_i| - |e_B_i|
  H0: median(d) = 0
  SP1: median(d) < 0   ->  A is significantly better than B  (p < alpha)

Summary per supply point:
  - Dominance matrix  (row beats column)
  - Win count + MAE + RMSE per model
  - Statistically-selected best model vs. RMSE-selected best model
"""

import pandas as pd
import numpy as np
from scipy import stats
from itertools import permutations
from pathlib import Path

from recpy_config import CONFIG

BASE  = Path(__file__).parent.parent 
ALPHA = 0.05

# ---------------------------------------------------------------------------
# All supply points — using _ALL files (15 ML models + DL from _RW + Naive-168)
# ---------------------------------------------------------------------------
SUPPLY_POINTS = [
    {
        "name":            "SP1",
        "predictions_csv": "models/SP1/Predictions_C_SP1_ALL.csv",
        "history_csv":     "data/consumption_SP1.csv",
        "history_col":     "C_SP1",
        "rmse_best":       "LSTM",
    },
    {
        "name":            "SP2",
        "predictions_csv": "models/SP2/Predictions_C_SP2_ALL.csv",
        "history_csv":     "data/consumption_SP2.csv",
        "history_col":     "C_SP2",
        "rmse_best":       "Huber",
    },
    {
        "name":            "SP3",
        "predictions_csv": "models/SP3/Predictions_C_SP3_ALL.csv",
        "history_csv":     "data/consumption_SP3.csv",
        "history_col":     "C_SP3",
        "rmse_best":       "LSTM",
    },
    {
        "name":            "SP4",
        "predictions_csv": "models/SP4/Predictions_C_SP4_ALL.csv",
        "history_csv":     "data/consumption_SP4.csv",
        "history_col":     "C_SP4",
        "rmse_best":       "GRU",
    },
    {
        "name":            "SP5",
        "predictions_csv": "models/SP5/Predictions_C_SP5_ALL.csv",
        "history_csv":     "data/consumption_SP5.csv",
        "history_col":     "C_SP5",
        "rmse_best":       "TCN",
    },
{
        "name":            "SP6",
        "predictions_csv": "models/SP6/Predictions_C_SP6_ALL.csv",
        "history_csv":     "data/consumption_SP6.csv",
        "history_col":     "C_SP6",
        "rmse_best":       "GRU",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "datetime"
    return df


def load_history(path: Path, col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date_time"], index_col="date_time")
    return df[col]


def compute_naive168(index: pd.DatetimeIndex, history: pd.Series) -> pd.Series:
    lagged = index - pd.Timedelta(hours=168)
    values = history.reindex(lagged).values
    return pd.Series(values, index=index, name="Naive-168")


def pairwise_wilcoxon(errors: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Run one-sided Wilcoxon for every ordered pair of models.
    Returns a boolean DataFrame: wins[A][B] = True means A significantly
    beats B (p < alpha).
    """
    models = errors.columns.tolist()
    wins = pd.DataFrame(False, index=models, columns=models)
    for a, b in permutations(models, 2):
        diff = errors[a] - errors[b]
        # Skip if all differences are zero (identical predictions)
        if (diff == 0).all():
            continue
        _, p = stats.wilcoxon(diff, alternative="less")
        wins.loc[a, b] = p < alpha
    return wins


# ---------------------------------------------------------------------------
# Per-supply-point analysis
# ---------------------------------------------------------------------------

def analyse(sp: dict) -> dict:
    pred_df  = load_predictions(BASE / "estudio" / sp["predictions_csv"])
    history  = load_history(BASE / sp["history_csv"], sp["history_col"])
    observed = pred_df["Observed"]
    models   = pred_df.drop(columns="Observed")

    # Add Naive-168
    naive = compute_naive168(observed.index, history)
    valid = naive.notna()
    models["Naive-168"] = naive

    # Absolute errors — valid rows only
    obs_v = observed[valid].values
    ae = pd.DataFrame(
        {col: np.abs(obs_v - models[col][valid].values) for col in models.columns}
    )

    wins_matrix = pairwise_wilcoxon(ae, ALPHA)

    # Per-model summary
    summary = pd.DataFrame({
        "MAE":  ae.mean().round(4),
        "RMSE": ae.pow(2).mean().pipe(np.sqrt).round(4),
        "Wins": wins_matrix.sum(axis=1).astype(int),
        "Losses": wins_matrix.sum(axis=0).astype(int),
    })
    summary["Net wins"] = summary["Wins"] - summary["Losses"]
    summary = summary.sort_values(["Net wins", "MAE"], ascending=[False, True])

    stat_best  = summary.index[0]
    rmse_best  = sp["rmse_best"]
    same       = stat_best == rmse_best

    return {
        "name":         sp["name"],
        "rmse_best":    rmse_best,
        "stat_best":    stat_best,
        "same":         same,
        "summary":      summary,
        "wins_matrix":  wins_matrix,
        "n":            int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_dominance_matrix(wins: pd.DataFrame, rmse_best: str, stat_best: str):
    models = wins.index.tolist()
    col_w  = max(len(m) for m in models) + 1

    # Header
    header = f"{'':>{col_w}} |"
    for m in models:
        header += f" {m[:6]:>6} |"
    print(header)
    print("-" * len(header))

    for row in models:
        line = f"{row:>{col_w}} |"
        for col in models:
            if row == col:
                cell = "  --  "
            elif wins.loc[row, col]:
                cell = "  W   "
            else:
                cell = "  .   "
            line += f"{cell}|"
        # Annotate row
        tags = []
        if row == rmse_best:
            tags.append("RMSE-best")
        if row == stat_best:
            tags.append("Stat-best")
        if tags:
            line += "  <- " + ", ".join(tags)
        print(line)

    print()
    print(f"  W = significantly better (p < {ALPHA})   . = no significant difference")


def print_summary_table(summary: pd.DataFrame, rmse_best: str, stat_best: str):
    print(f"\n{'Model':<35} {'MAE':>7} {'RMSE':>7} {'Wins':>5} {'Losses':>7} {'Net':>5}")
    print("-" * 65)
    for model, row in summary.iterrows():
        tags = []
        if model == rmse_best:
            tags.append("RMSE-best")
        if model == stat_best and model != rmse_best:
            tags.append("Stat-best")
        tag_str = f"  <- {', '.join(tags)}" if tags else ""
        print(
            f"{model:<35} {row['MAE']:>7.4f} {row['RMSE']:>7.4f} "
            f"{row['Wins']:>5} {row['Losses']:>7} {row['Net wins']:>5}{tag_str}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = [analyse(sp) for sp in SUPPLY_POINTS]

    for r in all_results:
        print("=" * 70)
        print(f"  {r['name']}  (n={r['n']} hourly observations)")
        print("=" * 70)

        print("\n-- Dominance matrix (row beats column) --\n")
        print_dominance_matrix(r["wins_matrix"], r["rmse_best"], r["stat_best"])

        print("\n-- Model ranking --")
        print_summary_table(r["summary"], r["rmse_best"], r["stat_best"])

        print(f"\n  RMSE-selected : {r['rmse_best']}")
        print(f"  Stat-selected : {r['stat_best']}", end="")
        if r["same"]:
            print("  (same model)")
        else:
            print("  *** DIFFERENT ***")
        print()

    # --- Cross-supply-point summary ---
    print("=" * 70)
    print("  SUMMARY: RMSE selection vs. statistical selection")
    print("=" * 70)
    print(f"\n{'Supply point':<15} {'RMSE-best':<35} {'Stat-best':<35} {'Same?'}")
    print("-" * 90)
    for r in all_results:
        same_str = "Yes" if r["same"] else "No  ***"
        print(f"{r['name']:<15} {r['rmse_best']:<35} {r['stat_best']:<35} {same_str}")

    # Save full results to CSV
    rows = []
    for r in all_results:
        for model, mrow in r["summary"].iterrows():
            rows.append({
                "Supply point": r["name"],
                "Model": model,
                "MAE": mrow["MAE"],
                "RMSE": mrow["RMSE"],
                "Wins": mrow["Wins"],
                "Losses": mrow["Losses"],
                "Net wins": mrow["Net wins"],
                "RMSE-selected": model == r["rmse_best"],
                "Stat-selected": model == r["stat_best"],
            })
    out = CONFIG.output_root / "wilcoxon_model_selection_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nDetailed results saved to: {out}")
