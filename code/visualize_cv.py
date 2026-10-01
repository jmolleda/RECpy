"""
Cross-validation visualizations.

Reads wilcoxon_cv_per_fold.csv and wilcoxon_cv_aggregated.csv and produces
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from recpy_config import CONFIG

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_CV  = CONFIG.output_root / "cross_validation"
FIG_DIR  = BASE_CV / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PER_FOLD = BASE_CV / "wilcoxon_cv_per_fold.csv"
AGG      = BASE_CV / "wilcoxon_cv_aggregated.csv"

FOLD_LABELS = {
    "fold1_apr": "Apr\n(Fold 1)",
    "fold2_jul": "Jul\n(Fold 2)",
    "fold3_oct": "Oct\n(Fold 3)",
    "fold4_dec": "Dec\n(Fold 4)",
}
FOLD_SHORT = {
    "fold1_apr": "Apr", "fold2_jul": "Jul",
    "fold3_oct": "Oct", "fold4_dec": "Dec",
}
SP_ORDER  = ["SP1", "SP2", "SP3", "SP4", "SP5", "SP6"]
SP_SHORT  = {"SP1": "SP1", "SP2": "SP2", "SP3": "SP3",
             "SP4": "SP4", "SP5": "SP5", "SP6": "SP6"}
SP_TAG    = {"SP1": "SP1", "SP2": "SP2", "SP3": "SP3",
             "SP4": "SP4", "SP5": "SP5", "SP6": "SP6"}

# Display-name mapping for figure labels (keeps CSV column names unchanged)
MODEL_DISPLAY = {
    "Linear Regression":           "LR",
    "Huber":                       "Huber",
    "Stochastics Gradient Descent":"SGD",
    "k-Nearest Neighbors":         "k-NN",
    "Decision Tree":               "DT",
    "Extra Tree":                  "ET",
    "Support Vector":              "SVR",
    "MLP":                         "MLP",
    "AdaBoost":                    "AdaBoost",
    "Bagged Decision Trees":       "BDT",
    "Random Forest":               "RF",
    "Extra Trees":                 "ETs",
    "Gradient Boosting":           "GBM",
    "XGBoost":                     "XGBoost",
    "LightGBM":                    "LightGBM",
    "Naive-168":                   "Naive-168",
    "ARIMA":                       "ARIMA",
    "SARIMAX":                     "SARIMAX",
    "LSTM_single":                 "LSTM_s",
    "GRU_single":                  "GRU_s",
    "TCN_single":                  "TCN_s",
    "LSTM_daily":                  "LSTM_d",
    "GRU_daily":                   "GRU_d",
    "TCN_daily":                   "TCN_d",
}


# Figures use the full model names; only the source-CSV typo is corrected.
NAME_FIX = {"Stochastics Gradient Descent": "Stochastic Gradient Descent"}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    pf  = pd.read_csv(PER_FOLD)
    agg = pd.read_csv(AGG)
    pf["model"]  = pf["model"].replace(NAME_FIX)
    agg["model"] = agg["model"].replace(NAME_FIX)
    return pf, agg


# ---------------------------------------------------------------------------
# Figure 1 — Net wins heatmap (2×3 grid, one panel per SP)
# ---------------------------------------------------------------------------
def fig1_heatmap(pf: pd.DataFrame):
    from matplotlib.colors import LinearSegmentedColormap

    pastel_cmap = LinearSegmentedColormap.from_list(
        "pastel_RdYlGn",
        [(0.0, "#e8a090"), (0.5, "#fffacd"), (1.0, "#90c890")]
    )
    VMIN, VMAX = -20, 20

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    folds = ["fold1_apr", "fold2_jul", "fold3_oct", "fold4_dec"]

    for idx, sp_name in enumerate(SP_ORDER):
        ax  = axes[idx]
        sp_df = pf[pf["supply_point"] == sp_name]

        pivot = sp_df.pivot_table(
            index="model", columns="fold", values="net_wins", aggfunc="first"
        ).reindex(columns=folds)
        pivot = pivot.sort_values(folds[0], ascending=False)

        best_per_fold = {
            fold: sp_df[sp_df["fold"] == fold].sort_values("net_wins", ascending=False).iloc[0]["model"]
            for fold in folds if fold in sp_df["fold"].values
        }

        im = ax.imshow(pivot.values, cmap=pastel_cmap, aspect="auto",
                       vmin=VMIN, vmax=VMAX)
        ax.set_xticks(range(len(folds)))
        ax.set_xticklabels([FOLD_SHORT[f] for f in folds], fontsize=9)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=9, fontweight="bold")
        ax.set_title(SP_TAG[sp_name], fontsize=11, fontweight="bold")

        # Annotate cells
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{int(val)}", ha="center", va="center",
                            fontsize=7, color="black")

        # Mark stat-best per fold with a border
        for c, fold in enumerate(folds):
            best_model = best_per_fold.get(fold)
            if best_model and best_model in pivot.index:
                r = pivot.index.tolist().index(best_model)
                ax.add_patch(plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96,
                                           fill=False, edgecolor="navy", lw=2))

    # Single shared colorbar on the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=pastel_cmap,
                                norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Net wins")

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig1_net_wins_heatmap.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig1_net_wins_heatmap saved.")


# ---------------------------------------------------------------------------
# Figure 2 — Average net wins bar chart (one per SP, 2×3 grid)
# ---------------------------------------------------------------------------
def fig2_avg_ranking(agg: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, sp_name in enumerate(SP_ORDER):
        ax     = axes[idx]
        sp_agg = agg[agg["supply_point"] == sp_name].copy()
        sp_agg = sp_agg.sort_values("mean_net_wins", ascending=True)

        colors = ["#2166ac" if v >= 0 else "#d6604d" for v in sp_agg["mean_net_wins"]]
        bars = ax.barh(sp_agg["model"], sp_agg["mean_net_wins"],
                       xerr=sp_agg["std_net_wins"], capsize=3,
                       color=colors, alpha=0.85, error_kw={"elinewidth": 1})

        # Mark models that are best in ≥1 fold
        for i, (_, row) in enumerate(sp_agg.iterrows()):
            if row["folds_as_best"] > 0:
                ax.text(sp_agg["mean_net_wins"].max() * 0.05,
                        i, f"★ {row['folds_as_best']}/4",
                        va="center", fontsize=7, color="white",
                        fontweight="bold")

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-25, 25)
        ax.set_title(SP_TAG[sp_name], fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Net Wins (±std)", fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        for _t in ax.get_yticklabels():
            _t.set_fontweight("bold")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig2_average_ranking.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig2_average_ranking saved.")


# ---------------------------------------------------------------------------
# Figure 3 — RMSE rank vs Wilcoxon rank scatter
# ---------------------------------------------------------------------------
def fig3_rmse_vs_wilcoxon(pf: pd.DataFrame):
    rows = []
    for (fold, sp), grp in pf.groupby(["fold", "supply_point"]):
        n = len(grp)
        grp = grp.copy()
        grp["rmse_rank"]     = grp["RMSE"].rank()
        grp["wilcoxon_rank"] = grp["net_wins"].rank(ascending=False)
        for _, row in grp.iterrows():
            rows.append({
                "fold": FOLD_SHORT[fold], "sp": SP_TAG[sp],
                "model": row["model"],
                "rmse_rank": row["rmse_rank"],
                "wilcoxon_rank": row["wilcoxon_rank"],
            })
    scatter_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 7))
    sp_colors = {SP_TAG[sp]: c for sp, c in
                 zip(SP_ORDER, plt.cm.tab10.colors)}

    for sp_name, grp in scatter_df.groupby("sp"):
        ax.scatter(grp["rmse_rank"], grp["wilcoxon_rank"],
                   label=sp_name, alpha=0.55, s=40,
                   color=sp_colors.get(sp_name, "grey"))

    # Perfect agreement diagonal
    lim = scatter_df[["rmse_rank", "wilcoxon_rank"]].max().max() + 1
    ax.plot([1, lim], [1, lim], "k--", linewidth=0.8, label="Perfect agreement")

    ax.set_xlabel("RMSE rank (1 = lowest RMSE)", fontsize=11)
    ax.set_ylabel("Wilcoxon rank (1 = highest net wins)", fontsize=11)
    ax.set_title("RMSE rank vs. Wilcoxon rank\n(all folds × all supply points)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Supply point", fontsize=8, title_fontsize=9)
    ax.grid(True, alpha=0.3)

    from scipy.stats import spearmanr
    corr, _ = spearmanr(scatter_df["rmse_rank"], scatter_df["wilcoxon_rank"])
    ax.text(0.97, 0.05, f"Spearman ρ = {corr:.2f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig3_rmse_vs_wilcoxon.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig3_rmse_vs_wilcoxon saved.")


# ---------------------------------------------------------------------------
# Figure 4 — Bump chart: model rank trajectory across folds (2×3 grid)
# ---------------------------------------------------------------------------
def fig4_bump_chart(pf: pd.DataFrame):
    folds = ["fold1_apr", "fold2_jul", "fold3_oct", "fold4_dec"]
    fold_labels = ["Apr", "Jul", "Oct", "Dec"]
    TOP_N = 8  # show only top-N models by mean net wins to avoid clutter

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, sp_name in enumerate(SP_ORDER):
        ax    = axes[idx]
        sp_df = pf[pf["supply_point"] == sp_name].copy()

        # Rank within each fold (1 = highest net wins)
        sp_df["rank"] = sp_df.groupby("fold")["net_wins"].rank(
            ascending=False, method="min"
        )

        # Select top-N models by mean rank across folds
        mean_rank = sp_df.groupby("model")["rank"].mean().sort_values()
        top_models = mean_rank.head(TOP_N).index.tolist()

        cmap      = plt.cm.tab10
        colors    = {m: cmap(i / TOP_N) for i, m in enumerate(top_models)}
        stat_best = (sp_df.groupby("model")["net_wins"]
                     .mean().idxmax())

        for model in top_models:
            m_df = (sp_df[sp_df["model"] == model]
                    .set_index("fold")
                    .reindex(folds)
                    .reset_index())
            is_best = (model == stat_best)
            ax.plot(fold_labels, m_df["rank"].values,
                    marker="*" if is_best else "o",
                    linewidth=3.5 if is_best else 1.5,
                    markersize=12 if is_best else 5,
                    color=colors[model],
                    label=f"★ {model}" if is_best else model,
                    zorder=5 if is_best else 2)

        ax.invert_yaxis()
        ax.set_yticks(range(1, TOP_N + 1))
        ax.set_ylabel("Wilcoxon rank (1 = best)", fontsize=8)
        ax.set_title(SP_TAG[sp_name], fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right", ncol=1)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig4_bump_chart.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig4_bump_chart saved.")


# ---------------------------------------------------------------------------
# Figure 5 — Stat-best vs RMSE-best agreement heatmap (6 SP × 4 folds)
# ---------------------------------------------------------------------------
def fig5_agreement_heatmap(pf: pd.DataFrame):
    folds = ["fold1_apr", "fold2_jul", "fold3_oct", "fold4_dec"]
    fold_labels = ["Apr\n(Fold 1)", "Jul\n(Fold 2)", "Oct\n(Fold 3)", "Dec\n(Fold 4)"]

    agree_matrix   = np.zeros((len(SP_ORDER), len(folds)))
    stat_labels    = [[""]*len(folds) for _ in SP_ORDER]
    rmse_labels    = [[""]*len(folds) for _ in SP_ORDER]

    for c, fold in enumerate(folds):
        fold_df = pf[pf["fold"] == fold]
        for r, sp_name in enumerate(SP_ORDER):
            sp_df = fold_df[fold_df["supply_point"] == sp_name]
            if sp_df.empty:
                continue
            stat_best = sp_df.loc[sp_df["net_wins"].idxmax(), "model"]
            rmse_best = sp_df.loc[sp_df["RMSE"].idxmin(), "model"]
            agree_matrix[r, c] = 1 if stat_best == rmse_best else 0
            stat_labels[r][c]  = stat_best
            rmse_labels[r][c]  = rmse_best

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap_agree = plt.cm.colors if hasattr(plt.cm, "colors") else None
    agree_cmap = plt.matplotlib.colors.ListedColormap(["#e8a090", "#90c890"])
    im = ax.imshow(agree_matrix, cmap=agree_cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels(fold_labels, fontsize=10)
    ax.set_yticks(range(len(SP_ORDER)))
    ax.set_yticklabels(
        [SP_TAG[sp] for sp in SP_ORDER], fontsize=10
    )

    for r in range(len(SP_ORDER)):
        for c in range(len(folds)):
            stat = stat_labels[r][c]
            rmse = rmse_labels[r][c]
            if stat == rmse:
                ax.text(c, r, stat, ha="center", va="center",
                        fontsize=7, fontweight="bold", color="black")
            else:
                ax.text(c, r, f"W: {stat}\nR: {rmse}",
                        ha="center", va="center", fontsize=6, color="black")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#90c890", label="Agreement (✓)"),
        Patch(facecolor="#e8a090", label="Disagreement (✗)  W=Wilcoxon  R=RMSE"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1.0, -0.15), fontsize=9, ncol=2)

    n_agree = int(agree_matrix.sum())
    n_total = int(np.count_nonzero(agree_matrix >= 0))
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig5_agreement_heatmap.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig5_agreement_heatmap saved.")


# ---------------------------------------------------------------------------
# Figure 6 — MAE box plots: top-5 models per SP across 4 folds (2×3 grid)
# ---------------------------------------------------------------------------
def fig6_mae_boxplots(pf: pd.DataFrame, agg: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    for idx, sp_name in enumerate(SP_ORDER):
        ax     = axes[idx]
        sp_agg = agg[agg["supply_point"] == sp_name].sort_values(
            "mean_net_wins", ascending=False
        )
        top8      = sp_agg.head(8)["model"].tolist()
        stat_best = sp_agg.iloc[0]["model"]

        sp_pf  = pf[pf["supply_point"] == sp_name]
        data   = [sp_pf[sp_pf["model"] == m]["MAE"].values for m in top8]
        labels = [f"★ {m}" if m == stat_best else m for m in top8]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))

        colors = ["#2166ac" if m == stat_best else "#d1e5f0" for m in top8]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_title(SP_TAG[sp_name], fontsize=11, fontweight="bold")
        ax.set_ylabel("MAE", fontsize=9)
        ax.set_xticks(range(1, len(top8) + 1))
        ax.set_xticklabels(labels, fontsize=9, fontweight="bold", rotation=90, ha="center")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig6_mae_boxplots.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig6_mae_boxplots saved.")


# ---------------------------------------------------------------------------
# Figure 7 — Learning curve: stat-best model MAE vs training size
# ---------------------------------------------------------------------------
def fig7_learning_curve(pf: pd.DataFrame, agg: pd.DataFrame):
    # Approximate training days for each fold (train starts 01-02)
    TRAIN_DAYS = {
        "fold1_apr": 89,   # Jan 2 – Mar 31
        "fold2_jul": 180,  # Jan 2 – Jun 30
        "fold3_oct": 272,  # Jan 2 – Sep 30
        "fold4_dec": 333,  # Jan 2 – Nov 30
    }
    folds = ["fold1_apr", "fold2_jul", "fold3_oct", "fold4_dec"]
    x_days = [TRAIN_DAYS[f] for f in folds]

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.tab10

    for i, sp_name in enumerate(SP_ORDER):
        stat_best = (
            agg[agg["supply_point"] == sp_name]
            .sort_values("mean_net_wins", ascending=False)
            .iloc[0]["model"]
        )
        sp_pf = pf[(pf["supply_point"] == sp_name) & (pf["model"] == stat_best)]
        mae_per_fold = [
            sp_pf[sp_pf["fold"] == f]["MAE"].values[0]
            if not sp_pf[sp_pf["fold"] == f].empty else np.nan
            for f in folds
        ]
        label = f"{SP_TAG[sp_name]} — {stat_best}"
        ax.plot(x_days, mae_per_fold, marker="o", linewidth=2, markersize=7,
                color=cmap(i / len(SP_ORDER)), label=label)

    ax.set_xlabel("Training set size (days)", fontsize=11)
    ax.set_ylabel("MAE", fontsize=11)
    ax.set_xticks(x_days)
    ax.set_xticklabels([f"{d}d\n(Fold {i+1})" for i, d in enumerate(x_days)], fontsize=9)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(True, alpha=0.3)
    ax.set_title("Learning Curve: Stat-best Model MAE vs Training Size",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig7_learning_curve.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("  fig7_learning_curve saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading results...")
    pf, agg = load_data()

    print("\nGenerating static figures...")
    fig1_heatmap(pf)
    fig2_avg_ranking(agg)
    fig3_rmse_vs_wilcoxon(pf)
    fig4_bump_chart(pf)
    fig5_agreement_heatmap(pf)
    fig6_mae_boxplots(pf, agg)
    fig7_learning_curve(pf, agg)

    print("\nDone. Outputs in cross_validation/")
