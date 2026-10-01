"""
Robustness analysis of the Wilcoxon model-selection tournament.

  * Temporal dependence of hourly errors  -> daily-aggregated absolute errors
    (one daily-MAE value per calendar day) in addition to the original hourly test.
  * Multiple comparisons across the pairwise tournament -> Holm (family-wise) and
    Benjamini-Hochberg (FDR) correction applied within each fold x supply-point
    family of C(24,2)=276 pairwise comparisons.
  * Serial-dependence-aware forecast comparison -> Diebold-Mariano test with a
    Newey-West HAC variance (lag h-1, h=24) and the Harvey-Leybourne-Newbold
    small-sample correction, for the deployed model vs every opponent.

The original pipeline (wilcoxon_cv.py) is left untouched; this script only reads
the same prediction CSVs and writes to cross_validation/robustness/.

Outputs (cross_validation/robustness/):
  net_wins_by_scheme.csv      per fold x SP x model net wins under every scheme
  selection_robustness.csv    selected model per SP per scheme + agreement flag
  dm_deployed_vs_opponents.csv DM dominance counts of the deployed model
  ml_vs_dl.csv                does ML still dominate all DL under each scheme
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from pathlib import Path

from recpy_config import CONFIG

BASE_CV = CONFIG.output_root / "cross_validation"
OUT_DIR = BASE_CV / "robustness"
OUT_DIR.mkdir(exist_ok=True)
ALPHA = 0.05
DM_H = 24  # day-ahead forecast horizon (hours) for the HAC lag

FOLDS = [f["name"] for f in CONFIG.folds]
SP_IDS = ["SP1", "SP2", "SP3", "SP4", "SP5", "SP6"]
SP_NAMES = {"SP1": "SP1", "SP2": "SP2", "SP3": "SP3",
            "SP4": "SP4", "SP5": "SP5", "SP6": "SP6"}

# Model the paper deploys per supply point (Wilcoxon tournament, hourly).
DEPLOYED = {
    "SP1":    "Support Vector",
    "SP2":    "k-Nearest Neighbors",
    "SP3":    "Huber",
    "SP4":    "Huber",
    "SP5":    "Huber",
    "SP6":    "Extra Trees",
}

# Deep-learning columns (used for the ML-vs-DL dominance check).
DL_MODELS = ["LSTM_single", "GRU_single", "TCN_single",
             "LSTM_daily", "GRU_daily", "TCN_daily"]


# ---------------------------------------------------------------------------
# Multiple-comparison corrections (manual, no statsmodels dependency)
# ---------------------------------------------------------------------------
def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg (FDR) adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        running = min(running, val)
        adj[idx] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------------------
# Diebold-Mariano with HAC variance + HLN small-sample correction
# ---------------------------------------------------------------------------
def dm_pvalue(loss_a, loss_b, h=DM_H):
    """One-sided DM test: SP1 = model A has lower expected loss than B.

    Returns the p-value for 'A better than B' (small p => A dominates)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 3:
        return np.nan
    dbar = d.mean()
    dev = d - dbar
    gamma0 = np.mean(dev ** 2)
    s = gamma0
    for k in range(1, min(h, T)):
        s += 2.0 * np.mean(dev[k:] * dev[:-k])
    var_dbar = s / T
    if var_dbar <= 0:
        return np.nan
    dm = dbar / np.sqrt(var_dbar)
    corr = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    dm_hln = dm * corr
    # d<0 means A has lower loss -> better; one-sided lower tail.
    return float(stats.t.cdf(dm_hln, df=T - 1))


# ---------------------------------------------------------------------------
# Per fold x SP analysis
# ---------------------------------------------------------------------------
def load_ae(fold, sp_id):
    """Return (hourly_ae_df, daily_ae_df) for finite models only."""
    path = BASE_CV / fold / f"Predictions_C_{sp_id}_{fold}.csv"
    if not path.exists():
        print(f"  MISSING {path.name}")
        return None, None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    obs = df["Observed"]
    preds = df.drop(columns="Observed")
    ae = preds.apply(lambda col: (col - obs).abs())
    # Drop models with any non-finite error (e.g. SGD divergence on SP1).
    finite = ae.columns[np.isfinite(ae.to_numpy()).all(axis=0)]
    ae = ae[finite]
    daily = ae.groupby(ae.index.normalize()).mean()
    return ae, daily


def net_wins(ae, two_sided, correction):
    """Net wins per model.

    two_sided=False replicates the original one-sided Wilcoxon (uncorrected).
    two_sided=True collects 276 pairwise p-values, applies `correction`
    ('holm' | 'bh' | None), then assigns the win to the lower-median model.
    """
    models = list(ae.columns)
    wins = {m: 0 for m in models}
    losses = {m: 0 for m in models}

    if not two_sided:  # original protocol: ordered one-sided tests
        for a, b in combinations(models, 2):
            diff = ae[a].to_numpy() - ae[b].to_numpy()
            if np.allclose(diff, 0):
                continue
            _, p_less = stats.wilcoxon(diff, alternative="less")
            _, p_grt = stats.wilcoxon(diff, alternative="greater")
            if p_less < ALPHA:
                wins[a] += 1; losses[b] += 1
            if p_grt < ALPHA:
                wins[b] += 1; losses[a] += 1
        return _to_series(models, wins, losses)

    pairs, pvals = [], []
    for a, b in combinations(models, 2):
        diff = ae[a].to_numpy() - ae[b].to_numpy()
        if np.allclose(diff, 0):
            continue
        _, p = stats.wilcoxon(diff, alternative="two-sided")
        pairs.append((a, b)); pvals.append(p)
    if not pairs:
        return _to_series(models, wins, losses)

    if correction == "holm":
        padj = holm(pvals)
    elif correction == "bh":
        padj = benjamini_hochberg(pvals)
    else:
        padj = np.asarray(pvals)

    med = ae.median()
    for (a, b), p in zip(pairs, padj):
        if p >= ALPHA or med[a] == med[b]:
            continue
        if med[a] < med[b]:
            wins[a] += 1; losses[b] += 1
        else:
            wins[b] += 1; losses[a] += 1
    return _to_series(models, wins, losses)


def _to_series(models, wins, losses):
    return pd.DataFrame({"wins": pd.Series(wins), "losses": pd.Series(losses)}) \
        .assign(net_wins=lambda d: d["wins"] - d["losses"])


SCHEMES = {
    "hourly_1sided_uncorr": dict(level="hourly", two_sided=False, correction=None),
    "hourly_holm":          dict(level="hourly", two_sided=True,  correction="holm"),
    "hourly_bh":            dict(level="hourly", two_sided=True,  correction="bh"),
    "daily_uncorr":         dict(level="daily",  two_sided=True,  correction=None),
    "daily_holm":           dict(level="daily",  two_sided=True,  correction="holm"),
    "daily_bh":             dict(level="daily",  two_sided=True,  correction="bh"),
}


def main():
    nw_rows = []
    dm_rows = []

    for fold in FOLDS:
        print(f"\n{'='*60}  {fold.upper()}")
        for sp_id in SP_IDS:
            ae_h, ae_d = load_ae(fold, sp_id)
            if ae_h is None:
                continue
            sp = SP_NAMES[sp_id]

            for scheme, cfg in SCHEMES.items():
                ae = ae_h if cfg["level"] == "hourly" else ae_d
                res = net_wins(ae, cfg["two_sided"], cfg["correction"])
                for model, r in res.iterrows():
                    nw_rows.append({
                        "fold": fold, "supply_point": sp, "scheme": scheme,
                        "model": model, "wins": int(r["wins"]),
                        "losses": int(r["losses"]), "net_wins": int(r["net_wins"]),
                    })
            best = net_wins(ae_h, True, "holm")["net_wins"].idxmax()
            print(f"  {sp:<11} hourly+Holm best: {best}")

            # --- DM robustness: deployed model vs every opponent (hourly) ---
            dep = DEPLOYED[sp]
            if dep not in ae_h.columns:
                continue
            opp_p = {}
            for opp in ae_h.columns:
                if opp == dep:
                    continue
                opp_p[opp] = dm_pvalue(ae_h[dep].to_numpy(), ae_h[opp].to_numpy())
            opp_names = list(opp_p)
            padj = benjamini_hochberg([opp_p[o] for o in opp_names])
            sig = sum(1 for p in padj if p < ALPHA)
            dm_rows.append({
                "fold": fold, "supply_point": sp, "deployed": dep,
                "n_opponents": len(opp_names),
                "dm_sig_wins_bh": int(sig),
            })

    nw = pd.DataFrame(nw_rows)
    nw.to_csv(OUT_DIR / "net_wins_by_scheme.csv", index=False)

    # --- Selection per SP per scheme (max mean net wins across folds) ---
    sel_rows = []
    agg = (nw.groupby(["supply_point", "scheme", "model"])["net_wins"]
             .mean().reset_index())
    for sp in [SP_NAMES[s] for s in SP_IDS]:
        for scheme in SCHEMES:
            sub = agg[(agg.supply_point == sp) & (agg.scheme == scheme)]
            if sub.empty:
                continue
            best = sub.loc[sub["net_wins"].idxmax()]
            sel_rows.append({
                "supply_point": sp, "scheme": scheme,
                "selected_model": best["model"],
                "mean_net_wins": round(best["net_wins"], 2),
                "matches_deployed": best["model"] == DEPLOYED[sp],
            })
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(OUT_DIR / "selection_robustness.csv", index=False)

    # --- ML vs DL: does the deployed (ML) model outrank every DL variant? ---
    mldl_rows = []
    for sp in [SP_NAMES[s] for s in SP_IDS]:
        for scheme in SCHEMES:
            sub = agg[(agg.supply_point == sp) & (agg.scheme == scheme)]
            if sub.empty:
                continue
            dep_nw = sub.loc[sub.model == DEPLOYED[sp], "net_wins"]
            dl_nw = sub[sub.model.isin(DL_MODELS)]["net_wins"]
            if dep_nw.empty or dl_nw.empty:
                continue
            mldl_rows.append({
                "supply_point": sp, "scheme": scheme,
                "deployed_net_wins": round(dep_nw.iloc[0], 2),
                "best_dl_net_wins": round(dl_nw.max(), 2),
                "deployed_beats_all_dl": dep_nw.iloc[0] > dl_nw.max(),
            })
    mldl = pd.DataFrame(mldl_rows)
    mldl.to_csv(OUT_DIR / "ml_vs_dl.csv", index=False)

    dm = pd.DataFrame(dm_rows)
    dm.to_csv(OUT_DIR / "dm_deployed_vs_opponents.csv", index=False)

    # ----------------------------- summary -----------------------------------
    print(f"\n{'='*70}\n  SELECTION ROBUSTNESS (selected model per scheme)\n{'='*70}")
    pivot = sel.pivot(index="supply_point", columns="scheme",
                      values="selected_model")
    pivot = pivot[list(SCHEMES)]
    print(pivot.to_string())

    print(f"\n{'='*70}\n  AGREEMENT WITH DEPLOYED MODEL (share of schemes)\n{'='*70}")
    share = sel.groupby("supply_point")["matches_deployed"].mean().round(2)
    print(share.to_string())

    print(f"\n{'='*70}\n  ML DEPLOYED MODEL BEATS ALL DL VARIANTS?\n{'='*70}")
    mldl_share = mldl.groupby("scheme")["deployed_beats_all_dl"].mean().round(2)
    print(mldl_share.to_string())

    print(f"\n{'='*70}\n  DM (HAC+HLN) dominance of deployed model "
          f"[mean sig. wins / 23 opponents, BH-corrected]\n{'='*70}")
    dm_share = dm.groupby("supply_point")["dm_sig_wins_bh"].mean().round(1)
    print(dm_share.to_string())
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()
