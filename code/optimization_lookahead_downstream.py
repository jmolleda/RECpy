"""
Two optimization analyses:

PV look-ahead bias.
      The pipeline feeds the optimizer the *observed* (measured) PV of
      the test day -> perfect PV foresight. We quantify the bias with three
      information sets, all evaluated on REALIZED (observed) demand and PV:
        perfect    decisions on observed demand + observed PV  (upper bound)
        paper      decisions on forecast demand + observed PV  (current paper)
        realistic  decisions on forecast demand + forecast PV  (no look-ahead;
                   day-ahead PV forecast = previous-day persistence)
      perfect->paper gap = demand-forecast cost; paper->realistic = PV look-ahead.

Forecast-to-decision evaluation.
      Feed the RMSE-selected models vs the Wilcoxon-selected models into the LP
      and compare operational cost, connecting the selection criterion to the
      paper's actual objective.

All runs use the consistent "gross" degradation accounting at nominal cost.
Outputs (optimization_results/):
  pv_lookahead.csv        per-fold & total realized cost for the 3 info sets
  downstream_selection.csv  Wilcoxon vs RMSE selection -> operational cost
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pyomo.environ import SolverFactory

import opt_core as oc

OUT = Path(__file__).parent / "optimization_results"
DEG_MODEL, DEG_MULT = "gross", 1.0


def build_days(prices_df, solar_df, model_map):
    """Per-day records with forecast demand, observed demand, observed PV and a
    previous-day persistence PV forecast."""
    days = []
    for fold in oc.FOLDS:
        forecasts, observed = oc.load_fold_predictions(
            fold["dir"], fold["suffix"], model_map)
        drange = pd.date_range(start=fold["month"] + "-01",
                               end=pd.Period(fold["month"]).end_time.date(), freq="D")
        for day in drange:
            demand_day, ok = {}, True
            for sp, s in forecasts.items():
                v = s[s.index.date == day.date()].iloc[:24]
                if len(v) < 24:
                    ok = False; break
                demand_day[sp] = v.tolist()
            if not ok:
                continue
            obs_day = {sp: s[s.index.date == day.date()].iloc[:24].tolist()
                       for sp, s in observed.items()}
            solar = solar_df[solar_df.index.date == day.date()].iloc[:24]
            price = prices_df[prices_df.index.date == day.date()].iloc[:24]
            if len(solar) < 24 or len(price) < 24:
                continue
            prev = day - pd.Timedelta(days=1)
            psolar = solar_df[solar_df.index.date == prev.date()].iloc[:24]
            pv_obs = solar["PV_total"].round(3).tolist()
            pv_fc = (psolar["PV_total"].round(3).tolist()
                     if len(psolar) >= 24 else pv_obs)  # day-1 fallback
            obs_total = [sum(obs_day[sp][t] for sp in obs_day) for t in range(24)]
            days.append({
                "fold": fold["name"], "day": day,
                "demand_fc": demand_day, "obs_total": obs_total,
                "obs_demand_sp": obs_day,
                "pv_obs": pv_obs, "pv_fc": pv_fc,
                "battery": solar["Battery Power"].round(3).tolist(),
                "pvpc": price["PVPC"].tolist(), "comp": price["Compensation"].tolist(),
            })
    return days


# ---------------------------------------------------------------------------
# PV look-ahead
#   Two information sets, both realized on OBSERVED demand and PV:
#     perfect    decisions on observed demand + observed PV  -> upper bound
#     realistic  decisions on forecast demand + forecast PV  -> headline
#                (day-ahead PV forecast = previous-day persistence)
#   The perfect->realistic gap reflects combined demand- and PV-forecast error.
# ---------------------------------------------------------------------------
def pv_lookahead(days, solver):
    per_fold = {}
    soc = {"perfect": oc.INITIAL_SOC, "realistic": oc.INITIAL_SOC}
    prev_fold = None
    for d in days:
        if d["fold"] != prev_fold:
            soc = {k: oc.INITIAL_SOC for k in soc}
            prev_fold = d["fold"]
        rec = per_fold.setdefault(d["fold"], {"perfect": 0.0, "realistic": 0.0,
                                              "B2": 0.0})
        _, b2 = oc.compute_b1_b2(d["obs_total"], d["pv_obs"], d["battery"],
                                 d["pvpc"], d["comp"], DEG_MODEL, DEG_MULT)
        rec["B2"] += b2

        # perfect foresight: decide on observed demand + observed PV
        obs_demand = {sp: d["obs_demand_sp"][sp] for sp in d["obs_demand_sp"]}
        sched, _, soc["perfect"] = oc.optimize_day(
            obs_demand, d["pv_obs"], d["pvpc"], d["comp"], soc["perfect"],
            DEG_MODEL, DEG_MULT, solver)
        rec["perfect"] += oc.realize_cost(sched, d["obs_total"], d["pv_obs"],
                                          d["pvpc"], d["comp"], DEG_MODEL, DEG_MULT)

        # realistic: decide on forecast demand + forecast PV; realize on observed
        sched, _, soc["realistic"] = oc.optimize_day(
            d["demand_fc"], d["pv_fc"], d["pvpc"], d["comp"], soc["realistic"],
            DEG_MODEL, DEG_MULT, solver)
        rec["realistic"] += oc.realize_cost(sched, d["obs_total"], d["pv_obs"],
                                            d["pvpc"], d["comp"], DEG_MODEL, DEG_MULT)

    rows = []
    for fold, r in per_fold.items():
        rows.append({
            "fold": fold,
            "B2_eur": round(r["B2"], 2),
            "perfect_eur": round(r["perfect"], 2),
            "realistic_eur": round(r["realistic"], 2),
            "save_perfect": round(r["B2"] - r["perfect"], 2),
            "save_realistic": round(r["B2"] - r["realistic"], 2),
        })
    df = pd.DataFrame(rows)
    total = {"fold": "TOTAL"}
    for c in df.columns[1:]:
        total[c] = round(df[c].sum(), 2)
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Downstream cost: Wilcoxon vs RMSE selection
# ---------------------------------------------------------------------------
def downstream(prices_df, solar_df, solver):
    rmse_map = oc.rmse_best_model()
    print("\nRMSE-best model per SP:", rmse_map)
    print("Wilcoxon model per SP :", oc.WILCOXON_MODEL)

    results = {}
    for label, mmap in [("Wilcoxon", oc.WILCOXON_MODEL), ("RMSE", rmse_map)]:
        days = build_days(prices_df, solar_df, mmap)
        soc, prev_fold = oc.INITIAL_SOC, None
        per_fold = {}
        for d in days:
            if d["fold"] != prev_fold:
                soc = oc.INITIAL_SOC; prev_fold = d["fold"]
            rec = per_fold.setdefault(d["fold"], {"opt": 0.0, "B2": 0.0})
            _, b2 = oc.compute_b1_b2(d["obs_total"], d["pv_obs"], d["battery"],
                                     d["pvpc"], d["comp"], DEG_MODEL, DEG_MULT)
            # FF basis: decisions on forecast demand + forecast PV, realized on observed
            sched, _, soc = oc.optimize_day(
                d["demand_fc"], d["pv_fc"], d["pvpc"], d["comp"], soc,
                DEG_MODEL, DEG_MULT, solver)
            rec["opt"] += oc.realize_cost(sched, d["obs_total"], d["pv_obs"],
                                          d["pvpc"], d["comp"], DEG_MODEL, DEG_MULT)
            rec["B2"] += b2
        results[label] = per_fold

    rows = []
    folds = [f["name"] for f in oc.FOLDS]
    for fold in folds:
        w, r = results["Wilcoxon"][fold], results["RMSE"][fold]
        rows.append({
            "fold": fold,
            "B2_eur": round(w["B2"], 2),
            "Wilcoxon_OPT_eur": round(w["opt"], 2),
            "RMSE_OPT_eur": round(r["opt"], 2),
            "Wilcoxon_save_vs_B2": round(w["B2"] - w["opt"], 2),
            "RMSE_save_vs_B2": round(r["B2"] - r["opt"], 2),
            "Wilcoxon_minus_RMSE_cost": round(w["opt"] - r["opt"], 2),
        })
    df = pd.DataFrame(rows)
    total = {"fold": "TOTAL"}
    for c in df.columns[1:]:
        total[c] = round(df[c].sum(), 2)
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    return df, rmse_map


def main():
    prices_df, solar_df = oc.load_prices(), oc.load_solar()
    solver = SolverFactory("appsi_highs")

    print("=" * 60, "\nR2.3  PV look-ahead\n", "=" * 60)
    days = build_days(prices_df, solar_df, oc.WILCOXON_MODEL)
    pv = pv_lookahead(days, solver)
    pv.to_csv(OUT / "pv_lookahead.csv", index=False)
    print(pv.to_string(index=False))

    print("\n", "=" * 60, "\nR2.2  Downstream cost: Wilcoxon vs RMSE\n", "=" * 60)
    ds, rmse_map = downstream(prices_df, solar_df, solver)
    ds.to_csv(OUT / "downstream_selection.csv", index=False)
    print(ds.to_string(index=False))
    print(f"\nWritten pv_lookahead.csv and downstream_selection.csv to {OUT}")


if __name__ == "__main__":
    main()
