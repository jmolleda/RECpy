"""
Battery-degradation analysis.

1. B2 and the optimizer use the SAME degradation accounting.
2. Sensitivity sweep over the degradation model and the per-kWh cost multiplier.
3. Validation: under the original accounting the totals reproduce the paper
4. Solver metadata: solver, termination, mean per-day runtime.

Outputs (optimization_results/):
  sensitivity_degradation.csv   deg_model x deg_mult -> totals & savings
  degradation_validation.csv    reproduction check vs the published numbers
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from pyomo.environ import SolverFactory, value

import opt_core as oc

from recpy_config import CONFIG

OUT = CONFIG.output_dir("optimization_results")
DEG_MODELS = ["gross", "throughput", "discharge"]
DEG_MULTS = [0.0, 0.5, 1.0, 1.5, 2.0]


def iter_days(prices_df, solar_df, model_map):
    """Yield per-day arrays needed by every configuration (load once)."""
    for fold in oc.FOLDS:
        forecasts, observed = oc.load_fold_predictions(
            fold["dir"], fold["suffix"], model_map)
        days = pd.date_range(start=fold["month"] + "-01",
                             end=pd.Period(fold["month"]).end_time.date(), freq="D")
        for day in days:
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
            obs_total = [sum(obs_day[sp][t] for sp in obs_day) for t in range(24)]
            prev = day - pd.Timedelta(days=1)
            psolar = solar_df[solar_df.index.date == prev.date()].iloc[:24]
            pv_obs = solar["PV_total"].round(3).tolist()
            pv_fc = (psolar["PV_total"].round(3).tolist()
                     if len(psolar) >= 24 else pv_obs)   # day-1 persistence fallback
            yield {
                "fold": fold["name"], "day": day,
                "demand_day": demand_day,
                "obs_total": obs_total,
                "pv_obs": pv_obs,
                "pv_fc": pv_fc,
                "battery": solar["Battery Power"].round(3).tolist(),
                "pvpc": price["PVPC"].tolist(),
                "comp": price["Compensation"].tolist(),
            }


def run_config(days, deg_model, deg_mult, solver, timings=None, realized=False):
    """Total B1/B2/OPT over all days for one (deg_model, deg_mult).

    realized=False : OPT = LP objective (forecast demand + observed PV) — used to
                     reproduce the published numbers (validation only).
    realized=True  : FF basis — decisions on forecast demand + forecast PV, cost
                     realized on observed demand + observed PV.
    """
    b1 = b2 = opt = 0.0
    soc = oc.INITIAL_SOC
    prev_fold = None
    for d in days:
        if d["fold"] != prev_fold:          # reset SoC at each fold start
            soc = oc.INITIAL_SOC
            prev_fold = d["fold"]
        c1, c2 = oc.compute_b1_b2(d["obs_total"], d["pv_obs"], d["battery"],
                                  d["pvpc"], d["comp"], deg_model, deg_mult)
        t0 = time.perf_counter()
        if realized:
            sched, _, soc = oc.optimize_day(
                d["demand_day"], d["pv_fc"], d["pvpc"], d["comp"], soc,
                deg_model=deg_model, deg_mult=deg_mult, solver=solver)
            cost = oc.realize_cost(sched, d["obs_total"], d["pv_obs"],
                                   d["pvpc"], d["comp"], deg_model, deg_mult)
        else:
            sched, cost, soc = oc.optimize_day(
                d["demand_day"], d["pv_obs"], d["pvpc"], d["comp"], soc,
                deg_model=deg_model, deg_mult=deg_mult, solver=solver)
        if timings is not None:
            timings.append(time.perf_counter() - t0)
        b1 += c1; b2 += c2; opt += cost
    return b1, b2, opt


def original_b2(days, deg_mult=1.0):
    """B2 under the paper's original abs(batt) accounting (for validation)."""
    b2 = 0.0
    for d in days:
        for t in range(24):
            dem, gen, batt = d["obs_total"][t], d["pv_obs"][t], d["battery"][t]
            net_import = max(0.0, dem - gen + batt)
            net_export = max(0.0, gen - dem - max(0.0, batt))
            b2 += net_import * d["pvpc"][t] - net_export * d["comp"][t] \
                + abs(batt) * oc.BATTERY_DEGRADATION * deg_mult
    return b2


def main():
    prices_df, solar_df = oc.load_prices(), oc.load_solar()
    days = list(iter_days(prices_df, solar_df, oc.WILCOXON_MODEL))
    print(f"Loaded {len(days)} evaluation days.")
    solver = SolverFactory("appsi_highs")

    # --- Validation (LP-objective basis): reproduce the published OPT total ---
    timings = []
    b1, b2_gross, opt_gross = run_config(days, "gross", 1.0, solver, timings,
                                         realized=False)
    b2_orig = original_b2(days)
    val = pd.DataFrame([
        {"quantity": "B1 grid-only", "value_eur": round(b1, 2),
         "paper": 2577.07},
        {"quantity": "OPT (gross, mult=1, LP-objective)", "value_eur": round(opt_gross, 2),
         "paper": -587.98},
        {"quantity": "B2 original abs() accounting", "value_eur": round(b2_orig, 2),
         "paper": -185.68},
        {"quantity": "B2 consistent gross accounting", "value_eur": round(b2_gross, 2),
         "paper": np.nan},
    ])
    val.to_csv(OUT / "degradation_validation.csv", index=False)
    print("\n=== Validation vs published numbers (LP-objective basis) ===")
    print(val.to_string(index=False))
    print(f"\nSolver: appsi_highs (HiGHS {_highs_version()}); "
          f"days={len(timings)}; mean per-day solve={np.mean(timings)*1000:.1f} ms; "
          f"max={np.max(timings)*1000:.1f} ms")

    # --- Sensitivity sweep (FF basis: realized, forecast demand + forecast PV) -
    print("\n=== Degradation sensitivity (FF basis, realized) ===")
    rows = []
    for dm in DEG_MODELS:
        for mult in DEG_MULTS:
            b1c, b2c, optc = run_config(days, dm, mult, solver, realized=True)
            rows.append({
                "deg_model": dm, "deg_mult": mult,
                "B1_eur": round(b1c, 2), "B2_eur": round(b2c, 2),
                "OPT_eur": round(optc, 2),
                "saving_vs_B2_eur": round(b2c - optc, 2),
                "saving_vs_B1_eur": round(b1c - optc, 2),
            })
            print(f"  {dm:<11} mult={mult:<4}  OPT={optc:8.2f}  "
                  f"B2={b2c:8.2f}  saveB2={b2c-optc:7.2f}")
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "sensitivity_degradation.csv", index=False)
    print(f"\nWritten sensitivity_degradation.csv ({len(sens)} configs, FF basis).")


def _highs_version():
    try:
        import highspy
        return highspy.Highs().version()
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
