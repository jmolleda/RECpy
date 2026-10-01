"""
Runs the day-ahead energy optimization aligned with the 4-fold
expanding-window cross-validation used for forecasting.

Three cost scenarios are computed for every day:
  B1  Grid-only        No PV, no battery.  Full demand bought at PVPC.
  B2  Hardware, no opt PV self-consumption + actual battery controller.
                       Observed demand; Battery Power from data logger.
  OPT Optimised        LP optimizer with out-of-sample CV forecasts.

Battery SoC is initialised at 50 % capacity (25.08 kWh) at the start
of every fold and carried forward day-to-day within the fold.

Outputs (in optimization_results/):
  optimization_cv_hourly.csv  – hour-by-hour LP decisions, all folds
  optimization_cv_daily.csv   – daily B1 / B2 / OPT costs per fold
  optimization_cv_summary.csv – per-fold and overall summary table
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, Constraint,
    Objective, SolverFactory, minimize, value
)

from recpy_config import CONFIG

# Calendar year of the dataset (placeholder;
# set this to the actual year of your data files).
YEAR = CONFIG.year

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
CV_DIR  = CONFIG.output_root / "cross_validation"
PRICES  = CONFIG.data_file("prices")
SOLAR   = CONFIG.data_file("solar")
OUT_DIR = CONFIG.output_dir("optimization_results")
OUT_DIR.mkdir(exist_ok=True)

# ── Selected model per supply point (Wilcoxon tournament) ─────────────────────
SELECTED_MODEL = CONFIG.deployed_models
SP_KEYS = [sp["column"] for sp in CONFIG.supply_points]

# ── Fold definitions ──────────────────────────────────────────────────────────
FOLDS = [
    {"name": f["label"], "dir": f["name"], "suffix": f["name"], "month": f["month"]}
    for f in CONFIG.folds
]

# ── Battery / system parameters ───────────────────────────────────────────────
BATTERY_CAPACITY    = CONFIG.battery["capacity_kwh"]
BATTERY_MIN_SOC     = CONFIG.battery["min_soc_kwh"]
BATTERY_EFFICIENCY  = CONFIG.battery["efficiency"]
BATTERY_POWER_LIMIT = CONFIG.battery["power_limit_kw"]
BATTERY_DEGRADATION = CONFIG.battery["degradation_eur_per_kwh"]
GRID_EXPORT_LIMIT   = CONFIG.grid_export_limit
INITIAL_SOC         = CONFIG.initial_soc


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_fold_predictions(fold_dir, suffix):
    """Return (forecasts, observed) — both dicts sp -> Series, clipped >= 0."""
    base = CV_DIR / fold_dir
    forecasts = {}
    observed  = {}
    for sp in SP_KEYS:
        df = pd.read_csv(base / f"Predictions_{sp}_{suffix}.csv",
                         index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        forecasts[sp] = df[SELECTED_MODEL[sp]].clip(lower=0)
        observed[sp]  = df["Observed"].clip(lower=0)
    return forecasts, observed


def load_prices():
    df = pd.read_csv(PRICES, index_col="datetime", parse_dates=True)
    df.index = pd.to_datetime(df.index)   # naive local datetime — date matches local calendar
    return df


def load_solar():
    df = pd.read_csv(SOLAR, index_col="datetime", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df["PV_total"] = df["PV to inverter Victron"] + df["PV to Inverter Huawei"]
    return df  # retains Battery Power column for B2 baseline


def compute_b2_day(obs_demand, solar_vals, battery_vals, pvpc_vals, comp_vals):
    """B2 baseline: PV + actual battery controller, no optimisation.

    Battery Power > 0 means net charging (consuming power from PV/grid).
    Battery Power < 0 means net discharging (supplying power to loads).
    """
    b1_cost = 0.0
    b2_cost = 0.0
    for t in range(24):
        d    = obs_demand[t]
        gen  = solar_vals[t]
        batt = battery_vals[t]   # signed: + charging, - discharging
        pb   = pvpc_vals[t]
        ps   = comp_vals[t]

        # B1: no hardware at all
        b1_cost += d * pb

        # B2: PV + actual battery behaviour
        net_import  = max(0.0,  d - gen + batt)
        net_export  = max(0.0,  gen - d - max(0.0, batt))
        degradation = abs(batt) * BATTERY_DEGRADATION
        b2_cost += net_import * pb - net_export * ps + degradation

    return b1_cost, b2_cost


def slice_day(df, date):
    """Return 24-hour slice for a calendar date, handling DST (take first 24 h)."""
    mask = df.index.date == pd.Timestamp(date).date()
    day = df[mask]
    return day.iloc[:24]   # DST days may have 25 h; keep first 24


# ── Single-day LP ─────────────────────────────────────────────────────────────
def optimize_day(demand_day, solar_vals, pvpc_vals, comp_vals, initial_soc):
    entities = list(demand_day.keys())
    T = range(24)

    model = ConcreteModel()
    model.T = RangeSet(0, 23)
    model.E = RangeSet(len(entities))

    model.demand = Param(
        model.E, model.T,
        initialize={(e + 1, t): float(demand_day[entities[e]][t])
                    for e in range(len(entities)) for t in T}
    )
    model.priceSell = Param(model.T, initialize=dict(enumerate(comp_vals)))
    model.priceBuy  = Param(model.T, initialize=dict(enumerate(pvpc_vals)))

    model.batterySOC       = Var(model.T, bounds=(BATTERY_MIN_SOC, BATTERY_CAPACITY))
    model.batteryCharge    = Var(model.T, bounds=(0, BATTERY_POWER_LIMIT))
    model.batteryDischarge = Var(model.T, bounds=(0, BATTERY_POWER_LIMIT))
    model.gridImport       = Var(model.T, bounds=(0, None))
    model.solarSurplus     = Var(model.T, bounds=(0, GRID_EXPORT_LIMIT))

    def energy_balance(m, t):
        d = sum(m.demand[e, t] for e in m.E)
        return d == solar_vals[t] + m.gridImport[t] - m.solarSurplus[t] \
                    + m.batteryDischarge[t] - m.batteryCharge[t]
    model.energyBalance = Constraint(model.T, rule=energy_balance)

    def battery_dynamics(m, t):
        if t == 0:
            return m.batterySOC[t] == initial_soc \
                   + m.batteryCharge[t] * BATTERY_EFFICIENCY \
                   - m.batteryDischarge[t] / BATTERY_EFFICIENCY
        return m.batterySOC[t] == m.batterySOC[t - 1] \
               + m.batteryCharge[t] * BATTERY_EFFICIENCY \
               - m.batteryDischarge[t] / BATTERY_EFFICIENCY
    model.batteryDynamics = Constraint(model.T, rule=battery_dynamics)

    def solar_surplus_balance(m, t):
        d = sum(m.demand[e, t] for e in m.E)
        return m.solarSurplus[t] >= solar_vals[t] - d - m.batteryCharge[t]
    model.solarSurplusBalance = Constraint(model.T, rule=solar_surplus_balance)

    def grid_import_limit(m, t):
        d = sum(m.demand[e, t] for e in m.E)
        return m.gridImport[t] <= d
    model.gridImportLimit = Constraint(model.T, rule=grid_import_limit)

    def objective_rule(m):
        return sum(
            m.gridImport[t] * m.priceBuy[t]
            - m.solarSurplus[t] * m.priceSell[t]
            + (m.batteryCharge[t] + m.batteryDischarge[t] / BATTERY_EFFICIENCY)
            * BATTERY_DEGRADATION
            for t in m.T
        )
    model.obj = Objective(rule=objective_rule, sense=minimize)

    solver = SolverFactory("appsi_highs")
    solver.solve(model)

    rows = []
    for t in T:
        d   = sum(value(model.demand[e, t]) for e in range(1, len(entities) + 1))
        bc  = value(model.batteryCharge[t])
        bd  = value(model.batteryDischarge[t])
        gi  = value(model.gridImport[t])
        ss  = value(model.solarSurplus[t])
        soc = value(model.batterySOC[t])
        pb  = pvpc_vals[t]
        ps  = comp_vals[t]
        rows.append({
            "Hour":                     t,
            "Demand [kWh]":             round(d, 3),
            "Solar [kWh]":              round(solar_vals[t], 3),
            "Export price [€/kWh]":     round(ps, 4),
            "Import price [€/kWh]":     round(pb, 4),
            "Battery charge [kWh]":     round(bc, 3),
            "Battery discharge [kWh]":  round(bd, 3),
            "Battery SoC [kWh]":        round(soc, 3),
            "Grid import [kWh]":        round(gi, 3),
            "Solar export [kWh]":       round(ss, 3),
            "Sell [€]":                 round(ss * ps, 4),
            "Buy [€]":                  round(gi * pb, 4),
            "Total cost [€]":           round(
                gi * pb - ss * ps
                + (bc + bd / BATTERY_EFFICIENCY) * BATTERY_DEGRADATION, 4),
        })

    final_soc = value(model.batterySOC[23])
    opt_cost  = value(model.obj)
    grid_cost = sum(
        sum(value(model.demand[e, t]) for e in range(1, len(entities) + 1))
        * pvpc_vals[t]
        for t in T
    )
    return rows, opt_cost, grid_cost, final_soc


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    prices_df = load_prices()
    solar_df  = load_solar()

    all_hourly   = []
    all_daily    = []
    fold_summary = []

    for fold in FOLDS:
        print(f"\n{'='*60}")
        print(f"  {fold['name']}")
        print(f"{'='*60}")

        predictions, observed = load_fold_predictions(fold["dir"], fold["suffix"])

        # Days actually covered by the forecasts, so a shortened test window
        # (config/quick.toml) schedules only the days it produced.
        test_days = pd.DatetimeIndex(
            sorted({t.normalize() for t in next(iter(predictions.values())).index})
        )

        soc            = INITIAL_SOC
        fold_b1_total  = 0.0
        fold_b2_total  = 0.0
        fold_opt_total = 0.0

        for day in test_days:
            day_str = day.strftime("%Y-%m-%d")

            # Forecasted demand (for optimizer)
            demand_day = {}
            ok = True
            for sp, series in predictions.items():
                mask = series.index.date == day.date()
                s = series[mask].iloc[:24]
                if len(s) < 24:
                    print(f"  SKIP {day_str}: {sp} has only {len(s)} hours")
                    ok = False
                    break
                demand_day[sp] = s.tolist()
            if not ok:
                continue

            # Observed demand (for B1 / B2 baselines)
            obs_day = {}
            for sp, series in observed.items():
                mask = series.index.date == day.date()
                obs_day[sp] = series[mask].iloc[:24].tolist()

            # Solar + battery (UTC-indexed)
            solar_slice = solar_df[solar_df.index.date == day.date()].iloc[:24]
            if len(solar_slice) < 24:
                print(f"  SKIP {day_str}: solar has only {len(solar_slice)} hours")
                continue
            solar_vals   = solar_slice["PV_total"].round(3).tolist()
            battery_vals = solar_slice["Battery Power"].round(3).tolist()

            # Prices (naive local datetime index)
            price_slice = prices_df[prices_df.index.date == day.date()].iloc[:24]
            if len(price_slice) < 24:
                print(f"  SKIP {day_str}: prices have only {len(price_slice)} hours")
                continue
            pvpc_vals = price_slice["PVPC"].tolist()
            comp_vals = price_slice["Compensation"].tolist()

            # B1 and B2 baselines (observed demand)
            obs_total = [sum(obs_day[sp][t] for sp in obs_day) for t in range(24)]
            b1_cost, b2_cost = compute_b2_day(
                obs_total, solar_vals, battery_vals, pvpc_vals, comp_vals
            )

            # Optimised (forecasted demand)
            rows, opt_cost, _, soc = optimize_day(
                demand_day, solar_vals, pvpc_vals, comp_vals, soc
            )

            fold_b1_total  += b1_cost
            fold_b2_total  += b2_cost
            fold_opt_total += opt_cost

            for r in rows:
                r["Fold"] = fold["name"]
                r["Day"]  = day_str
            all_hourly.extend(rows)

            all_daily.append({
                "Fold":               fold["name"],
                "Day":                day_str,
                "B1: Grid-only [€]":  round(b1_cost, 3),
                "B2: HW no opt [€]":  round(b2_cost, 3),
                "Optimised [€]":      round(opt_cost, 3),
                "Saving vs B1 [€]":   round(b1_cost - opt_cost, 3),
                "Saving vs B2 [€]":   round(b2_cost - opt_cost, 3),
            })
            print(f"  {day_str}  B1={b1_cost:7.3f}  B2={b2_cost:7.3f}  opt={opt_cost:7.3f}  sav={b2_cost-opt_cost:6.3f}  SoC={soc:.1f}")

        fold_sav_b2 = fold_b2_total - fold_opt_total
        fold_summary.append({
            "Fold":               fold["name"],
            "Days":               len([d for d in all_daily if d["Fold"] == fold["name"]]),
            "B1: Grid-only [€]":  round(fold_b1_total, 2),
            "B2: HW no opt [€]":  round(fold_b2_total, 2),
            "Optimised [€]":      round(fold_opt_total, 2),
            "Saving vs B1 [€]":   round(fold_b1_total - fold_opt_total, 2),
            "Saving vs B2 [€]":   round(fold_sav_b2, 2),
        })
        print(f"\n  {fold['name']} TOTAL:  B1={fold_b1_total:.2f}  B2={fold_b2_total:.2f}  opt={fold_opt_total:.2f}  sav={fold_sav_b2:.2f}")

    # Overall totals
    df_daily   = pd.DataFrame(all_daily)
    df_hourly  = pd.DataFrame(all_hourly)
    df_summary = pd.DataFrame(fold_summary)

    totals_row = pd.DataFrame([{
        "Fold":               "TOTAL",
        "Days":               df_summary["Days"].sum(),
        "B1: Grid-only [€]":  round(df_summary["B1: Grid-only [€]"].sum(), 2),
        "B2: HW no opt [€]":  round(df_summary["B2: HW no opt [€]"].sum(), 2),
        "Optimised [€]":      round(df_summary["Optimised [€]"].sum(), 2),
        "Saving vs B1 [€]":   round(df_summary["Saving vs B1 [€]"].sum(), 2),
        "Saving vs B2 [€]":   round(df_summary["Saving vs B2 [€]"].sum(), 2),
    }])
    df_summary = pd.concat([df_summary, totals_row], ignore_index=True)

    # Save outputs
    df_hourly.to_csv(OUT_DIR  / "optimization_cv_hourly.csv",  index=False)
    df_daily.to_csv(OUT_DIR   / "optimization_cv_daily.csv",   index=False)
    df_summary.to_csv(OUT_DIR / "optimization_cv_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  OVERALL ({int(df_summary.loc[df_summary['Fold']=='TOTAL','Days'].iloc[0])} days)")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
