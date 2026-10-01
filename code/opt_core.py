"""
Reusable core for the optimization analyses.

Mirrors the LP in run_optimization_cv.py but exposes the knobs 
for degradation accounting, solver, and realized-cost evaluation:
  * `deg_model`  -- battery-degradation accounting:
        "gross"        charge + discharge/eff      (paper original; two-sided)
        "throughput"   0.5*(charge + discharge)    (classical cycle/throughput)
        "discharge"    discharge                   (delivered-energy throughput)
    The SAME accounting is applied to the optimizer and to the B2 baseline, so
    the comparison is fair (addresses the B2/OPT mismatch in R1.4/R2.4).
  * `deg_mult`   -- multiplier on the per-kWh degradation cost (sensitivity).
  * realized-cost evaluation: decisions taken on the *information set* (forecast
    demand / forecast PV) but cost realized on *observed* demand and PV
    (addresses the PV look-ahead bias).
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, Constraint,
    Objective, SolverFactory, minimize, value
)

from recpy_config import CONFIG

# All settings come from the active configuration (config/default.toml unless
# RECPY_CONFIG points elsewhere); see code/recpy_config.py.
YEAR = CONFIG.year

# ── System parameters ─────────────────────────────────────────────────
BATTERY_CAPACITY    = CONFIG.battery["capacity_kwh"]
BATTERY_MIN_SOC     = CONFIG.battery["min_soc_kwh"]
BATTERY_EFFICIENCY  = CONFIG.battery["efficiency"]
BATTERY_POWER_LIMIT = CONFIG.battery["power_limit_kw"]
BATTERY_DEGRADATION = CONFIG.battery["degradation_eur_per_kwh"]
GRID_EXPORT_LIMIT   = CONFIG.grid_export_limit
INITIAL_SOC         = CONFIG.initial_soc

CV_DIR = CONFIG.output_root / "cross_validation"
PRICES = CONFIG.data_file("prices")
SOLAR  = CONFIG.data_file("solar")

SP_KEYS = [sp["column"] for sp in CONFIG.supply_points]

WILCOXON_MODEL = CONFIG.deployed_models

FOLDS = [
    {"name": f["label"], "dir": f["name"], "suffix": f["name"], "month": f["month"]}
    for f in CONFIG.folds
]


# ── Degradation accounting (single source of truth for OPT and B2) ───────────
def deg_cost(charge, discharge, deg_model, deg_mult):
    """Per-hour degradation cost for non-negative charge/discharge energy."""
    base = BATTERY_DEGRADATION * deg_mult
    if deg_model == "gross":
        return (charge + discharge / BATTERY_EFFICIENCY) * base
    if deg_model == "throughput":
        return 0.5 * (charge + discharge) * base
    if deg_model == "discharge":
        return discharge * base
    raise ValueError(f"unknown deg_model {deg_model!r}")


def deg_expr(m, t, deg_model, deg_mult):
    """Pyomo expression version (linear in the decision variables)."""
    base = BATTERY_DEGRADATION * deg_mult
    if deg_model == "gross":
        return (m.batteryCharge[t] + m.batteryDischarge[t] / BATTERY_EFFICIENCY) * base
    if deg_model == "throughput":
        return 0.5 * (m.batteryCharge[t] + m.batteryDischarge[t]) * base
    if deg_model == "discharge":
        return m.batteryDischarge[t] * base
    raise ValueError(f"unknown deg_model {deg_model!r}")


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_prices():
    df = pd.read_csv(PRICES, index_col="datetime", parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def load_solar():
    df = pd.read_csv(SOLAR, index_col="datetime", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df["PV_total"] = df["PV to inverter Victron"] + df["PV to Inverter Huawei"]
    return df


def load_fold_predictions(fold_dir, suffix, model_map):
    """Return (forecasts, observed) dicts sp -> Series, clipped >= 0."""
    base = CV_DIR / fold_dir
    forecasts, observed = {}, {}
    for sp in SP_KEYS:
        df = pd.read_csv(base / f"Predictions_{sp}_{suffix}.csv",
                         index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        forecasts[sp] = df[model_map[sp]].clip(lower=0)
        observed[sp] = df["Observed"].clip(lower=0)
    return forecasts, observed


def rmse_best_model(model_candidates_path=None):
    """RMSE-best model per SP (lowest mean RMSE across folds) from the CSVs."""
    best = {}
    for sp in SP_KEYS:
        rmse_by_model = {}
        for fold in FOLDS:
            df = pd.read_csv(
                CV_DIR / fold["dir"] / f"Predictions_{sp}_{fold['suffix']}.csv",
                index_col=0, parse_dates=True)
            obs = df["Observed"].to_numpy()
            for col in df.columns:
                if col == "Observed":
                    continue
                err = df[col].to_numpy() - obs
                if not np.isfinite(err).all():
                    continue
                rmse = np.sqrt(np.mean(err ** 2))
                rmse_by_model.setdefault(col, []).append(rmse)
        means = {m: np.mean(v) for m, v in rmse_by_model.items() if len(v) == 4}
        best[sp] = min(means, key=means.get)
    return best


# ── Single-day LP ────────────────────────────────────────────────────────────
def optimize_day(demand_day, solar_vals, pvpc_vals, comp_vals, initial_soc,
                 deg_model="gross", deg_mult=1.0, solver=None):
    """Solve one day. demand_day: dict sp -> list[24]; solar_vals: list[24].

    Returns (schedule, opt_cost, final_soc) where schedule is a list of dicts
    with charge/discharge/import/export/SoC per hour. opt_cost uses the demand
    and PV passed in (so decisions and evaluation share this information set).
    """
    entities = list(demand_day.keys())
    T = range(24)
    m = ConcreteModel()
    m.T = RangeSet(0, 23)
    m.E = RangeSet(len(entities))
    m.demand = Param(m.E, m.T, initialize={
        (e + 1, t): float(demand_day[entities[e]][t])
        for e in range(len(entities)) for t in T})
    m.priceSell = Param(m.T, initialize=dict(enumerate(comp_vals)))
    m.priceBuy = Param(m.T, initialize=dict(enumerate(pvpc_vals)))

    m.batterySOC = Var(m.T, bounds=(BATTERY_MIN_SOC, BATTERY_CAPACITY))
    m.batteryCharge = Var(m.T, bounds=(0, BATTERY_POWER_LIMIT))
    m.batteryDischarge = Var(m.T, bounds=(0, BATTERY_POWER_LIMIT))
    m.gridImport = Var(m.T, bounds=(0, None))
    m.solarSurplus = Var(m.T, bounds=(0, GRID_EXPORT_LIMIT))

    def energy_balance(m, t):
        d = sum(m.demand[e, t] for e in m.E)
        return d == solar_vals[t] + m.gridImport[t] - m.solarSurplus[t] \
            + m.batteryDischarge[t] - m.batteryCharge[t]
    m.energyBalance = Constraint(m.T, rule=energy_balance)

    def battery_dynamics(m, t):
        prev = initial_soc if t == 0 else m.batterySOC[t - 1]
        return m.batterySOC[t] == prev \
            + m.batteryCharge[t] * BATTERY_EFFICIENCY \
            - m.batteryDischarge[t] / BATTERY_EFFICIENCY
    m.batteryDynamics = Constraint(m.T, rule=battery_dynamics)

    def solar_surplus_balance(m, t):
        d = sum(m.demand[e, t] for e in m.E)
        return m.solarSurplus[t] >= solar_vals[t] - d - m.batteryCharge[t]
    m.solarSurplusBalance = Constraint(m.T, rule=solar_surplus_balance)

    def grid_import_limit(m, t):
        d = sum(m.demand[e, t] for e in m.E)
        return m.gridImport[t] <= d
    m.gridImportLimit = Constraint(m.T, rule=grid_import_limit)

    m.obj = Objective(rule=lambda m: sum(
        m.gridImport[t] * m.priceBuy[t] - m.solarSurplus[t] * m.priceSell[t]
        + deg_expr(m, t, deg_model, deg_mult) for t in m.T), sense=minimize)

    if solver is None:
        solver = SolverFactory("appsi_highs")
    solver.solve(m)

    schedule = []
    for t in T:
        schedule.append({
            "charge": value(m.batteryCharge[t]),
            "discharge": value(m.batteryDischarge[t]),
            "soc": value(m.batterySOC[t]),
            "import": value(m.gridImport[t]),
            "export": value(m.solarSurplus[t]),
        })
    return schedule, value(m.obj), value(m.batterySOC[23])


# ── B2 baseline + realized-cost evaluator (shared accounting) ────────────────
def compute_b1_b2(obs_total, solar_vals, battery_vals, pvpc_vals, comp_vals,
                  deg_model="gross", deg_mult=1.0):
    """B1 (grid only) and B2 (PV + observed controller) for one day, using the
    same degradation accounting as the optimizer."""
    b1 = b2 = 0.0
    for t in range(24):
        d, gen, batt = obs_total[t], solar_vals[t], battery_vals[t]
        pb, ps = pvpc_vals[t], comp_vals[t]
        b1 += d * pb
        charge = max(0.0, batt)
        discharge = max(0.0, -batt)
        net_import = max(0.0, d - gen + batt)
        net_export = max(0.0, gen - d - charge)
        b2 += net_import * pb - net_export * ps \
            + deg_cost(charge, discharge, deg_model, deg_mult)
    return b1, b2


def realize_cost(schedule, obs_total, obs_solar, pvpc_vals, comp_vals,
                 deg_model="gross", deg_mult=1.0):
    """Realized cost of a battery schedule under OBSERVED demand and PV.

    The battery charge/discharge decisions are taken as scheduled; grid import
    and export re-balance to the realized net load. Used to measure PV
    look-ahead bias: schedule decided on forecast PV, cost on observed."""
    cost = 0.0
    for t in range(24):
        charge = schedule[t]["charge"]
        discharge = schedule[t]["discharge"]
        net = obs_total[t] - obs_solar[t] - discharge + charge
        grid_import = max(0.0, net)
        export = min(max(0.0, -net), GRID_EXPORT_LIMIT)
        cost += grid_import * pvpc_vals[t] - export * comp_vals[t] \
            + deg_cost(charge, discharge, deg_model, deg_mult)
    return cost
