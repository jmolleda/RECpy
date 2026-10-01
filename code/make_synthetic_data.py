"""
Generate a fully SYNTHETIC example dataset so the pipeline can be run end-to-end
without the (withheld) real data. No real consumption is used or reproduced here;
all series are drawn from simple seasonal models plus noise.

Writes, under ``../data/`` (relative to this script):
  consumption_SP1.csv ... consumption_SP6.csv   date_time, C_SPk
  consumption_meteo_calendar.csv                date_time, C_SP1..C_SP6,
                                                calendar + weather features
  prices_cv_folds.csv                           date_time, PVPC, Compensation
  solar_generation.csv                          date_time, PV to inverter Victron,
                                                PV to Inverter Huawei, Battery Power

The calendar year is controlled by YEAR and MUST match the YEAR constant used by
the analysis scripts.
"""

import numpy as np
import pandas as pd
from pathlib import Path

YEAR = 2025                 # must match YEAR in the analysis scripts
RNG = np.random.default_rng(123)
OUT = Path(__file__).parent.parent / "data"
OUT.mkdir(exist_ok=True)

idx = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31 23:00", freq="h")
n = len(idx)
hour = idx.hour.to_numpy()
doy = idx.dayofyear.to_numpy()
dow = idx.dayofweek.to_numpy()
weekend = (dow >= 5).astype(int)

# daily/seasonal shape helpers
day_shape = 0.6 + 0.4 * np.sin((hour - 7) / 24 * 2 * np.pi) \
            + 0.5 * np.exp(-((hour - 20) ** 2) / 8)           # evening peak
season = 1 + 0.25 * np.cos((doy - 15) / 365 * 2 * np.pi)      # winter higher
daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)  # 0 at night

# ── Per-supply-point consumption ──
scale = {"C_SP1": 0.5, "C_SP2": 0.1, "C_SP3": 0.13, "C_SP4": 0.5,
         "C_SP5": 0.22, "C_SP6": 1.05}
cons = {}
for col, s in scale.items():
    base = s * day_shape * season * (1 - 0.25 * weekend)
    noise = RNG.normal(0, 0.15 * s, n).clip(-0.5 * s, None)
    cons[col] = np.round(np.clip(base + noise, 0.001, None), 3)

for k in range(1, 7):
    col = f"C_SP{k}"
    pd.DataFrame({"date_time": idx, col: cons[col]}).to_csv(
        OUT / f"consumption_SP{k}.csv", index=False)

# ── Weather features (synthetic) ─────────────────────────────────────────────
temp = 12 + 10 * np.cos((doy - 200) / 365 * 2 * np.pi) + 4 * np.sin((hour - 15) / 24 * 2 * np.pi)
sw = 900 * daylight * season
weather = {
    "temperature_2m (°C)": np.round(temp, 1),
    "relative_humidity_2m (%)": np.round((70 - 0.8 * (temp - 12) + RNG.normal(0, 5, n)).clip(20, 100), 0),
    "dew_point_2m (°C)": np.round(temp - RNG.uniform(2, 8, n), 1),
    "apparent_temperature (°C)": np.round(temp - RNG.uniform(0, 3, n), 1),
    "precipitation (mm)": np.round((RNG.random(n) < 0.05) * RNG.exponential(1.0, n), 2),
    "cloud_cover (%)": np.round(RNG.uniform(0, 100, n), 0),
    "wind_speed_10m (km/h)": np.round(RNG.gamma(2, 5, n), 1),
    "wind_direction_10m (°)": np.round(RNG.uniform(0, 360, n), 0),
    "wind_gusts_10m (km/h)": np.round(RNG.gamma(2, 8, n), 1),
    "shortwave_radiation_instant (W/m²)": np.round(sw, 1),
    "direct_radiation_instant (W/m²)": np.round(sw * 0.7, 1),
    "diffuse_radiation_instant (W/m²)": np.round(sw * 0.3, 1),
}

meteo = pd.DataFrame({"date_time": idx})
for k in range(1, 7):
    meteo[f"C_SP{k}"] = cons[f"C_SP{k}"]
meteo["Hour"] = hour
meteo["DoW"] = dow
meteo["Month"] = idx.month
meteo["Weekend"] = weekend
meteo["Holidays"] = 0
for name, vals in weather.items():
    meteo[name] = vals
meteo.to_csv(OUT / "consumption_meteo_calendar.csv", index=False)

# ── Prices (EUR/kWh): PVPC (buy) and Compensation (sell) ─────────────────────
pvpc = 0.24 + 0.06 * np.sin((hour - 20) / 24 * 2 * np.pi) + RNG.normal(0, 0.02, n)
comp = (0.13 + 0.03 * np.sin((hour - 14) / 24 * 2 * np.pi) + RNG.normal(0, 0.01, n)).clip(0.02, None)
pd.DataFrame({"datetime": idx, "PVPC": np.round(pvpc, 4),
              "Compensation": np.round(comp, 4)}).to_csv(
    OUT / "prices_cv_folds.csv", index=False)

# ── Solar generation + observed battery-controller behaviour ─────────────────
pv_total = 32 * daylight * season * RNG.uniform(0.6, 1.0, n)      # kWh, up to ~PV peak
victron = np.round(pv_total * 0.55, 3)
huawei = np.round(pv_total * 0.45, 3)
# naive controller: charge on midday surplus, discharge in the evening
battery = np.where((hour >= 10) & (hour <= 15), RNG.uniform(0, 6, n),
                   np.where((hour >= 18) & (hour <= 22), -RNG.uniform(0, 5, n), 0.0))
pd.DataFrame({"datetime": idx,
              "PV to inverter Victron": victron,
              "PV to Inverter Huawei": huawei,
              "Battery Power": np.round(battery, 3)}).to_csv(
    OUT / "solar_generation.csv", index=False)

print(f"Synthetic dataset written to {OUT} ({n} hourly rows, year {YEAR}).")
