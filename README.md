# A Particularized Forecasting and Optimization Framework for Rural Energy Communities — Code

This repository contains the source code accompanying the paper:

> *A Particularized Forecasting and Optimization Framework for Rural Energy Communities.*
> Machine Learning with Applications (under review), 2026.
>
> Author names, affiliations, and citation details are omitted for double-blind
> peer review, and will be added upon acceptance.

It provides the full forecasting, statistical model-selection, and day-ahead
optimization pipeline evaluated on an operational rural energy community (EC) of
six supply points, anonymized here as **SP1–SP6**.

## Repository layout

```
code/    all analysis scripts (+ make_synthetic_data.py)
data/    a fully synthetic example dataset (see below)
requirements.txt
LICENSE  (CC BY 4.0)
```

Running the scripts creates their own output folders under `code/`
(`cross_validation/`, `cross_validation/figures/`, `optimization_results/`),
so no manual setup is required.

## Setup

Python 3.12 with the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Data availability

The **raw household consumption series are not included** for privacy reasons;
they can be requested from the authors (contact details withheld for double-blind
peer review).

So that the pipeline can nonetheless be **run end-to-end**, the `data/` folder
ships a **fully synthetic** example dataset produced by
[`code/make_synthetic_data.py`](code/make_synthetic_data.py) (seeded and
reproducible). It contains **no real consumption** — every series is drawn from
simple seasonal models plus noise — and follows the same schema and file names
the scripts expect. Regenerate it at any time with:

```bash
python code/make_synthetic_data.py
```

Expected files and columns (real or synthetic):

| File | Columns |
|------|---------|
| `data/consumption_SP1.csv` … `consumption_SP6.csv` | `date_time`, `C_SPk` |
| `data/consumption_meteo_calendar.csv` | `date_time`, `C_SP1`…`C_SP6`, `Hour`, `DoW`, `Month`, `Weekend`, `Holidays`, weather features |
| `data/prices_cv_folds.csv` | `datetime`, `PVPC`, `Compensation` |
| `data/solar_generation.csv` | `datetime`, `PV to inverter Victron`, `PV to Inverter Huawei`, `Battery Power` |

Prices are taken from the Spanish TSO ESIOS service and weather features from the
Open-Meteo API; the corresponding synthetic columns imitate their format.

## Running the full experiment

With a dataset in `data/` (synthetic or real), reproduce the study by running, in
order:

```bash
# 1. Forecasting -- build the per-fold prediction files
python code/generate_ml_cv.py          # ML/statistical regressors + Naive-168
python code/generate_arima_cv.py       # ARIMA / SARIMAX
python code/generate_dl_cv.py          # LSTM / GRU / TCN (single-fit)
python code/generate_dl_cv_daily.py    # LSTM / GRU / TCN (daily-retrained)

# 2. Statistical model selection
python code/wilcoxon_cv.py             # Wilcoxon net-wins tournament
python code/wilcoxon_cv_robust.py      # robustness: daily agg., Holm/BH, Diebold-Mariano

# 3. Figures
python code/visualize_cv.py            # figures -> code/cross_validation/figures/

# 4. Optimization
python code/run_optimization_cv.py                 # day-ahead LP over the folds
python code/optimization_degradation.py            # battery-degradation sensitivity
python code/optimization_lookahead_downstream.py   # PV look-ahead + downstream cost

# 5. Additional analyses
python code/ensemble_analysis.py       # forecast-combination strategies
python code/cold_start_analysis.py     # limited-history (cold-start) experiment
python code/timing_analysis.py         # training / retraining times
```

## Reproducibility

- **Python** 3.12; dependencies pinned in `requirements.txt`.
- **Random seed** `RS = 123` (with `PYTHONHASHSEED=0`, and NumPy/TensorFlow
  seeded) is fixed in the training scripts.
- **Calendar year**: fold boundaries are specified as month–day only and combined
  with a `YEAR` constant (placeholder `2025`) defined at the top of each script;
  the synthetic dataset uses the same value. Set `YEAR` to the calendar year of
  your data before running on real data.
- **Evaluation protocol**: blocked expanding-window cross-validation with four
  seasonal folds (April, July, October, December) and walk-forward 24-hour
  forecasting. ML/statistical models are univariate (192-hour consumption lag);
  deep-learning models are multivariate (calendar + weather features).

## Code layout (`code/`)

**Forecasting**
- `generate_ml_cv.py` — 15 ML/linear regressors, expanding-window walk-forward.
- `generate_arima_cv.py` — ARIMA / SARIMAX folds.
- `generate_dl_cv.py`, `generate_dl_cv_daily.py` — LSTM/GRU/TCN (single-fit and
  daily-retrained regimes); architecture grids and seeds are defined inside.

**Statistical model selection**
- `wilcoxon_cv.py` — pairwise Wilcoxon signed-rank tournament (net wins).
- `wilcoxon_cv_robust.py` — robustness: daily-aggregated errors, Holm and
  Benjamini–Hochberg correction, and Diebold–Mariano (HAC + HLN) tests.
- `wilcoxon_model_selection.py`, `wilcoxon_test.py` — helpers.

**Optimization**
- `opt_core.py` — shared LP core (Pyomo + HiGHS): parameterized battery
  degradation model, baselines, and a realized-cost evaluator.
- `run_optimization_cv.py` — day-ahead LP over the CV folds.
- `optimization_degradation.py` — degradation-model / cost sensitivity sweep.
- `optimization_lookahead_downstream.py` — PV look-ahead (perfect vs realistic)
  and Wilcoxon-vs-RMSE downstream-cost comparison.

**Additional analyses**
- `ensemble_analysis.py` — forecast combination (top-k average, NNLS stacking).
- `cold_start_analysis.py` — limited-history (2/4/8-week) onboarding experiment.
- `timing_analysis.py` — training / retraining time measurement.

**Figures**
- `visualize_cv.py` — generates the result figures.

**Synthetic data**
- `make_synthetic_data.py` — generates the synthetic example dataset in `data/`.

## License

Released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**; see
`LICENSE`. Author, affiliation, and citation information will be added upon
acceptance.
