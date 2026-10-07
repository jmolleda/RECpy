# Reading the results

Everything a run produces lands under the output root — `paths.output`, or
whatever `--output` / `RECPY_OUTPUT_DIR` points at.

```
<output>/
├── run_manifest.json                  what produced these results
├── model_failures.csv                 only if a model failed
├── cross_validation/
│   ├── <period>/Predictions_C_<ID>_<period>.csv
│   ├── wilcoxon_cv_per_fold.csv
│   ├── wilcoxon_cv_aggregated.csv
│   └── figures/fig1..fig7 .png and .pdf
└── optimization_results/
    ├── optimization_cv_hourly.csv
    ├── optimization_cv_daily.csv
    └── optimization_cv_summary.csv
```

---

## The prediction files

One per consumer per period, and the central artifact of a run: the observations
and every model's forecast, side by side, on the same hourly index.

| Column | Contents |
| --- | --- |
| `Observed` | What the consumer actually used. |
| one per model | That model's forecast for the same hour. |
| `Naive-168` | The seasonal baseline: the same hour one week earlier. |

A full run has 22 columns. Everything downstream is derived from these, so they
are the place to start if a number looks wrong.

---

## The model comparison

`wilcoxon_cv_per_fold.csv` has one row per consumer, period and model, with its
MAE, RMSE, wins, losses and **net wins**. `wilcoxon_cv_aggregated.csv` averages
across periods and counts how often each model came top.

Net wins is the figure to read: how many models this one beats *significantly*,
minus how many significantly beat it. A model with the lowest MAE but no
significant wins has not demonstrated anything.

!!! warning "Few test days, few conclusions"
    With a short test window the tournament will still produce a ranking, and it
    will not mean much. `config/quick.toml` uses five days; the published
    configuration uses whole months.

---

## The schedule and the saving

| File | Grain |
| --- | --- |
| `optimization_cv_hourly.csv` | Every hour: demand, generation, prices, battery charge, discharge and state of charge, grid import, export, and the cost. |
| `optimization_cv_daily.csv` | Each day's cost under the three baselines, and the savings. |
| `optimization_cv_summary.csv` | Per period and overall. |

The columns to read in the summary are **`Saving vs B2`** — against the battery
as actually operated today, which is the honest comparison for a community that
already has one — and **`Saving vs B1`**, against having no battery at all.

!!! note "A negative month is a result, not a bug"
    Savings are costed against realised consumption. A period where the
    optimised schedule does no better is a finding about forecast quality in
    that season, and the article reports one.

---

## What produced the results

`run_manifest.json` describes the run itself: the configuration in full, the git
commit (and whether the working tree was clean), Python and package versions,
when it ran and how long each stage took.

```json
{
  "software":      { "describe": "v2.0.0", "uncommitted_changes": false },
  "run":           { "duration_seconds": 270.4, "stages": [ ... ] },
  "configuration": { "name": "quick", "supply_points": ["SP2", "SP3"], ... },
  "environment":   { "python": "3.12.10", "packages": { ... } }
}
```

Hand someone a results directory and they can tell what made it.

## If a model failed

A model that raises is skipped rather than killing a run of several hours, but
the reason is recorded in `model_failures.csv` and listed at the end of the run:

```
  1 model(s) failed and were left out of the comparison:
    machine learning: SP3 MLP - ValueError: ...
  listed in <output>/model_failures.csv
```

Worth checking when a comparison has fewer models than you expected.
