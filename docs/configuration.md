# Configuration reference

Every setting lives in a TOML file. Nothing about the community, the data, the
models or the hardware is written into the scripts, so adapting RECpy means
writing a configuration, not editing code.

```bash
python code/run_pipeline.py                        # config/default.toml
python code/run_pipeline.py --config config/mine.toml
RECPY_CONFIG=config/mine.toml python code/...      # same thing, as an env var
python code/recpy_config.py                        # print what resolves
```

Two configurations ship with the software:

| File | Purpose |
| --- | --- |
| `config/default.toml` | The published experiment: 6 consumers, 4 seasonal periods, every model. Reproduces the results in the article. |
| `config/quick.toml` | A reduced run of the same pipeline in about five minutes. For smoke-testing. Does **not** reproduce the published numbers. |

!!! tip "Check before you run"
    `python code/check_data.py` reports whether the files a configuration names
    are the shape the pipeline expects, before anything expensive starts. See
    [Your own community](your-own-community.md).

---

## Identity

| Key | Meaning |
| --- | --- |
| `name` | A label for this setup. Appears in the run summary and the manifest. |
| `description` | Free text, kept with the configuration. |
| `year` | Calendar year of your data. Period boundaries below are month–day only and are combined with this, so a different year of data needs only this one change. |

---

## `[paths]`

| Key | Meaning |
| --- | --- |
| `data` | Folder holding the CSV files. Relative to the repository, or absolute. |
| `output` | Where results are written. `RECPY_OUTPUT_DIR` or `--output` overrides it. |

!!! warning "Hosted platforms"
    The output root is overridable on its own because reproducibility platforms
    such as Code Ocean discard anything written under the code directory and
    capture only a dedicated results directory. A run that wrote into `code/`
    there would report success and save nothing.

---

## `[data]` and `[data_files]`

| Key | Meaning |
| --- | --- |
| `data.consumption_columns` | **Every** consumption column in the weather file, including consumers you are not analysing. The forecasting models take one as their target and must set the others aside; omitting one would train a model on the wrong meter. |
| `data_files.prices` | Hourly import and export prices. |
| `data_files.solar` | Hourly generation, and the metered battery flow. |
| `data_files.meteo_calendar` | Hourly weather and calendar features, with every consumer's consumption alongside them. |

---

## `[columns]`

The names *inside* your CSV files. Set these to match your data rather than
renaming your columns. The defaults are the names this study's files use.

| Key | Default | Meaning |
| --- | --- | --- |
| `timestamp_consumption` | `date_time` | Timestamp column of the consumption and weather files. |
| `timestamp_market` | `datetime` | Timestamp column of the price and generation files. The two differ because the source datasets did. |
| `import_price` | `PVPC` | Price paid for energy drawn from the grid. |
| `export_price` | `Compensation` | Price received for energy exported. |
| `generation` | two inverters | One entry per array or inverter; they are summed to give total production. A single string is accepted. |
| `battery_power` | `Battery Power` | Metered battery flow, positive when charging. Used for the baseline that reflects how the battery is run today. |

---

## `[models]`

| Key | Meaning |
| --- | --- |
| `families` | Which kinds of model to train: `ml`, `arima`, `dl`. Each is a separate stage; dropping one skips that stage entirely. |
| `ml_subset` | Which of the fifteen machine-learning regressors to run. **An empty list means all of them.** Any model named as a consumer's `deployed_model` must appear here. |
| `max_test_days` | Forecast only the first *N* days of each period. **`0` means the whole period.** The quickest way to try the pipeline out. |

---

## `[[folds]]` — the periods to test on

One table per period. Called a *fold* in the method.

| Key | Meaning |
| --- | --- |
| `name` | Identifier; names the output folder for this period. |
| `label` | Written into the optimization result files as a key. Changing it changes those keys, so comparisons with earlier runs break. |
| `train_end` | Training ends here (month–day). |
| `test_start`, `test_end` | The window forecast day by day (month–day). |

Models train on everything from `forecasting.train_start` up to `train_end`,
then forecast across the window, re-fitting as each day's observations arrive.

---

## `[[supply_points]]` — the consumers

| Key | Meaning |
| --- | --- |
| `id` | Short name. Used in output filenames. |
| `history_csv` | That consumer's hourly consumption file, relative to `paths.data`. |
| `column` | Which column to read from it, and from the weather file. |
| `deployed_model` | *Optional.* The model whose forecasts drive the battery schedule. **Leave it out** to use whichever model wins the comparison for that consumer. |

---

## `[arima.<ID>]` — statistical model orders

One table per consumer, from the order selection reported in the article.

| Key | Meaning |
| --- | --- |
| `order` | ARIMA `(p, d, q)`. |
| `sarimax_order` | SARIMAX `(p, d, q)`. |
| `sarimax_seasonal` | Seasonal `(P, D, Q, s)`; `s = 24` for a daily cycle. |
| `sarimax_exog` | Exogenous columns, by name, from the weather file. |

---

## `[dl.<ID>]` — neural-network architectures

One table per consumer, each with `lstm`, `gru` and `tcn`, from the architecture
search in the article's appendix.

| Key | Applies to | Meaning |
| --- | --- | --- |
| `n_input` | all | Hours of history fed to the network. |
| `nodes` | `lstm`, `gru` | Units in the recurrent layer. |
| `filters` | `tcn` | Convolution filters. |
| `dilations` | `tcn` | Dilation factors, e.g. `[1, 2, 4, 8]`. |
| `epochs`, `batch`, `dropout` | all | Training settings. |

---

## `[battery]` and `[grid]`

| Key | Meaning |
| --- | --- |
| `capacity_kwh` | Usable capacity. |
| `min_soc_kwh` | Reserve the schedule never draws below. |
| `power_limit_kw` | Charge and discharge limit. |
| `efficiency` | Round-trip efficiency, as a fraction. |
| `degradation_eur_per_kwh` | Wear priced into the schedule, so it does not cycle the battery for savings smaller than the wear they cause. |
| `initial_soc_fraction` | Charge assumed at the start of the first day, as a fraction of capacity. |
| `grid.export_limit_kw` | Maximum export to the grid. |

---

## `[forecasting]`

| Key | Meaning |
| --- | --- |
| `train_start` | First day of training history (month–day). |
| `lookback_hours` | How far back each forecast looks. `192` is eight days. |
| `horizon_hours` | Hours forecast at a time. `24` for day-ahead, which is what the schedule needs. |
| `random_seed` | Fixed so a rerun reproduces the same models. See [Reproducibility](reproducibility.md) for the one place this is not quite enough. |

---

## Consistency checks

The configuration is validated before any stage runs. It is rejected, with the
reason, when:

- a model family is named that the pipeline does not know;
- a consumer's `deployed_model` is not among `models.ml_subset`, which would
  leave the scheduling stage with no forecasts for it;
- a period or a consumer is missing a required field;
- `arima` or `dl` is enabled but a consumer has no corresponding table;
- a consumer's `column` is not in `data.consumption_columns`;
- a consumption file does not exist.
