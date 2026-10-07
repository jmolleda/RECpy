# Your own community

Nothing about the community in the published study is written into the code.
Running RECpy on a different one means describing it in a configuration file.

## What you need

| Data | Shape |
| --- | --- |
| One consumption file per consumer | hourly, a timestamp column and a consumption column |
| Hourly prices | a timestamp, the import price and the export price |
| Hourly generation | a timestamp, one column per array or inverter, and the metered battery flow |
| Hourly weather and calendar | a timestamp, the weather features, and **every** consumer's consumption alongside them |

All of it hourly, covering whole days, with no gaps, spanning the training
history and the periods you want to test on.

!!! warning "Hourly, for now"
    The pipeline models hourly data and reshapes each series into whole days.
    Sub-hourly series — which is what most European meters produce — need
    resampling before use. Supporting them directly is planned, not present.

## Describe it

Copy `config/default.toml` and change the parts that describe the community.
The three sections that matter most:

```toml
[paths]
data = "/path/to/my/csv/files"

[columns]
# The names inside your files. Set these rather than renaming your columns.
timestamp_consumption = "Timestamp"
timestamp_market      = "Timestamp"
import_price          = "buy_eur_kwh"
export_price          = "sell_eur_kwh"
generation            = ["roof_array_kw", "barn_array_kw"]
battery_power         = "battery_kw"

[data]
# Every consumption column in the weather file, including any you are not
# analysing: the models have to set the others aside.
consumption_columns = ["meter_01", "meter_02", "meter_03"]

[[supply_points]]
id          = "meter_01"
history_csv = "meter_01.csv"
column      = "meter_01"
# deployed_model omitted: the comparison picks the model for this consumer
```

Then the periods, the battery and the grid limits — see the
[configuration reference](configuration.md) for every key.

## Check it before running anything

```bash
python code/check_data.py --config config/mine.toml
```

This is the step worth not skipping. It reports, per file:

- whether the timestamp column is there and parses as dates;
- whether the series is hourly, in order, without duplicates or gaps;
- whether it covers whole days;
- whether it spans the training start and every period you configured;
- whether the named columns exist, with a list of what the file does have;
- whether the values are numeric, non-missing and plausible.

A clean run names each file and its span, so you can see it looked:

```
  ok       data folder                        /path/to/my/csv/files
  ok       prices.csv                         8,760 rows, 2025-01-01 to 2025-12-31, 2 columns
  ok       meter_01.csv                       8,760 rows, 2025-01-01 to 2025-12-31, 1 columns

The data is the shape the pipeline expects.
```

A problem names the file and the remedy rather than surfacing as an error from
inside a forecasting script an hour into a run:

```
  ERROR    meter_03.csv    no column named 'Timestamp'. It has: fecha, kwh.
                           Set the timestamp name under [columns] to match.
  ERROR    prices.csv      the usual step between rows is 0:30:00, not one hour.
  ERROR    meter_02.csv    1 gaps differ from 1:00:00, the first at 2025-03-14 02:00
  ERROR    meter_01.csv    ends 2025-05-31, before "Fold 2 – Jul" finishes 2025-07-31
```

Warnings do not stop a run; they are things worth knowing, such as consumption
that goes negative.

The web interface runs the same checks from its **Check my data** button.

## Choosing the models

By default, leave `deployed_model` out of each consumer and let the comparison
decide: every selected model is trained for every consumer, and the winner of
that consumer's Wilcoxon tournament drives its schedule. That is the method the
article describes, and the run reports which model it used for each.

How many candidates to compare is a trade-off between time and confidence. In
the interface this is the **Effort** control; in a configuration file it is
`models.families`, `models.ml_subset` and `models.max_test_days`.

## Expect

The machine-learning stage dominates: roughly one and a half seconds per
regressor per forecast day, per consumer. Fifteen regressors over a thirty-day
period is ten to fourteen minutes for **one** consumer and **one** period. The
statistical stage adds a few seconds per consumer-day, and each neural-network
fit is about half a minute.

Start with `max_test_days = 7` and a handful of regressors. Widen once the setup
is right.
