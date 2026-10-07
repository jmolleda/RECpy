# Reproducibility

## What is fixed

| | |
| --- | --- |
| **Python** | 3.12 |
| **Dependencies** | every one pinned in `requirements.txt` |
| **Random seed** | `forecasting.random_seed = 123`, with NumPy and TensorFlow seeded |
| **Evaluation** | blocked expanding-window cross-validation, four seasonal periods, walk-forward 24-hour forecasting |

## What is not quite bit-for-bit

Every model is exactly reproducible **except Random Forest and Extra Trees**.
Both are fitted with `n_jobs=-1`, so their trees are averaged in whatever order
the threads happen to finish. Their predictions move by about one machine
epsilon — around `1e-16` — between runs of *identical* code.

Re-running therefore reproduces 20 of the 22 prediction columns byte for byte,
and those two to about `1e-15` relative. Setting `n_jobs=1` removes the
variation at a large cost in time.

!!! success "It reaches no conclusion"
    The Wilcoxon outputs computed from those columns are byte-identical across
    runs. No reported figure, ranking or saving depends on the difference.

## Equivalence with the archived release

v2.0.0 made the pipeline configurable. It did not change the method, and that is
checked rather than asserted.

```bash
python code/test_config_equivalence.py
```

This asserts that `config/default.toml` resolves to the **108 values** the
archived v1.0.0 scripts held as literals, across all six consumers and all four
periods: period boundaries and labels, consumers and their files, the deployed
models, neural-network hyperparameters and feature exclusions, ARIMA and SARIMAX
orders with their exogenous variables, battery and grid limits, the seed, the
lookback and the horizon.

The transcribed literals were themselves checked against the tagged v1.0.0
sources, so the test does not merely agree with itself.

Beyond that, a full forecasting run for the richest consumer was compared
against v1.0.0 column by column: **20 of 22 prediction columns byte-identical**,
both Wilcoxon output files byte-identical, and the two thread-parallel ensembles
differing by `6.7e-16` for the reason above.

Run this after changing anything in `config/default.toml`.

## Why the output root is configurable

Hosted reproducibility platforms such as Code Ocean discard anything written
under the code directory and capture only a dedicated results directory. In
v1.0.0 every output path resolved inside `code/`, so an unmodified import would
have run, reported success, and saved nothing. Set `RECPY_OUTPUT_DIR` — or
`--output` — to whatever the platform captures.

## The synthetic dataset

The real household series cannot be redistributed: they are personal data, and
the participating households agreed to a study, not to publication. The
repository therefore ships a fully synthetic dataset of the same shape, drawn
from simple seasonal models plus noise.

It exercises every code path and reproduces **no** published number. It is
generated with a fixed seed, so `python code/make_synthetic_data.py` rewrites
exactly the files already committed.

The real data may be requested from the corresponding author, and will be shared
where the privacy constraints of the participating households allow.
