# Getting started

## Install

Python 3.12 and the pinned dependencies:

```bash
git clone https://github.com/jmolleda/RECpy.git
cd RECpy
pip install -r requirements.txt
```

The versions are pinned because the results depend on them. `requirements.txt`
covers the pipeline; two optional files add the extras:

| File | Adds |
| --- | --- |
| `requirements-gui.txt` | the local web interface |
| `requirements-docs.txt` | this documentation site |

## Run something small first

The repository ships a **fully synthetic** example dataset, so the pipeline runs
end to end without the real household data, which cannot be redistributed. The
reduced configuration exercises every stage in about five minutes:

```bash
python code/run_pipeline.py --config config/quick.toml --output /tmp/recpy-quick
```

You should see the six stages announce themselves and report their times:

```
========================================================================
  Machine-learning forecasts  (generate_ml_cv.py)
========================================================================
  --- SP2 (SP2) ---
    [SP2] Huber ... MAE=0.0122  RMSE=0.0152
  ...
  PIPELINE SUMMARY  -  configuration 'quick'
  Machine-learning forecasts              44.9 s
  Statistical forecasts                   45.8 s
  Deep-learning forecasts                173.8 s
  Wilcoxon model selection                 1.6 s
  Visualization                            4.3 s
  Day-ahead scheduling                     1.1 s
  TOTAL                                  270.4 s
```

!!! warning "It is a smoke test, not a result"
    `quick.toml` reduces the *scale*, not the method: two consumers, one period,
    the first few days, five of the fifteen regressors. Every model keeps its
    published settings, but five test days are far too few for the comparison to
    be conclusive. Reproducing the published numbers needs
    `config/default.toml` and the better part of a day.

## The web interface

For setting up a community without editing TOML by hand:

```bash
pip install -r requirements-gui.txt
streamlit run app/recpy_app.py
```

Then open <http://127.0.0.1:8501>. Three tabs:

- **My community** — your data files and the columns inside them, your
  consumers, your battery and grid limits, and how thorough the model comparison
  should be. The sidebar says whether the setup is ready to run.
- **Run** — a time estimate, then the pipeline with a per-stage progress bar and
  a Stop button.
- **Results** — savings, the battery schedule hour by hour, forecast accuracy,
  and the model comparison behind them.

!!! info "It stays on your machine"
    The interface binds to `127.0.0.1`. Household consumption is personal data,
    so there is no hosted version and nothing is uploaded. If you serve it
    beyond this machine, put authentication in front of it and point it at
    synthetic data.

## Reproduce the published study

```bash
python code/run_pipeline.py                  # config/default.toml
```

Six consumers, four seasonal periods, every model. Expect **hours**, dominated
by the machine-learning stage: fifteen regressors re-fitted for each of thirty
walk-forward days, for every consumer and period.

The additional experiments reported in the article are separate scripts, listed
under [What the pipeline does](pipeline.md).

## Next

- [Your own community](your-own-community.md) — pointing it at your data
- [Reading the results](results.md) — what the output files contain
