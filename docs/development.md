# Development

## Layout

```
app/     local web front end (Streamlit) and the application icon
code/    the pipeline, the configuration layer and the checks
config/  experiment configurations
data/    a fully synthetic example dataset
docs/    this site
tests/   the test suite
```

Within `code/`:

| | |
| --- | --- |
| `recpy_config.py` | loads the active configuration; every other script imports its settings from here. Run it directly to print what resolves. |
| `run_pipeline.py` | runs the stages in dependency order, times them, writes the manifest. |
| `check_data.py` | checks a dataset before anything expensive starts. |
| `run_report.py` | the run manifest and the record of failed models. |
| `generate_*_cv.py` | the three forecasting stages. |
| `wilcoxon_cv*.py` | model selection and its robustness checks. |
| `opt_core.py`, `run_optimization_cv.py`, `optimization_*.py` | the scheduling linear programme and its sensitivity analyses. |
| `visualize_cv.py` | the visualizations. |
| `make_synthetic_data.py` | the example dataset. |
| `test_config_equivalence.py` | equivalence with the archived release. |

## Tests

```bash
pip install pytest tomli-w
python -m pytest                 # the whole suite, about 20 seconds
python -m pytest -m "not slow"   # skip the two end-to-end runs
```

| File | Covers |
| --- | --- |
| `test_config.py` | what the configuration resolves and what it refuses, including the per-entity settings a blanket edit could silently drop |
| `test_check_data.py` | the data checker against a dataset damaged one fault at a time, asserting the *message*, not just that it fired |
| `test_run_report.py` | the manifest and the record of failed models |
| `test_pipeline.py` | the 108-value equivalence check, stage ordering, and two end-to-end runs — one minimal, one on a dataset using another community's column names |

Continuous integration runs the suite on every push, installing the **pinned**
requirements, so a red build also means the declared environment has stopped
installing.

## Conventions worth knowing

**Settings belong in the configuration, not in the scripts.** Three settings
were silently lost when the scripts were first migrated onto the configuration
layer, and two of them left code that still imported and compiled. If you add a
per-consumer or per-period setting, add it to `config/default.toml`, expose it
through `recpy_config.py`, and add it to `test_config_equivalence.py` so it
cannot be dropped unnoticed.

**A stage announces when it finishes.** `run_pipeline.py` reads progress from
the line each stage prints, and the web interface reads the same line. Changing
that format changes both.

**Failures are recorded, not just printed.** A model that raises is skipped so a
long run survives, but `record_failure()` writes the reason where a later reader
will find it.

**The front end adds no analysis.** It edits a configuration, runs the pipeline
as a subprocess and reads the results back. A chart obtained through the
interface is the chart the scripts produce, and a configuration saved from it
runs unchanged from the command line.

## Building this site

```bash
pip install -r requirements-docs.txt
mkdocs serve      # http://127.0.0.1:8000
```

It is published to GitHub Pages by an Action on every push to `main`.

## Known limitations

- **Hourly data only.** The pipeline reshapes each series into whole days;
  sub-hourly series need resampling first.
- **Not installable as a package.** There is no `pyproject.toml` yet; scripts
  are run from `code/`.
- **ARIMA falls back silently.** A day whose ARIMA fit fails is filled with
  persistence, so an `ARIMA` column can contain naive forecasts for some days.
- **Retrospective only.** The framework evaluates on past windows so results can
  be scored against what actually happened. It does not produce tomorrow's
  schedule from today's data.
