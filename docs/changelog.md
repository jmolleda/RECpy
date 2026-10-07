# Changelog

!!! info "This page is generated"
    It is copied from `CHANGELOG.md` at the repository root when the site is
    built, so the two cannot drift. Edit that file, not this one.

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/); each released version is archived on
Zenodo.

## [2.0.0] — 2026-10-07

**This version computes the same results as v1.0.0**: the pipeline was made
configurable and reusable, not changed. See *Verification* below.

The major version marks a breaking interface change, not changed behaviour:
settings moved out of the scripts, so code that imported constants from them no
longer works. Nothing about the method changed.

### Added

- **Configuration layer** (`code/recpy_config.py`, `config/*.toml`). Every
  setting that used to be a literal is now read from a TOML file.
  `RECPY_CONFIG` selects one; `RECPY_OUTPUT_DIR` redirects outputs.
  TOML is read with the standard library, so the package gains no dependency.
- **Configurable column names** (`[columns]`). The timestamp, the import and
  export price columns, the generation columns and the metered battery flow are
  no longer fixed to this study's names. A community can point the framework at
  its own files instead of renaming their columns.
- **`code/run_pipeline.py`**, running the six stages in dependency order,
  timing each and skipping the model families a configuration excludes.
- **`code/check_data.py`**, which reports whether a dataset is the shape the
  pipeline expects, before anything expensive starts.
- **Local web front end** (`app/recpy_app.py`, Streamlit). Edits a
  configuration, runs the experiment and reads the results back: savings,
  the battery schedule hour by hour, forecast accuracy, and the model comparison
  behind them. It binds to `127.0.0.1`, because the series it reads are
  personal data.
- **Reduced configuration** (`config/quick.toml`) running every stage at
  published hyperparameters in about five minutes, for smoke-testing and for
  hosted reproducibility platforms. It does not reproduce the published numbers
  and says so.
- **Run manifest** (`run_manifest.json`), written beside the results: the
  configuration used, the git commit, Python and package versions, timings and
  the stages that ran. A results directory now describes itself.
- **Record of failed models** (`model_failures.csv`). A model that fails is
  still skipped rather than killing a long run, but the reason is now written
  down and reported at the end.
- **Test suite** (`tests/`, 67 tests) and continuous integration. Covers what
  the configuration resolves and refuses, the data checker against a dataset
  damaged one fault at a time, and two end-to-end runs.
- **`code/test_config_equivalence.py`**, asserting that the configuration
  resolves to the 108 values the v1.0.0 scripts held as literals.
- **`CITATION.cff` and `codemeta.json`**, so the software can be cited and
  indexed.
- **`--config`, `--output` and `--help`** on the scripts a person runs by hand.

### Changed

- **Licence: CC BY 4.0 → MIT.** MIT is the better fit for this software. The
  v1.0.0 archive remains CC BY.
- **Scheduling without a configured model.** A supply point with no
  `deployed_model` now uses the winner of its own Wilcoxon tournament, and says
  which it used, instead of failing.
- **Deep-learning target selection.** The stage now selects the target column
  and the feature columns rather than dropping the others, so the target is the
  first column by construction. It also refuses to run if its own target is not
  the `Observed` column of the file it is appending to.
- **Software authorship.** v1.0.0 recorded F. Cosío and J. Molleda; v2.0.0
  records J. Molleda and F. Cosío, ordered by participation in this version.

### Removed

- **`code/wilcoxon_model_selection.py` and `code/wilcoxon_test.py`.** Both read
  `Predictions_C_*_RW.csv` from a `models/` directory that nothing in this
  repository produces, so neither could run on a checkout. They referred to an
  earlier rolling-window design superseded by the cross-validation in
  `wilcoxon_cv.py`. They remain in the v1.0.0 archive.

### Fixed

- **Outputs were written under `code/`**, which hosted reproducibility platforms
  discard. The output root is now configurable, and such a run no longer reports
  success having saved nothing.
- Importing `make_synthetic_data` no longer overwrites the dataset: the
  generator is behind `main()`.

### Fixed before release

Defects introduced while this version was being built and fixed before it
shipped. No released version carried them. They are recorded because two of the
three left code that still imported and compiled, which is how they survived
review.

- **Deep-learning hyperparameters and feature exclusions were lost** when the
  scripts were first migrated onto the configuration. Nothing failed until a
  deep-learning run was actually attempted.
- **ARIMA and SARIMAX orders and exogenous variables were lost** the same way,
  surfacing as `KeyError: 'col'` at run time.
- **Deep-learning models could train on the wrong consumer.** With a
  configuration analysing only some supply points, the others stayed in the
  frame and the models trained on the first column, which was then the wrong
  meter. The stage reported a plausible error against its own target while the
  columns it wrote were scored against another.

### Verification

- The configuration resolves to the **108 values** v1.0.0 held as literals,
  across all six supply points and all four folds. The transcription was itself
  checked against the tagged v1.0.0 sources.
- A full forecasting run for the richest supply point reproduces v1.0.0 **column
  for column**: 20 of 22 prediction columns byte-identical, both Wilcoxon output
  files byte-identical. Random Forest and Extra Trees differ by about 6.7e-16,
  which is thread-ordering in their parallel tree averaging. See *Bit-for-bit
  reruns* in the README.

## [1.0.0] — 2026-07-02

The release that produced the results in *Machine Learning with Applications*
**26** (2026) 101016, archived at
[10.5281/zenodo.21129973](https://doi.org/10.5281/zenodo.21129973) under
CC BY 4.0.
