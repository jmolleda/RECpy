"""
Run the whole RECpy pipeline, in dependency order, with one command.

    python code/run_pipeline.py                       # config/default.toml
    RECPY_CONFIG=config/quick.toml python code/run_pipeline.py
    RECPY_OUTPUT_DIR=/results python code/run_pipeline.py

Stages run as separate processes so that a heavy optional dependency
(TensorFlow) is imported only by the stage that needs it, and so that one
stage failing cannot leave a half-initialised module behind. Each stage is
timed and a summary table is printed at the end.

Stage order matters: generate_ml_cv.py creates one prediction CSV per supply
point and fold; generate_arima_cv.py and generate_dl_cv.py add their columns to
those same files; wilcoxon_cv.py ranks every column; and run_optimization_cv.py
schedules the battery on the forecasts of the statistically selected model.

Which stages run is decided by the active configuration: the statistical and
deep-learning stages are skipped unless "arima" and "dl" appear in
[models].families.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from recpy_config import CONFIG

# Stage output goes straight to the inherited stream, so the driver's own
# prints must not sit in a block buffer when stdout is a pipe or a log file.
sys.stdout.reconfigure(line_buffering=True)

CODE = Path(__file__).resolve().parent

# (script, label, model family it needs or None if always run)
STAGES = [
    ("generate_ml_cv.py",     "Machine-learning forecasts",   "ml"),
    ("generate_arima_cv.py",  "Statistical forecasts",        "arima"),
    ("generate_dl_cv.py",     "Deep-learning forecasts",      "dl"),
    ("wilcoxon_cv.py",        "Wilcoxon model selection",     None),
    ("visualize_cv.py",       "Result figures",               None),
    ("run_optimization_cv.py", "Day-ahead scheduling",        None),
]


def ensure_data() -> None:
    """Generate the synthetic dataset if the configured data files are absent."""
    missing = [k for k in CONFIG.raw["data_files"]
               if not CONFIG.data_file(k).is_file()]
    if not missing:
        return
    print(f"Data files missing ({', '.join(missing)}); generating synthetic data.")
    run_stage("make_synthetic_data.py", "Synthetic dataset")


def run_stage(script: str, label: str) -> float:
    print()
    print("=" * 72)
    print(f"  {label}  ({script})")
    print("=" * 72)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(CODE / script)], cwd=CODE)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"FAILED: {script} exited with code {proc.returncode}")
        sys.exit(proc.returncode)
    print(f"  -> {label} finished in {elapsed:,.1f} s")
    return elapsed


def main() -> None:
    print(CONFIG.summary())
    if CONFIG.max_test_days:
        print(f"  test window     : first {CONFIG.max_test_days} day(s) of each fold")
    if CONFIG.ml_subset:
        print(f"  ML regressors   : {len(CONFIG.ml_subset)} of 15 "
              f"({', '.join(CONFIG.ml_subset)})")

    problems = CONFIG.validate()
    if problems:
        print()
        print("This configuration cannot be run:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(2)

    ensure_data()

    timings = []
    skipped = []
    for script, label, family in STAGES:
        if family and not CONFIG.runs(family):
            skipped.append(f"{label} (family '{family}' not in this configuration)")
            continue
        timings.append((label, run_stage(script, label)))

    print()
    print("=" * 72)
    print(f"  PIPELINE SUMMARY  -  configuration '{CONFIG.name}'")
    print("=" * 72)
    for label, elapsed in timings:
        print(f"  {label:<34} {elapsed:>9,.1f} s")
    print(f"  {'TOTAL':<34} {sum(t for _, t in timings):>9,.1f} s")
    for note in skipped:
        print(f"  skipped: {note}")
    print(f"  outputs under {CONFIG.output_root}")


if __name__ == "__main__":
    main()
