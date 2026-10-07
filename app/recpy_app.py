"""
RECpy - local web front end.

Everything runs on this machine. The consumption series are read from the local
data directory and are never uploaded anywhere, which matters because household
consumption is personal data: a hosted version of this interface would move it
off the premises of the energy community that owns it.

    pip install -r requirements-gui.txt
    streamlit run app/recpy_app.py

The interface is a thin layer over the pieces the command line already uses. It
edits a configuration, writes it out as TOML, runs code/run_pipeline.py against
it as a subprocess, and reads the results back from the output directory. No
analysis happens here, so a chart obtained through the interface is the chart
the scripts produce.

It asks what someone running an energy community needs to answer -- whose meters,
which battery, what would it have saved -- and keeps the method out of the way
until it is wanted. Nothing is withheld: the model families, the individual
regressors, the cross-validation periods, the lookback and the seed are all
reachable behind the controls that say so, and whatever the interface writes is
an ordinary configuration file that runs unchanged from the command line.
"""

from __future__ import annotations

import calendar
import copy
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import tomli_w

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from recpy_config import Config, load  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
ICON = ASSETS / "icon.png"

st.set_page_config(page_title="RECpy", layout="wide",
                   page_icon=str(ICON) if ICON.is_file() else None)

NL = chr(10)
MONTHS = list(calendar.month_name)[1:]

#: What each model family in the configuration actually trains. The keys are
#: terse on purpose -- they name directories and stages -- so the interface shows
#: these instead.
FAMILY_LABELS = {
    "ml": "Machine learning",
    "arima": "Statistical (ARIMA, SARIMAX)",
    "dl": "Neural networks (LSTM, GRU, TCN)",
}

#: How hard to look for each consumer's model, in Community mode. The trade-off
#: a community actually faces is time against confidence, not which regressor to
#: tick, so that is what the control asks about.
THOROUGHNESS = {
    "Quick look": {
        "families": ["ml"],
        "subset": ["Linear Regression", "Huber", "k-Nearest Neighbors",
                   "LightGBM"],
        "days": 7,
        "blurb": "Four fast models over one week. Minutes, and enough to "
                 "confirm the setup works end to end.",
    },
    "Balanced": {
        "families": ["ml", "arima"],
        "subset": ["Linear Regression", "Huber", "k-Nearest Neighbors",
                   "Random Forest", "Extra Trees", "Gradient Boosting",
                   "XGBoost", "LightGBM"],
        "days": 14,
        "blurb": "Eight models plus the statistical ones over a fortnight. "
                 "A reasonable answer without a long wait.",
    },
    "Thorough": {
        "families": ["ml", "arima", "dl"],
        "subset": [],
        "days": 0,
        "blurb": "Everything the framework has, over the whole period. This is "
                 "what the published study ran, and it takes hours.",
    },
}


# ---------------------------------------------------------------------------
# The pipeline: starting it, stopping it, following it
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def regressor_names() -> list[str]:
    """The 15 regressor names, taken from the script that defines them."""
    from generate_ml_cv import get_models
    return list(get_models().keys())


@st.cache_resource(show_spinner=False)
def pipeline_stages() -> list[tuple[str, str, str | None]]:
    """The stages the driver runs, in order, read from the driver itself so the
    progress display cannot drift from what actually runs."""
    from run_pipeline import STAGES
    return list(STAGES)


def start_run(environ: dict, sink: list[str]) -> subprocess.Popen:
    """Launch the pipeline and pump its output into sink from a worker thread.

    The process outlives this script run on purpose. Streamlit reruns the script
    on every interaction, and stops it when the viewer presses its own Stop
    button; a pipeline read inside the script body would be orphaned by that,
    still running with nothing holding a handle to it. Kept in session state
    instead, it can be shown and stopped by a later run.
    """
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "code" / "run_pipeline.py")],
        cwd=str(ROOT / "code"), env=environ,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)

    def pump() -> None:
        for line in proc.stdout:
            sink.append(line.rstrip())
        proc.wait()

    threading.Thread(target=pump, daemon=True).start()
    return proc


def stop_run(proc: subprocess.Popen) -> None:
    """Stop the pipeline and the stage it is running.

    Terminating the driver alone is not enough: it runs each stage as a child
    process, and that child would carry on holding the CPU.
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        family = parent.children(recursive=True) + [parent]
        for p in family:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(family, timeout=8)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except ImportError:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def draw_progress(running: bool) -> None:
    """Show how far the run has got, from the driver announcing each stage as it
    finishes rather than from elapsed time against an estimate."""
    stages = st.session_state.get("stages") or []
    log = st.session_state.get("run_log") or []
    proc = st.session_state.get("proc")
    if proc is None or not stages:
        return

    done = [l.strip()[2:].partition("finished in") for l in log
            if l.strip().startswith("->") and "finished in" in l]
    done = [(lbl.strip(), took.strip()) for lbl, _, took in done
            if lbl.strip() in stages][:len(stages)]
    at = len(done)
    where = stages[min(at, len(stages) - 1)]
    elapsed = (time.perf_counter() - st.session_state.started) / 60

    if running:
        st.progress(at / len(stages),
                    text=f"Step {min(at + 1, len(stages))} of {len(stages)}: "
                         f"{where}  ({elapsed:.1f} min so far)")
    for i, label in enumerate(stages):
        if i < at:
            st.markdown(f"- **{label}** — done in {done[i][1]}")
        elif i == at and running:
            st.markdown(f"- **{label}** — running")
        else:
            st.markdown(f"- {label}")

    if log:
        st.code(NL.join(log[-25:]))

    if not running:
        st.session_state.log = NL.join(log)
        if st.session_state.get("stopped"):
            st.warning(f"Stopped after {elapsed:.1f} min, during \"{where}\". "
                       f"Output written up to that point is kept.")
        elif proc.returncode == 0:
            st.success(f"Finished in {elapsed:.1f} min. See the Results tab.")
        else:
            st.error(f"Stopped during \"{where}\" with exit code "
                     f"{proc.returncode}, after {elapsed:.1f} min.")


@st.fragment(run_every=1.0)
def run_monitor() -> None:
    """Redraw the progress of a run once a second, on its own.

    A fragment rerun repaints only this block, so the page as a whole is not
    rerun and Streamlit's toolbar does not flash a running indicator between
    refreshes. When the run ends, one full rerun switches the page out of
    polling mode and re-enables the button.
    """
    draw_progress(running=True)
    proc = st.session_state.get("proc")
    if proc is None or proc.poll() is not None:
        st.rerun()


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def raw_of(path: Path) -> dict:
    return copy.deepcopy(load(path).raw)


def as_config(raw: dict, source: Path) -> Config:
    return Config(raw=raw, source=source)


def frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: r.get(c) for c in columns} for r in rows],
                        columns=columns)


def rows_of(df: pd.DataFrame) -> list[dict]:
    """Editor rows back to config entries, dropping blank rows and empty cells."""
    out = []
    for rec in df.to_dict(orient="records"):
        kept = {k: v for k, v in rec.items()
                if v is not None and v == v and v != ""}
        if kept:
            out.append(kept)
    return out


def data_folder(raw: dict) -> Path:
    p = Path(raw["paths"]["data"])
    return p if p.is_absolute() else ROOT / p


def csv_choices(folder: Path) -> list[str]:
    try:
        return sorted(p.name for p in folder.glob("*.csv"))
    except OSError:
        return []


def columns_of(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def file_picker(label: str, raw: dict, key: str, choices: list[str],
                help_text: str) -> None:
    """Choose one of the CSVs in the data folder, or type a name that is not
    there yet. Falls back to a free text box when the folder cannot be read."""
    current = raw["data_files"].get(key, "")
    if choices:
        options = choices if current in choices else ([current] + choices
                                                      if current else choices)
        raw["data_files"][key] = st.selectbox(
            label, options,
            index=options.index(current) if current in options else 0,
            help=help_text)
    else:
        raw["data_files"][key] = st.text_input(label, current, help=help_text)


def estimate(cfg: Config) -> list[tuple[str, float]]:
    """Rough seconds per part, from times measured on one desktop core: 1.5 s per
    regressor per forecast day, 4.6 s per consumer per day for ARIMA and SARIMAX,
    29 s per neural-network fit. Individual models vary by more than an order of
    magnitude around those averages, so a cheap selection finishes well inside."""
    n_sp = len(cfg.supply_point_ids)
    n_fold = len(cfg.folds)
    n_days = cfg.max_test_days or 30
    n_models = len(cfg.ml_subset) or 15
    out = []
    if cfg.runs("ml"):
        out.append(("machine learning", n_sp * n_fold * n_days * n_models * 1.5))
    if cfg.runs("arima"):
        out.append(("statistical", n_sp * n_fold * n_days * 4.6))
    if cfg.runs("dl"):
        out.append(("neural networks", n_sp * n_fold * 3 * 29.0))
    return out


def apply_thoroughness(raw: dict, level: str) -> list[str]:
    """Apply a preset, and return any model it had to add.

    A preset narrows the list of models. If a consumer has been pinned to a
    model outside that list, the scheduling stage would find no forecasts for
    it, so the pinned models are added back. Without this, picking a preset
    could leave the setup unrunnable through no fault of the person using it.
    """
    preset = THOROUGHNESS[level]
    raw["models"]["families"] = list(preset["families"])
    raw["models"]["max_test_days"] = preset["days"]

    subset = list(preset["subset"])
    added: list[str] = []
    if subset:
        pinned = dict.fromkeys(sp.get("deployed_model")
                               for sp in raw["supply_points"])
        added = [m for m in pinned if m and m not in subset]
        subset += added
    raw["models"]["ml_subset"] = subset
    return added


def set_single_period(raw: dict, month: int) -> None:
    """One test period, the chosen month, trained on everything before it."""
    year = int(raw["year"])
    last = calendar.monthrange(year, month)[1]
    prev_last = calendar.monthrange(year, month - 1)[1]
    raw["folds"] = [{
        "name": f"test_{month:02d}",
        "label": f"{MONTHS[month - 1]} {year}",
        "train_end": f"{month - 1:02d}-{prev_last:02d}",
        "test_start": f"{month:02d}-01",
        "test_end": f"{month:02d}-{last:02d}",
    }]


def drop_derived_state() -> None:
    """Forget the widgets whose contents were seeded from a configuration."""
    for key in ("ml_sel", "families_sel", "folds_df", "sp_df",
                "folds_editor", "sp_editor"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "raw" not in st.session_state:
    st.session_state.base = ROOT / "config" / "default.toml"
    st.session_state.raw = raw_of(st.session_state.base)
    st.session_state.log = ""
    st.session_state.output_dir = str(ROOT / "code")
    st.session_state.proc = None
    st.session_state.run_log = []
    st.session_state.stages = []
    st.session_state.stopped = False
    st.session_state.started = 0.0

raw = st.session_state.raw


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    if ICON.is_file():
        head, name = st.columns([1, 2], vertical_alignment="center")
        head.image(str(ASSETS / "icon-192.png"))
        name.title("RECpy")
    else:
        st.title("RECpy")
    st.caption("Demand forecasting and day-ahead battery scheduling for "
               "energy communities")

    def load_base(path: Path) -> None:
        st.session_state.base = path
        st.session_state.raw = raw_of(path)
        drop_derived_state()
        st.rerun()

    labels = [p.name for p in sorted((ROOT / "config").glob("*.toml"))]
    current = st.session_state.base.name
    choice = st.selectbox("Start from", labels,
                          index=labels.index(current) if current in labels else 0,
                          help="A shipped configuration to use as the starting "
                               "point. Your edits do not change the file unless "
                               "you save over it.")
    if choice != current:
        load_base(ROOT / "config" / choice)
    if st.button("Discard my changes", width="stretch"):
        load_base(st.session_state.base)

    st.session_state.output_dir = st.text_input(
        "Where to write results", st.session_state.output_dir,
        help="Give each configuration its own folder to keep runs apart.")

    st.divider()
    # Filled at the very end of the script. The sidebar is drawn before the tab
    # that edits the setup, so reporting here would describe the previous run's
    # state -- the period count would lag by one interaction.
    status_slot = st.empty()
    summary_slot = st.empty()


cfg = as_config(raw, st.session_state.base)

tab_cfg, tab_run, tab_res = st.tabs(["My community", "Run", "Results"])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
with tab_cfg:
    names = regressor_names()

    # ---- identity -------------------------------------------------------
    st.subheader("Community")
    c1, c2 = st.columns([2, 1])
    raw["name"] = c1.text_input(
        "Name", raw.get("name", ""),
        help="Used to label this setup. Any name will do.")
    raw["year"] = int(c2.number_input(
        "Year of your data", 1900, 2200, int(raw["year"]), 1,
        help="The calendar year the consumption files cover."))
    raw["description"] = st.text_area(
        "Notes", raw.get("description", ""), height=80,
        help="Free text, kept with the configuration.")

    # ---- where the data comes from --------------------------------------
    st.subheader("Data")
    st.caption("Everything is read from your machine and stays there.")
    raw["paths"]["data"] = st.text_input(
        "Folder holding your CSV files", str(raw["paths"]["data"]),
        help="Relative to the RECpy folder, or an absolute path.")

    folder = data_folder(raw)
    choices = csv_choices(folder)
    if not folder.is_dir():
        st.error(f"No such folder: {folder}")
    elif not choices:
        st.warning(f"No CSV files in {folder}. The synthetic example dataset is "
                   f"generated automatically if the configured files are absent.")
    else:
        st.caption(f"{len(choices)} CSV files found in {folder}")

    d1, d2, d3 = st.columns(3)
    with d1:
        file_picker("Electricity prices", raw, "prices", choices,
                    "Hourly import and export prices.")
    with d2:
        file_picker("Solar generation", raw, "solar", choices,
                    "Hourly generation from your panels, and the battery's "
                    "own record where you have one.")
    with d3:
        file_picker("Weather and calendar", raw, "meteo_calendar", choices,
                    "Hourly weather and calendar features, with every "
                    "consumer's consumption alongside them.")

    meteo = folder / raw["data_files"].get("meteo_calendar", "")
    meteo_cols = columns_of(meteo)
    if meteo_cols:
        known = [c for c in raw.get("data", {}).get("consumption_columns", [])
                 if c in meteo_cols]
        raw.setdefault("data", {})["consumption_columns"] = st.multiselect(
            "Which of its columns are consumption", meteo_cols,
            default=known or [c for c in meteo_cols if c.startswith("C_")],
            help="Every consumer's column in that file, including any you are "
                 "not analysing. The forecasting models have to set the others "
                 "aside, so leaving one out would train on the wrong meter.")
    elif raw["data_files"].get("meteo_calendar"):
        st.caption("Column names will be read from that file once it exists.")

    with st.expander("Column names inside the files"):
        st.caption("Set these to match your own files rather than renaming your "
                   "columns. The defaults are the names this study's data uses.")
        cols = raw.setdefault("columns", {})
        k1, k2 = st.columns(2)
        cols["timestamp_consumption"] = k1.text_input(
            "Timestamp, consumption and weather files",
            cols.get("timestamp_consumption", "date_time"))
        cols["timestamp_market"] = k2.text_input(
            "Timestamp, price and generation files",
            cols.get("timestamp_market", "datetime"))
        k1, k2 = st.columns(2)
        cols["import_price"] = k1.text_input(
            "Price paid for energy taken from the grid",
            cols.get("import_price", "PVPC"))
        cols["export_price"] = k2.text_input(
            "Price received for energy exported",
            cols.get("export_price", "Compensation"))
        gen = cols.get("generation", [])
        cols["generation"] = [c.strip() for c in st.text_input(
            "Generation columns, comma separated",
            ", ".join(gen) if isinstance(gen, list) else str(gen),
            help="Summed to give total production. One entry per inverter or "
                 "array.").split(",") if c.strip()]
        cols["battery_power"] = st.text_input(
            "Metered battery flow", cols.get("battery_power", "Battery Power"),
            help="Positive when charging. Used for the baseline that reflects "
                 "how the battery is run today.")

    if st.button("Check my data"):
        from check_data import check
        findings = check(as_config(raw, st.session_state.base))
        errs = [f for f in findings if f.level == "error"]
        warns = [f for f in findings if f.level == "warning"]
        if errs:
            st.error(f"{len(errs)} problem(s) would stop a run:")
            for f in errs:
                st.caption(f"- **{f.where}** — {f.message}")
        if warns:
            st.warning(f"{len(warns)} thing(s) worth knowing:")
            for f in warns:
                st.caption(f"- **{f.where}** — {f.message}")
        if not errs:
            st.success("The data is the shape the pipeline expects.")
        with st.expander("What was checked"):
            for f in findings:
                if f.level == "ok":
                    st.caption(f"- **{f.where}** — {f.message}")

    # ---- consumers -------------------------------------------------------
    st.subheader("Consumers")
    st.caption("One row per metered consumer. Each needs a CSV of its hourly "
               "consumption and the name of the column to read from it.")
    sp_cols = ["id", "history_csv", "column", "deployed_model"]
    if "sp_df" not in st.session_state:
        st.session_state.sp_df = frame(raw["supply_points"], sp_cols)

    show_model_column = st.toggle(
        "Choose each consumer's model myself", key="pick_models",
        help="Off, the model is whichever wins the comparison on your own data, "
             "for each consumer separately. That is the method the article "
             "describes, and the better default.")
    sp_config = {
        "id": st.column_config.TextColumn("Name"),
        "history_csv": (st.column_config.SelectboxColumn(
            "Consumption file", options=choices) if choices
            else st.column_config.TextColumn("Consumption file")),
        "column": (st.column_config.SelectboxColumn(
            "Column to read", options=meteo_cols) if meteo_cols
            else st.column_config.TextColumn("Column to read")),
        "deployed_model": st.column_config.SelectboxColumn(
            "Model used for scheduling", options=names,
            help="Leave blank to use whichever model wins the comparison."),
    }
    rows = rows_of(st.data_editor(
        st.session_state.sp_df, num_rows="dynamic", width="stretch",
        key="sp_editor", column_config=sp_config,
        column_order=sp_cols if show_model_column else sp_cols[:3]))
    if not show_model_column:
        # Nobody has asked to pin a model, so carry none: the comparison picks
        # each consumer's winner and the scheduling stage uses it. The choices
        # stay in the table and come back when the column is shown again.
        for sp in rows:
            sp.pop("deployed_model", None)
    raw["supply_points"] = rows
    if show_model_column and any(sp.get("deployed_model") for sp in rows):
        st.caption("Any model pinned here is added to the comparison, whatever "
                   "the effort preset below selects, so its forecasts exist.")

    # ---- battery and grid -------------------------------------------------
    st.subheader("Battery and grid connection")
    b = raw["battery"]
    c1, c2, c3 = st.columns(3)
    b["capacity_kwh"] = c1.number_input("Battery capacity (kWh)", 0.0, 1e5,
                                        float(b["capacity_kwh"]), 0.01)
    b["power_limit_kw"] = c2.number_input("Charge/discharge limit (kW)", 0.0,
                                          1e5, float(b["power_limit_kw"]), 0.01)
    raw["grid"]["export_limit_kw"] = c3.number_input(
        "Export limit to the grid (kW)", 0.0, 1e5,
        float(raw["grid"]["export_limit_kw"]), 0.01)
    c1, c2, c3 = st.columns(3)
    b["min_soc_kwh"] = c1.number_input(
        "Reserve never used (kWh)", 0.0, 1e5, float(b["min_soc_kwh"]), 0.01,
        help="The schedule keeps at least this much charge in the battery.")
    b["efficiency"] = c2.number_input(
        "Round-trip efficiency", 0.0, 1.0, float(b["efficiency"]), 0.01,
        help="Fraction of stored energy you get back out.")
    b["degradation_eur_per_kwh"] = c3.number_input(
        "Wear cost (EUR per kWh cycled)", 0.0, 10.0,
        float(b["degradation_eur_per_kwh"]), 0.001, format="%.3f",
        help="Priced into the schedule, so it does not cycle the battery for "
             "savings smaller than the wear it causes.")
    with st.expander("More battery settings"):
        b["initial_soc_fraction"] = st.number_input(
            "Starting charge (fraction of capacity)", 0.0, 1.0,
            float(b["initial_soc_fraction"]), 0.01,
            help="How full the battery is assumed to be when the first day "
                 "starts.")

    # ---- what to run ------------------------------------------------------
    st.subheader("Effort")

    # Off, the two questions below settle which models run and over what period.
    # On, they are chosen outright. The two are mutually exclusive on purpose:
    # a preset that silently overwrote a hand-picked selection, or a selection
    # that quietly contradicted the preset, is worse than one or the other.
    detailed = st.toggle(
        "Choose the models and periods myself", key="detailed",
        help="Off, a preset settles both from how thorough you want to be.")

    if not detailed:
        months = [f["test_start"][:2] for f in raw["folds"]]
        default_month = int(months[0]) if months and months[0].isdigit() else 4
        was_many = len(raw["folds"]) > 1
        month = st.selectbox(
            "Period to test on", range(2, 13),
            index=max(0, min(10, default_month - 2)),
            format_func=lambda m: f"{MONTHS[m - 1]} {raw['year']}",
            help="The models train on everything before this month, then "
                 "forecast day by day through it. January is not offered: "
                 "there would be no history to train on.")
        set_single_period(raw, month)
        if was_many:
            st.caption(f"The setup you started from tested on {len(months)} "
                       f"periods across the year; this uses one. Turn on "
                       f"\"Choose the models and periods myself\" to keep them "
                       f"all, or \"Discard my changes\" to get them back.")

        level = st.radio(
            "How hard to look for each consumer's best model",
            list(THOROUGHNESS), horizontal=True,
            index=list(THOROUGHNESS).index(st.session_state.get("level",
                                                                "Balanced")),
            key="level",
            help="Every consumer gets its own model, chosen by comparing the "
                 "candidates on your data. More candidates take longer and "
                 "give a better-founded choice.")
        added = apply_thoroughness(raw, level)
        note = THOROUGHNESS[level]["blurb"]
        if added:
            note += (" Plus " + ", ".join(added) +
                     ", kept because a consumer is pinned to it.")
        st.caption(note)
    else:
        c1, c2 = st.columns([2, 1])
        if "families_sel" not in st.session_state:
            st.session_state.families_sel = list(raw["models"]["families"])
        c1.multiselect("Kinds of model", list(Config.KNOWN_FAMILIES),
                       key="families_sel",
                       format_func=lambda f: FAMILY_LABELS.get(f, f),
                       help="Each kind is a separate stage of the run.")
        raw["models"]["families"] = [f for f in Config.KNOWN_FAMILIES
                                     if f in st.session_state.families_sel]
        raw["models"]["max_test_days"] = int(c2.number_input(
            "Days to forecast per period (0 = all of them)", 0, 366,
            int(raw["models"].get("max_test_days", 0)), 1))

        if "ml_sel" not in st.session_state:
            configured = [n for n in raw["models"].get("ml_subset", [])
                          if n in names]
            st.session_state.ml_sel = configured or list(names)
        st.multiselect(
            f"Which of the {len(names)} machine-learning models to try", names,
            key="ml_sel",
            help="All of them is the published setting. Unselecting some speeds "
                 "a run up considerably.")
        runs_ml = "ml" in raw["models"]["families"]
        if not runs_ml:
            st.caption("Not used: machine learning is unselected above.")
        ordered = [n for n in names if n in st.session_state.ml_sel]
        st.session_state.no_regressors = runs_ml and not ordered
        if ordered:
            raw["models"]["ml_subset"] = ([] if len(ordered) == len(names)
                                          else ordered)
        elif runs_ml:
            st.warning("Select at least one machine-learning model, or unselect "
                       f"\"{FAMILY_LABELS['ml']}\" above.")

        st.markdown("**Periods to test on**")
        st.caption("One row per period, called a fold in the method. Dates are "
                   "month-day; the year comes from the field above. The label "
                   "is written into the result files, so changing it changes "
                   "those keys.")
        if "folds_df" not in st.session_state:
            st.session_state.folds_df = frame(
                raw["folds"], ["name", "label", "train_end",
                               "test_start", "test_end"])
        raw["folds"] = rows_of(st.data_editor(
            st.session_state.folds_df, num_rows="dynamic", width="stretch",
            key="folds_editor",
            column_config={
                "name": st.column_config.TextColumn("Identifier"),
                "label": st.column_config.TextColumn("Label in results"),
                "train_end": st.column_config.TextColumn("Training ends (MM-DD)"),
                "test_start": st.column_config.TextColumn(
                    "Forecasting from (MM-DD)"),
                "test_end": st.column_config.TextColumn(
                    "Forecasting to (MM-DD)")}))

    with st.expander("Forecasting details"):
        f = raw["forecasting"]
        c1, c2, c3, c4 = st.columns(4)
        f["train_start"] = c1.text_input(
            "History starts (MM-DD)", str(f["train_start"]),
            help="First day of consumption history used for training.")
        f["lookback_hours"] = int(c2.number_input(
            "Hours of history per forecast", 1, 10000,
            int(f["lookback_hours"]), 24,
            help="How far back each forecast looks. 192 hours is eight days."))
        f["horizon_hours"] = int(c3.number_input(
            "Hours forecast at a time", 1, 168, int(f["horizon_hours"]), 1,
            help="24 for day-ahead, which is what the battery schedule needs."))
        f["random_seed"] = int(c4.number_input(
            "Random seed", 0, 10 ** 9, int(f["random_seed"]), 1,
            help="Fixed so a rerun reproduces the same models."))

    # ---- save -------------------------------------------------------------
    st.divider()
    # Bottom-aligned: the text input carries a label above its box, so a button
    # in a plain column would sit level with the label instead of the box.
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    target = c1.text_input("Save this setup as",
                           str(ROOT / "config" / "custom.toml"),
                           help="A TOML file you can run from the command line "
                                "with RECPY_CONFIG, or reopen here.")
    if c2.button("Save", type="primary", width="stretch"):
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tomli_w.dumps(raw).encode("utf-8"))
        st.success(f"Written to {path}")

    with st.expander("The settings as the pipeline sees them"):
        st.code(as_config(raw, st.session_state.base).summary())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
with tab_run:
    cfg = as_config(raw, st.session_state.base)
    problems = cfg.validate()

    st.subheader("Run")
    st.caption("Each consumer's demand is forecast day by day, the models are "
               "compared, the winner's forecasts are used to schedule the "
               "battery against the hourly prices, and the result is costed "
               "against what actually happened.")

    est = estimate(cfg)
    if est:
        detail = ", ".join(f"{k} {v / 60:.0f} min" for k, v in est)
        st.info(f"Expect up to about {sum(v for _, v in est) / 60:.0f} minutes "
                f"({detail}) on one processor core. Cheaper models finish well "
                f"inside this.")

    proc = st.session_state.get("proc")
    busy = proc is not None and proc.poll() is None
    blocked = st.session_state.get("no_regressors") or problems

    if st.session_state.get("no_regressors"):
        st.error("No machine-learning models are selected.")
    elif problems:
        st.error("Not ready to run:")
        for problem in problems:
            st.caption(f"- {problem}")

    go, halt, _ = st.columns([1, 1, 4])
    start = go.button("Run experiment", type="primary", width="stretch",
                      disabled=bool(blocked) or busy)
    if busy and halt.button("Stop", width="stretch"):
        stop_run(proc)
        st.session_state.stopped = True
        st.rerun()

    if start:
        run_cfg = Path(st.session_state.output_dir) / "run_config.toml"
        run_cfg.parent.mkdir(parents=True, exist_ok=True)
        run_cfg.write_bytes(tomli_w.dumps(raw).encode("utf-8"))

        environ = dict(os.environ)
        environ["RECPY_CONFIG"] = str(run_cfg)
        environ["RECPY_OUTPUT_DIR"] = st.session_state.output_dir
        environ["PYTHONIOENCODING"] = "utf-8"

        # Frozen now, so editing the setup mid-run cannot change the bar.
        st.session_state.stages = [label for _, label, family in pipeline_stages()
                                   if not family or cfg.runs(family)]
        st.session_state.run_log = []
        st.session_state.stopped = False
        st.session_state.started = time.perf_counter()
        st.session_state.proc = start_run(environ, st.session_state.run_log)
        st.rerun()

    if busy:
        # Refreshing this block on its own, rather than rerunning the whole
        # script on a timer: a full rerun every second makes Streamlit's own
        # running indicator and Stop label blink in the toolbar.
        run_monitor()
    elif proc is not None:
        draw_progress(running=False)

    if st.session_state.log and not busy:
        with st.expander("Full log of the last run"):
            st.code(st.session_state.log)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def euro_column(df: pd.DataFrame, prefix: str) -> str | None:
    for c in df.columns:
        if c.startswith(prefix):
            return c
    return None


def show_savings(opt: Path) -> bool:
    summary = opt / "optimization_cv_summary.csv"
    if not summary.is_file():
        return False
    s = pd.read_csv(summary)
    total = s[s[s.columns[0]] == "TOTAL"]
    row = total.iloc[0] if not total.empty else s.iloc[-1]
    vs_b2 = euro_column(s, "Saving vs B2")
    vs_b1 = euro_column(s, "Saving vs B1")
    days = int(row["Days"]) if "Days" in s.columns else None

    c1, c2, c3 = st.columns(3)
    if vs_b2:
        c1.metric("Saved against your current operation",
                  f"{row[vs_b2]:,.2f} EUR",
                  help="Compared with running the battery the way it is run "
                       "today, over the same days.")
    if vs_b1:
        c2.metric("Saved against no battery at all", f"{row[vs_b1]:,.2f} EUR")
    if days:
        c3.metric("Days scheduled", days)

    body = s[s[s.columns[0]] != "TOTAL"]
    if len(body) > 1 and vs_b2:
        st.bar_chart(body.set_index(body.columns[0])[[vs_b2]],
                     x_label="Period", y_label="Saving (EUR)")
    with st.expander("The full cost breakdown"):
        st.dataframe(s, width="stretch", hide_index=True)
    return True


def show_battery_plan(opt: Path) -> bool:
    hourly = opt / "optimization_cv_hourly.csv"
    if not hourly.is_file():
        return False
    h = pd.read_csv(hourly)
    if "Day" not in h.columns or h.empty:
        return False
    days = sorted(h["Day"].astype(str).unique())
    day = st.selectbox("Day", days, index=len(days) - 1)
    d = h[h["Day"].astype(str) == day].set_index("Hour")

    soc = [c for c in d.columns if c.startswith("Battery SoC")]
    flows = [c for c in d.columns
             if c.startswith(("Demand", "Solar [", "Grid import"))]
    price = [c for c in d.columns if c.startswith("Import price")]

    if soc:
        st.caption("Charge held in the battery through the day")
        st.line_chart(d[soc], x_label="Hour of day",
                      y_label="Energy stored (kWh)")
    if flows:
        st.caption("Demand, generation and what is drawn from the grid")
        st.line_chart(d[flows], x_label="Hour of day",
                      y_label="Energy per hour (kWh)")
    if price:
        st.caption("Price being paid, for comparison with the charging "
                   "pattern")
        st.line_chart(d[price], x_label="Hour of day",
                      y_label="Price (EUR/kWh)")
    with st.expander("Hour by hour"):
        st.dataframe(d, width="stretch")
    return True


def show_accuracy(cv: Path) -> bool:
    preds = sorted(cv.glob("*/Predictions_*.csv"))
    if not preds:
        return False
    pick = st.selectbox("Consumer", [p.name for p in preds])
    d = pd.read_csv(next(p for p in preds if p.name == pick),
                    index_col=0, parse_dates=True)
    models = [c for c in d.columns if c != "Observed"]
    err = pd.DataFrame(
        {"Average error (kWh)": [(d["Observed"] - d[m]).abs().mean()
                                 for m in models]},
        index=models).sort_values("Average error (kWh)")
    best = err.index[0]
    st.caption(f"Closest for this consumer: **{best}**")
    show = st.multiselect("Compare against what actually happened", models,
                          default=[best])
    st.line_chart(d[["Observed"] + show], x_label="Date and time",
                  y_label="Consumption (kWh)")
    st.dataframe(err, width="stretch")
    return True


def show_model_selection(cv: Path) -> bool:
    agg = cv / "wilcoxon_cv_aggregated.csv"
    if not agg.is_file():
        return False
    a = pd.read_csv(agg)
    best = (a.sort_values("mean_net_wins", ascending=False)
             .groupby("supply_point").head(1)
             [["supply_point", "model", "mean_net_wins", "mean_MAE"]])
    st.caption("Winner of the Wilcoxon net-wins tournament per consumer. With "
               "few test days the ranking is not conclusive.")
    st.dataframe(best, width="stretch", hide_index=True)
    with st.expander("Full ranking"):
        st.dataframe(a, width="stretch", hide_index=True)
    return True


def show_figures(cv: Path) -> bool:
    figs = sorted((cv / "figures").glob("*.png"))
    if not figs:
        return False
    for i in range(0, len(figs), 2):
        for col, fig in zip(st.columns(2), figs[i:i + 2]):
            col.image(str(fig), caption=fig.stem, width="stretch")
    return True


with tab_res:
    out = Path(st.session_state.output_dir)
    cv = out / "cross_validation"
    opt = out / "optimization_results"

    if not cv.exists() and not opt.exists():
        st.info(f"No results under {out} yet. Run the experiment first.")
    else:
        # Ordered by what someone running a community wants to know first. The
        # model comparison and the article's visualizations are the working
        # so they are here but folded away.
        # The headings are deliberately short; the captions carry the tense.
        # These are results on days that have already happened, scored against
        # what the community actually did, not a projection of future savings.
        st.subheader("Savings")
        st.caption("What the optimised schedule would have saved over the days "
                   "tested, costed against what actually happened.")
        if not show_savings(opt):
            st.caption("No scheduling results yet.")

        st.subheader("Battery behaviour")
        st.caption("The schedule the optimiser would have followed on the "
                   "chosen day, hour by hour.")
        if not show_battery_plan(opt):
            st.caption("No hourly schedule yet.")

        st.subheader("Forecast accuracy")
        if not show_accuracy(cv):
            st.caption("No forecasts yet.")

        with st.expander("How each consumer's model was chosen"):
            if not show_model_selection(cv):
                st.caption("No model comparison yet.")
        with st.expander("Visualizations from the article"):
            if not show_figures(cv):
                st.caption("No visualizations yet.")


# ---------------------------------------------------------------------------
# Sidebar status, filled last so it describes the setup as it now stands
# ---------------------------------------------------------------------------
final = as_config(raw, st.session_state.base)
final_problems = final.validate()
with status_slot.container():
    if final_problems:
        st.error("Not ready to run:")
        for problem in final_problems:
            st.caption(f"- {problem}")
    else:
        st.success("Ready to run")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


summary_slot.caption(", ".join([
    _plural(len(final.supply_point_ids), "consumer"),
    _plural(len(final.folds), "period"),
]))
