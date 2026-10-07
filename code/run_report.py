"""
Provenance for a set of results.

A results directory used to say nothing about what produced it: not which
configuration, which version of the code, or which package versions. Handed one
on its own, you could not tell whether it came from the published configuration
or from somebody's experiment. The driver now writes a manifest beside the
outputs, and the stages record any model that failed, so a directory describes
itself.

    from run_report import record_failure
    record_failure(CONFIG, stage="ml", entity="SP3", model="MLP", error=exc)
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from recpy_config import CONFIG, Config, ROOT

MANIFEST = "run_manifest.json"
FAILURES = "model_failures.csv"

#: Recorded in the manifest. The pinned ones are what reproducibility rests on.
TRACKED = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm",
           "statsmodels", "tensorflow", "keras-tcn", "pyomo", "highspy",
           "matplotlib", "streamlit")


def _git(*args: str) -> str | None:
    """Ask git something about this checkout, or None if it cannot answer."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_version() -> dict:
    """Which code produced this, as precisely as the checkout allows."""
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    return {
        "commit": commit,
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_changes": bool(dirty) if dirty is not None else None,
    }


def package_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in TRACKED:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def record_failure(cfg: Config, stage: str, entity: str, model: str,
                   error: BaseException | str) -> None:
    """Note that one model failed, where a later reader will see it.

    A stage that catches an exception per model and carries on is the right
    behaviour -- one failure should not waste a run of many hours -- but it
    leaves that model's column absent from the predictions, and the comparison
    downstream then ranks fewer models with nothing to say why. This puts the
    reason somewhere that outlives the console.
    """
    path = cfg.output_root / FAILURES
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["when", "stage", "entity", "model", "error"])
        writer.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         stage, entity, model,
                         f"{type(error).__name__}: {error}"
                         if isinstance(error, BaseException) else str(error)])


def failures(cfg: Config = CONFIG) -> list[dict]:
    path = cfg.output_root / FAILURES
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_manifest(cfg: Config, started: datetime, finished: datetime,
                   stages: list[tuple[str, float]], returncode: int) -> Path:
    """Write the manifest beside the results and return where it went."""
    manifest = {
        "software": {
            "name": "RECpy",
            "repository": "https://github.com/jmolleda/RECpy",
            **code_version(),
        },
        "run": {
            "started": started.isoformat(timespec="seconds"),
            "finished": finished.isoformat(timespec="seconds"),
            "duration_seconds": round((finished - started).total_seconds(), 1),
            "exit_code": returncode,
            "stages": [{"name": name, "seconds": round(secs, 1)}
                       for name, secs in stages],
        },
        "configuration": {
            "source": str(cfg.source),
            "name": cfg.name,
            "year": cfg.year,
            "supply_points": cfg.supply_point_ids,
            "folds": [f["name"] for f in cfg.folds],
            "model_families": cfg.model_families,
            "ml_subset": cfg.ml_subset or "all",
            "max_test_days": cfg.max_test_days or "whole period",
            "random_seed": cfg.random_seed,
            "settings": cfg.raw,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.machine(),
            "packages": package_versions(),
        },
        "outputs": {
            "root": str(cfg.output_root),
            "model_failures": len(failures(cfg)),
        },
    }
    path = cfg.output_root / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                               default=str) + "\n", encoding="utf-8")
    return path
