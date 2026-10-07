"""Provenance: the manifest, and the record of models that failed.

A model that fails is caught and the run carries on, which is right for a run of
many hours. These tests are about the consequence: the failure has to leave a
trace, or the comparison downstream quietly ranks fewer models with nothing to
say why.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from run_report import (code_version, failures, package_versions,
                        record_failure, write_manifest)


@pytest.fixture
def cfg(default_cfg, cfg_with, tmp_path, monkeypatch):
    monkeypatch.setenv("RECPY_OUTPUT_DIR", str(tmp_path))
    return cfg_with()


def test_nothing_is_recorded_until_something_fails(cfg):
    assert failures(cfg) == []
    assert not (cfg.output_root / "model_failures.csv").exists()


def test_a_failure_is_written_with_its_reason(cfg):
    record_failure(cfg, "machine learning", "SP3", "MLP",
                   ValueError("did not converge"))
    rows = failures(cfg)
    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "machine learning"
    assert row["entity"] == "SP3"
    assert row["model"] == "MLP"
    assert "ValueError" in row["error"] and "did not converge" in row["error"]
    assert row["when"], "the time it happened is recorded"


def test_failures_accumulate_across_stages(cfg):
    record_failure(cfg, "machine learning", "SP1", "MLP", RuntimeError("a"))
    record_failure(cfg, "deep learning", "SP2", "LSTM_single", RuntimeError("b"))
    record_failure(cfg, "statistical", "SP3", "SARIMAX", "no convergence")
    rows = failures(cfg)
    assert [r["stage"] for r in rows] == ["machine learning", "deep learning",
                                          "statistical"]
    assert rows[2]["error"] == "no convergence", "a plain string is accepted too"


def test_the_manifest_describes_the_run(cfg):
    started = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    finished = started + timedelta(minutes=3, seconds=30)
    path = write_manifest(cfg, started, finished,
                          [("Machine-learning forecasts", 180.4),
                           ("Day-ahead scheduling", 29.6)], 0)
    m = json.loads(path.read_text(encoding="utf-8"))

    assert m["run"]["duration_seconds"] == 210.0
    assert [s["name"] for s in m["run"]["stages"]] == [
        "Machine-learning forecasts", "Day-ahead scheduling"]
    assert m["run"]["exit_code"] == 0

    assert m["configuration"]["random_seed"] == cfg.random_seed
    assert m["configuration"]["supply_points"] == cfg.supply_point_ids
    assert m["configuration"]["settings"], "the whole configuration is embedded"

    assert m["environment"]["python"]
    assert m["environment"]["packages"]["pandas"], "pinned versions recorded"
    assert m["software"]["name"] == "RECpy"


def test_the_manifest_counts_failures(cfg):
    record_failure(cfg, "machine learning", "SP1", "MLP", RuntimeError("x"))
    now = datetime.now(timezone.utc)
    path = write_manifest(cfg, now, now, [], 0)
    assert json.loads(path.read_text(encoding="utf-8"))["outputs"]["model_failures"] == 1


def test_the_manifest_lands_beside_the_results(cfg, tmp_path):
    now = datetime.now(timezone.utc)
    assert write_manifest(cfg, now, now, [], 0).parent == tmp_path


def test_code_version_reports_this_checkout():
    v = code_version()
    assert set(v) == {"commit", "describe", "branch", "uncommitted_changes"}
    if v["commit"] is not None:          # absent in an exported archive
        assert len(v["commit"]) == 40


def test_package_versions_cover_the_pinned_dependencies():
    versions = package_versions()
    for name in ("numpy", "pandas", "scikit-learn", "pyomo"):
        assert versions[name], f"{name} version not recorded"


def test_a_manifest_is_json_even_with_awkward_values(cfg):
    """The configuration is embedded wholesale, so it must survive whatever a
    user put in it."""
    cfg.raw["notes"] = {"path": cfg.data_dir, "when": datetime.now(timezone.utc)}
    now = datetime.now(timezone.utc)
    path = write_manifest(cfg, now, now, [], 0)
    json.loads(path.read_text(encoding="utf-8"))   # must not raise
