"""The documentation must keep up with the configuration.

A reference page that silently falls behind is worse than none: someone adapting
the framework trusts it. These tests fail when a setting exists but is not
described, or is described but no longer exists.
"""
from __future__ import annotations

import pathlib
import re
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REFERENCE = DOCS / "configuration.md"

#: Per-entity tables are documented by their shape, not one page per consumer.
BY_SHAPE = {"arima", "dl", "folds", "supply_points"}


def leaf_keys(node, prefix=""):
    """Every setting a user could write, as a dotted path."""
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from leaf_keys(value, path + ".")
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            for sub in sorted(value[0]):
                yield f"{path}[].{sub}"
        else:
            yield path


@pytest.fixture(scope="module")
def reference() -> str:
    return REFERENCE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def default_raw() -> dict:
    return tomllib.loads((ROOT / "config" / "default.toml").read_text(
        encoding="utf-8"))


def test_every_setting_is_documented(default_raw, reference):
    """Each key in the shipped configuration appears in the reference."""
    undocumented = []
    for path in leaf_keys(default_raw):
        # The reference names some keys in full (`data.consumption_columns`)
        # and others by their leaf inside a section's table (`output`). Either
        # is fine; what matters is that the setting is described somewhere.
        leaf = path.split(".")[-1].split("[].")[-1]
        if f"`{leaf}`" not in reference and f"`{path}`" not in reference:
            undocumented.append(path)
    assert not undocumented, (
        "settings with no entry in docs/configuration.md: "
        + ", ".join(sorted(set(undocumented))))


def test_the_reference_documents_nothing_imaginary(default_raw, reference):
    """Every top-level section the reference describes really exists."""
    sections = set(re.findall(r"^## `\[(\w+)[\].]", reference, re.M))
    real = set(default_raw) | {"folds", "supply_points"}
    assert sections <= real, f"documented but absent: {sections - real}"


def test_every_page_in_the_nav_exists():
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    nav = config[config.index("nav:"):].split("exclude_docs")[0]
    pages = re.findall(r":\s*([\w-]+\.md)\s*$", nav, re.M)
    assert len(pages) >= 8, pages
    missing = [p for p in pages if not (DOCS / p).is_file()]
    assert not missing, missing


def test_the_changelog_page_matches_the_changelog():
    """docs/changelog.md is generated; a difference means it was not rebuilt."""
    source = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    page = (DOCS / "changelog.md").read_text(encoding="utf-8")
    body = source.split("\n", 1)[1].lstrip("\n")
    assert page.endswith(body), (
        "docs/changelog.md is stale; run python docs/build_changelog.py")


@pytest.mark.parametrize("claim, path", [
    ("config/quick.toml", "config/quick.toml"),
    ("config/default.toml", "config/default.toml"),
    ("code/check_data.py", "code/check_data.py"),
    ("code/run_pipeline.py", "code/run_pipeline.py"),
    ("app/recpy_app.py", "app/recpy_app.py"),
    ("requirements-gui.txt", "requirements-gui.txt"),
    ("requirements-docs.txt", "requirements-docs.txt"),
])
def test_files_the_documentation_points_at_exist(claim, path):
    mentioned = any(claim in (DOCS / p).read_text(encoding="utf-8")
                    for p in ("index.md", "getting-started.md",
                              "your-own-community.md", "configuration.md",
                              "development.md", "results.md",
                              "reproducibility.md", "pipeline.md"))
    if mentioned:
        assert (ROOT / path).exists(), f"documented but missing: {path}"
