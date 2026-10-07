"""Rebuild docs/changelog.md from the real CHANGELOG.md.

Keeping the changelog in one place: the site shows a copy, generated here and by
the documentation workflow, so the page cannot drift from the file people
actually edit.

    python docs/build_changelog.py
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "CHANGELOG.md"
TARGET = ROOT / "docs" / "changelog.md"

HEADER = """# Changelog

!!! info "This page is generated"
    It is copied from `CHANGELOG.md` at the repository root when the site is
    built, so the two cannot drift. Edit that file, not this one.

"""

body = SOURCE.read_text(encoding="utf-8")
body = body.split("\n", 1)[1].lstrip("\n")      # drop its own "# Changelog"
TARGET.write_text(HEADER + body, encoding="utf-8")
print(f"{TARGET.relative_to(ROOT)} rebuilt from {SOURCE.name} "
      f"({len(body.splitlines())} lines)")
