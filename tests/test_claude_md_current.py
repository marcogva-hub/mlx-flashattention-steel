"""CLAUDE.md "Current status" tracks the package version (2.63.0).

The section said "v2.61.0 (held/unpublished)" through 2.62.0 … 2.62.3 — nothing tied it to the
version bump.  This lock makes every bump update it: the first line of the section must start
with ``v<pyproject version>``.  CLAUDE.md is an internal doc excluded from the sdist, so the
lock skips where it is absent.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = ROOT / "CLAUDE.md"


def _pyproject_version() -> str:
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert m, "no version in pyproject.toml"
    return m.group(1)


@pytest.mark.skipif(not CLAUDE_MD.exists(), reason="CLAUDE.md is not shipped in the sdist")
def test_current_status_names_the_package_version():
    text = CLAUDE_MD.read_text(encoding="utf-8")
    section = text.split("\n## Current status\n", 1)
    assert len(section) == 2, "CLAUDE.md has no '## Current status' section"
    first = next(l for l in section[1].splitlines() if l.strip())
    version = _pyproject_version()
    assert first.startswith(f"v{version} "), (
        f"CLAUDE.md 'Current status' starts with {first[:40]!r}; update it with the version "
        f"bump (pyproject {version})")
