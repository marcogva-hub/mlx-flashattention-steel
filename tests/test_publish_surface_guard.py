"""Publish-surface guard (audit Phase D) — the journal must not leak into the wheel/sdist.

Marco's publication policy: the published wheel/sdist ships ONLY changelog +
current-state docs (README) + verified code + tests + LICENSE. Devlogs / plans /
analyses / phase-reports / campaign docs are RETAINED in-repo (and in the gitignored
`.doc-archive/`) but NEVER published.

This guard enforces it mechanically: the sdist `MANIFEST.in` is an include-WHITELIST,
and it must NOT whitelist any journal path (devnotes/, docs/, .doc-archive/), and the
wheel must ship only the `mlx_mfa` package. A change that would package the journal
fails CI. (The self-test demonstrates the check catches a planted leak.)
"""
from __future__ import annotations

import re
from pathlib import Path
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST = _ROOT / "MANIFEST.in"
_PYPROJECT = _ROOT / "pyproject.toml"

# Paths that are RETAIN-class (journal) — must never be in the publish surface.
_FORBIDDEN = ("devnotes", "docs", ".doc-archive")
# include directives that ADD files to the sdist.
_INCLUDE_DIRECTIVES = ("include", "recursive-include", "graft")


def _manifest_include_lines():
    lines = []
    for raw in _MANIFEST.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        verb = s.split()[0] if s.split() else ""
        if verb in _INCLUDE_DIRECTIVES:
            lines.append(s)
    return lines


def _leaks(include_lines):
    """Return include lines that would package a forbidden journal path."""
    bad = []
    for line in include_lines:
        # the path target(s) start after the verb
        target = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
        first = target.split()[0] if target.split() else target  # path is the first token
        if any(first == f or first.startswith(f + "/") or first.startswith(f + "\\")
               for f in _FORBIDDEN):
            bad.append(line)
    return bad


def test_manifest_does_not_package_the_journal():
    """MANIFEST.in (include-whitelist) must not whitelist devnotes/docs/.doc-archive."""
    bad = _leaks(_manifest_include_lines())
    assert not bad, f"MANIFEST.in would PUBLISH journal paths (RETAIN-class): {bad}"


def test_manifest_publishes_the_expected_current_state_docs():
    """The published current-state docs (README + CHANGELOG + LICENSE) MUST be whitelisted."""
    text = _MANIFEST.read_text(encoding="utf-8")
    for required in ("README.md", "CHANGELOG.md", "LICENSE"):
        assert required in text, f"{required} missing from MANIFEST.in publish whitelist"


def test_wheel_ships_only_the_package():
    """The wheel must ship only the mlx_mfa package (no docs/devnotes in the wheel)."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    assert 'wheel.packages = ["mlx_mfa"]' in text, "wheel.packages changed — re-verify no journal in the wheel"


def test_guard_catches_a_planted_leak():
    """Self-test: the leak detector trips on a planted forbidden include (demonstration)."""
    planted = [
        "recursive-include devnotes *.md",
        "graft docs/v50/campaign-2026-06",
        "include docs/v6-nax/sparse-bug-investigation.md",
    ]
    bad = _leaks(planted)
    assert len(bad) == 3, f"guard FAILED to catch planted journal leaks: caught {bad}"
    # and a legitimate include must NOT trip it
    assert _leaks(["include README.md", "recursive-include mlx_mfa *.py",
                   "recursive-include csrc *.cpp"]) == []
