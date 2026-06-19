"""Doc-accuracy guards — kill the recurring stale-doc class mechanically.

WHY (the 2.61.0 audit's standing fix): prior releases repeatedly shipped stale
docs despite the release-prep skill, because a read-through checks prose for
internal coherence — stale content reads fine.  Staleness is only catchable by
verifying each claim against the SOURCE.  These tests encode the three
mechanically-checkable claim classes so the drift fails CI next time, not a
human eyeball:

  G1 — version tokens in current-state docs == ``mlx_mfa.__version__`` (the SoT).
  G2 — every ``MFA_*`` knob documented as CURRENT in ENV_VARS.md exists in the
       ``_knobs.py`` registry (a doc listing a removed/typo knob fails).
  G3 — variant availability: V4/V5 are removed from the build, so no doc may
       present them as a live/available kernel (source-derived: the .cpp are
       gone + the enable knobs are absent from the registry).

NOT covered here (flagged follow-up): executing every doc code-snippet in CI
(a doctest-extraction harness).  The 2.61.0 audit executed the snippets by hand
(all run); automating extraction is a separate, larger harness.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import mlx_mfa
from mlx_mfa import _knobs

_ROOT = Path(__file__).parent.parent


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text()


# ── G1: version consistency ──────────────────────────────────────────────────
# The version-bearing lines in the current-state docs must equal the SoT.
# (Historical/changelog version refs are NOT matched — only these explicit
# current-state stamps, which is exactly where the recurring miss happened.)
_VERSION_STAMPS = [
    ("README.md", r"Current version:\s*\*\*([0-9]+\.[0-9]+\.[0-9]+)\*\*"),
    ("docs/reference/API_MANUAL.md", r"^Version:\s*\*\*([0-9]+\.[0-9]+\.[0-9]+)\*\*"),
]


@pytest.mark.parametrize("rel,pat", _VERSION_STAMPS, ids=[s[0] for s in _VERSION_STAMPS])
def test_doc_version_matches_sot(rel, pat):
    m = re.search(pat, _read(rel), re.MULTILINE)
    assert m is not None, f"{rel}: no current-version stamp found (pattern {pat!r})"
    assert m.group(1) == mlx_mfa.__version__, (
        f"{rel} version stamp {m.group(1)} != SoT mlx_mfa.__version__ "
        f"{mlx_mfa.__version__} — bump the doc (or the SoT).")


# ── G2: documented knobs ⊆ registry (+ intentional removed-mentions allowlist) ─
# Knobs intentionally NAMED in ENV_VARS.md as REMOVED (so a reader migrating off
# them finds the note).  Anything else outside KNOWN_KNOBS is a stale/typo knob.
_DOC_REMOVED_KNOBS = {
    "MFA_ENABLE_V4", "MFA_ENABLE_V5",
    "MFA_V5_FORCE_BK", "MFA_V5_FORCE_BD_TILE", "MFA_V5_FORCE_BQ", "MFA_V5_FORCE_WM",
}
# Templated/prefix knob families (live name is built at runtime) — registry lists
# the prefix, the doc may show a templated example.
_PREFIX_OK = _knobs.PREFIX_KNOBS


def test_env_vars_doc_knobs_in_registry():
    text = _read("ENV_VARS.md")
    tokens = set(re.findall(r"`(MFA_[A-Z0-9_]+|MLX_MFA_[A-Z0-9_]+)`", text))
    known = set(_knobs.KNOWN_KNOBS) | set(_knobs.CPP_KNOBS) | _DOC_REMOVED_KNOBS
    unknown = {
        t for t in tokens
        if t not in known and not any(t.startswith(p) for p in _PREFIX_OK)
    }
    assert not unknown, (
        f"ENV_VARS.md documents knob(s) not in the _knobs.py registry and not "
        f"in the removed-allowlist: {sorted(unknown)} — either a typo, or a knob "
        f"was removed from the registry while the doc still lists it as live.")


# ── G3: V4/V5 are removed from the build → no doc may present them as live ─────
def test_v4_v5_removed_from_build_source():
    csrc = _ROOT / "csrc"
    for v in ("v4", "v5"):
        assert not (csrc / f"mfa_steel_fwd_{v}.cpp").exists(), (
            f"mfa_steel_fwd_{v}.cpp exists — V{v[-1]} was supposed to be removed "
            f"from the build (Lot-2); update this guard if it was re-added.")
    assert "MFA_ENABLE_V4" not in _knobs.KNOWN_KNOBS
    assert "MFA_ENABLE_V5" not in _knobs.KNOWN_KNOBS


_INSCOPE_DOCS = [
    "README.md", "ENV_VARS.md", "NAMING.md",
    "docs/reference/ARCHITECTURE.md", "docs/reference/FEATURE_COVERAGE.md",
    "docs/reference/dense-steel-family-spec.md", "docs/reference/SERVING_GUIDE.md",
]


@pytest.mark.parametrize("rel", _INSCOPE_DOCS)
def test_no_doc_presents_v4_v5_as_live(rel):
    """Every MFA_ENABLE_V4/V5 mention must sit in a removal/retired CONTEXT (the
    line itself or a nearby section header / note — the only honest way to
    mention a removed knob)."""
    _RM = re.compile(r"remov|retir|no longer|gone|absent|REMOVED|dropped|deprecat|stale",
                     re.IGNORECASE)
    lines = _read(rel).splitlines()
    bad = []
    for i, ln in enumerate(lines):
        if "MFA_ENABLE_V4" in ln or "MFA_ENABLE_V5" in ln:
            # removal keyword on the line, or within the preceding 6 lines
            # (e.g. a "## ... REMOVED" section header above a knob list).
            window = "\n".join(lines[max(0, i - 6):i + 1])
            if not _RM.search(window):
                bad.append(ln.strip()[:100])
    assert not bad, (
        f"{rel}: MFA_ENABLE_V4/V5 mentioned outside a removal context (presented "
        f"as live?) — V4/V5 were removed from the build (Lot-2):\n  " + "\n  ".join(bad))
