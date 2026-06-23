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


def _doc_knobs() -> set:
    # CC-12 (volet E): skip rows explicitly marked "(not an env var)" — those are
    # anti-knob clarifications (e.g. MFA_REQUIRE_MSL4, a Metal-source sentinel), NOT
    # knob declarations, so they must not be required in the registry.
    out: set = set()
    for line in _read("ENV_VARS.md").splitlines():
        if "not an env var" in line.lower():
            continue
        out.update(re.findall(r"`(MFA_[A-Z0-9_]+|MLX_MFA_[A-Z0-9_]+)`", line))
    return out


def test_env_vars_doc_knobs_in_registry():
    """G2-forward: every documented knob exists in the registry (catch stale/typo)."""
    # REMOVED_KNOBS (audit 2026-06-21) are intentionally NAMED in ENV_VARS.md as
    # removed/ghost so migrating readers find the note — allow them like the V4/V5 set.
    known = (set(_knobs.KNOWN_KNOBS) | set(_knobs.CPP_KNOBS)
             | set(_knobs.REMOVED_KNOBS) | _DOC_REMOVED_KNOBS)
    unknown = {
        t for t in _doc_knobs()
        if t not in known and not any(t.startswith(p) for p in _PREFIX_OK)
    }
    assert not unknown, (
        f"ENV_VARS.md documents knob(s) not in the _knobs.py registry and not "
        f"in the removed-allowlist: {sorted(unknown)} — either a typo, or a knob "
        f"was removed from the registry while the doc still lists it as live.")


def test_registry_knobs_documented_or_baselined():
    """G2-reverse (bidirectional): a registry knob must be documented OR in the
    frozen 'known-undocumented' baseline — so a NEW knob added to the registry
    without a doc entry fails CI (forces a doc decision). Shrinking the baseline
    (documenting more) always passes."""
    import json
    baseline = set(json.loads(
        (Path(__file__).parent / "doc_knobs_undocumented_baseline.json").read_text())["knobs"])
    documented = _doc_knobs()
    reg = set(_knobs.KNOWN_KNOBS) | set(_knobs.CPP_KNOBS)
    undocumented_now = {
        k for k in reg
        if k not in documented and not any(k.startswith(p) for p in _PREFIX_OK)
    }
    new_gap = undocumented_now - baseline
    assert not new_gap, (
        f"registry knob(s) neither documented in ENV_VARS.md nor in the frozen "
        f"undocumented-baseline: {sorted(new_gap)} — a new knob was added without "
        f"a doc decision. Document it, or (if intentionally internal) regenerate "
        f"tests/doc_knobs_undocumented_baseline.json.")


# ── G2-source: every LIVE python-read knob is registered (audit H4 / M-01) ────
# The direction that let MFA_REQUIRE_NAX slip through: a knob READ in mlx_mfa/*.py
# but absent from KNOWN_KNOBS.  (registry ⊆ live-read is NOT enforceable — ~75
# registry knobs are read C++-side / templated; that's by design.)
_ENV_READ_PATTERNS = [
    r'os\.environ\.get\(\s*["\'](MFA_[A-Z0-9_]+|MLX_MFA_[A-Z0-9_]+)["\']',
    r'os\.environ\[\s*["\'](MFA_[A-Z0-9_]+|MLX_MFA_[A-Z0-9_]+)["\']',
    r'os\.getenv\(\s*["\'](MFA_[A-Z0-9_]+|MLX_MFA_[A-Z0-9_]+)["\']',
    r'\bgetenv\(\s*["\'](MFA_[A-Z0-9_]+|MLX_MFA_[A-Z0-9_]+)["\']',
    r'getenv_aliased\(\s*["\'](MFA_[A-Z0-9_]+|MLX_MFA_[A-Z0-9_]+)["\']',
]


def _python_read_knobs() -> set:
    pkg = Path(mlx_mfa.__file__).parent
    read = set()
    for py in pkg.rglob("*.py"):
        if py.name == "_knobs.py":
            continue
        t = py.read_text(encoding="utf-8")
        for pat in _ENV_READ_PATTERNS:
            read |= set(re.findall(pat, t))
    return read


def test_python_read_knobs_are_registered():
    """G2-source: a knob explicitly read in mlx_mfa/*.py must be in KNOWN_KNOBS
    (or a prefix family). Catches the 'live but unregistered' class — the exact
    miss (MFA_REQUIRE_NAX/MFA_SILENCE_NAX_WARNING) that the validator then
    false-flagged as a typo."""
    known = set(_knobs.KNOWN_KNOBS)
    missing = sorted(
        k for k in _python_read_knobs()
        if k not in known and not any(k.startswith(p) for p in _PREFIX_OK))
    assert not missing, (
        f"knob(s) read in mlx_mfa/*.py but absent from _knobs.KNOWN_KNOBS: "
        f"{missing} — add them to the registry (and ENV_VARS.md).")


def test_removed_knobs_are_separated_from_registry():
    """The ghost registry must be DISJOINT from the live registry — a removed
    knob re-appearing in KNOWN_KNOBS would defeat the 'removed — no effect'
    signal (audit M7/L-01)."""
    overlap = set(_knobs.REMOVED_KNOBS) & set(_knobs.KNOWN_KNOBS)
    assert not overlap, f"REMOVED_KNOBS must not also be in KNOWN_KNOBS: {sorted(overlap)}"


# ── G1b: documented export count == runtime len(__all__) (audit H3) ───────────
def test_export_count_matches_doc():
    """The 'Public exports: N' claim in API_MANUAL.md must equal the runtime
    len(mlx_mfa.__all__). The recurring miss: new exports (has_nax/NaxUnavailable)
    bumped __all__ to 103 while the docs still said 101, and G1 only checked the
    version string."""
    text = _read("docs/reference/API_MANUAL.md")
    m = re.search(r"Public exports:\s*\*\*([0-9]+)\*\*", text)
    assert m is not None, "API_MANUAL.md: no 'Public exports: **N**' stamp found"
    documented = int(m.group(1))
    actual = len(mlx_mfa.__all__)
    assert documented == actual, (
        f"API_MANUAL.md documents {documented} public exports but "
        f"len(mlx_mfa.__all__)={actual} — update the doc count.")


_INSCOPE_DOCS = [
    "README.md", "ENV_VARS.md", "NAMING.md",
    "docs/reference/ARCHITECTURE.md", "docs/reference/FEATURE_COVERAGE.md",
    "docs/reference/dense-steel-family-spec.md", "docs/reference/SERVING_GUIDE.md",
]


# ── G3: source-derived variant availability ──────────────────────────────────
# The historical STEEL-forward variant universe; "retired" is derived as
# (universe − compiled), so a FUTURE retirement (e.g. dropping v3.cpp) auto-joins
# the retired set without editing this test. NOT hardcoded to V4/V5.
_VARIANT_UNIVERSE = {"v1", "v2", "v3", "v4", "v5", "v6_nax"}


def _compiled_variants() -> set:
    """Derived from the build: which STEEL-forward .cpp actually exist."""
    out = set()
    for p in (_ROOT / "csrc").glob("mfa_steel_fwd*.cpp"):
        suffix = p.stem.replace("mfa_steel_fwd", "").lstrip("_")
        out.add(suffix or "v1")  # bare mfa_steel_fwd.cpp == V1
    return out


def _retired_variants() -> set:
    return _VARIANT_UNIVERSE - _compiled_variants()


def test_variant_set_source_derived():
    """Compiled variants are exactly the expected current set; retired = the rest.
    A future retirement shrinks `_compiled_variants()` and the retired set + the
    no-doc-presents-retired guard below auto-cover it."""
    compiled = _compiled_variants()
    assert "v1" in compiled and "v2" in compiled and "v3" in compiled and "v6_nax" in compiled
    retired = _retired_variants()
    # V4/V5 are the current retired set; their enable knobs must be gone.
    for v in retired:
        assert f"MFA_ENABLE_{v.upper()}" not in _knobs.KNOWN_KNOBS, (
            f"variant {v} is retired (no compiled .cpp) but MFA_ENABLE_{v.upper()} "
            f"is still in the registry.")


@pytest.mark.parametrize("rel", _INSCOPE_DOCS)
def test_no_doc_presents_retired_variant_as_live(rel):
    """No in-scope doc may present a RETIRED variant's enable knob as live (the
    retired set is source-derived, so this auto-covers a future retirement)."""
    _RM = re.compile(r"remov|retir|no longer|gone|absent|REMOVED|dropped|deprecat|stale",
                     re.IGNORECASE)
    retired_knobs = [f"MFA_ENABLE_{v.upper()}" for v in _retired_variants()]
    lines = _read(rel).splitlines()
    bad = []
    for i, ln in enumerate(lines):
        if any(rk in ln for rk in retired_knobs):
            window = "\n".join(lines[max(0, i - 6):i + 1])
            if not _RM.search(window):
                bad.append(ln.strip()[:100])
    assert not bad, (
        f"{rel}: a RETIRED variant enable-knob {retired_knobs} mentioned outside a "
        f"removal context (presented as live?):\n  " + "\n  ".join(bad))


# (the former hardcoded-V4/V5 `test_no_doc_presents_v4_v5_as_live` is superseded by
#  the source-derived `test_no_doc_presents_retired_variant_as_live` above.)


# ── Volet O: doc-staleness guards (CX-R11-01/02) ────────────────────────────────
# The sparse_attention_dispatch f32 contract: f16/bf16 may use the native OR SDPA
# route; f32 is accepted but routes to SDPA ONLY (the native kernel is f16/bf16-
# only). A positive "f32 … both routes" claim is therefore stale/wrong.
_F32_BOTH_ROUTES = re.compile(
    r"(?:f32|float32)[^.\n]*both\s+routes", re.IGNORECASE)

# current-contract surfaces (NOT historical CHANGELOG/inventory narrative).
_CURRENT_CONTRACT_FILES = (
    "mlx_mfa/lcsa_nax.py",
    "tests/test_hardening_l.py",
    "tests/test_hardening_m.py",
    "tests/test_hardening_n.py",
)


def _is_stale_f32_line(line):
    """A line is a stale f32-both-routes claim if it puts f32 before 'both routes'
    AND does not attribute f32 to SDPA / explicitly negate ('NOT both routes')."""
    if not _F32_BOTH_ROUTES.search(line):
        return False
    low = line.lower()
    if "sdpa" in low:                       # f32 attributed to the SDPA path
        return False
    if re.search(r'not\s+"?both\s+routes', low):  # explicit negation
        return False
    return True


@pytest.mark.parametrize("rel", _CURRENT_CONTRACT_FILES)
def test_no_stale_f32_both_routes_in_current_contract(rel):
    bad = [ln.strip()[:120] for ln in _read(rel).splitlines() if _is_stale_f32_line(ln)]
    assert not bad, (
        f"{rel}: stale 'f32 … both routes' claim — f32 routes to SDPA only:\n  "
        + "\n  ".join(bad))


def test_stale_f32_guard_bites():
    # the guard must catch a positive f32-both-routes claim …
    assert _is_stale_f32_line("Dtype not restricted — f32 runs on both routes.")
    assert _is_stale_f32_line("# f32 is valid on both routes")
    # … and must NOT false-positive on the accurate contract phrasings:
    assert not _is_stale_f32_line("f16/bf16 (both routes), f32 (SDPA path)")
    assert not _is_stale_f32_line("f32 routes to SDPA only, NOT both routes")
    assert not _is_stale_f32_line("# reject-malformed: both routes")  # no f32


def _enumeration_counts():
    """Live (public_computational, raw_computational) from the enumeration."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_enum_doc", str(_ROOT / "scripts" / "enumerate_api_surface.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    pub = m.public_exports()
    npub = sum(1 for n in pub if m.classify_public(n, pub[n])[0] == "computational")
    nraw = sum(1 for n in m.raw_bindings() if m.classify_raw(n)[0] == "computational")
    return npub, nraw


def test_inventory_current_counts_match_enumeration():
    """The inventory's 'INVENTORY COMPLETE' current count must equal the live
    enumeration (so a future entry that bumps the count fails this until the doc
    is updated). Historical volet sections are unaffected — this matches only the
    '**N public** + M raw' completeness line."""
    npub, nraw = _enumeration_counts()
    text = _read("audit/round3_remediation/surface_inventory.md")
    matches = re.findall(r"\*\*(\d+) public\*\*\s*\+\s*(\d+) raw", text)
    assert matches, "inventory: no '**N public** + M raw' completeness line found"
    # the LAST such line is the current 'INVENTORY COMPLETE' statement.
    cur_pub, cur_raw = int(matches[-1][0]), int(matches[-1][1])
    assert (cur_pub, cur_raw) == (npub, nraw), (
        f"inventory current count **{cur_pub} public + {cur_raw} raw** != live "
        f"enumeration ({npub} public + {nraw} raw) — update the inventory.")


def test_inventory_count_guard_bites():
    # the count parser/comparison must catch a wrong current count.
    npub, nraw = _enumeration_counts()
    fake = f"## INVENTORY COMPLETE\n(**{npub + 99} public** + {nraw} raw, all AUDITED)"
    m = re.findall(r"\*\*(\d+) public\*\*\s*\+\s*(\d+) raw", fake)
    assert m and (int(m[-1][0]), int(m[-1][1])) != (npub, nraw)
