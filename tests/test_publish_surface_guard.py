"""Publish-surface guard — the journal/dev cruft must not leak into the published artifact.

Marco's publication policy: the published sdist ships ONLY the LICENSE files + current-state
docs + code (csrc/mlx_mfa/examples/scripts) + tests + docs/reference/. Devlogs / plans /
phase-reports / campaign docs are RETAINED in-repo (git history + the gitignored
`.doc-archive/`) but NEVER published; dev harnesses (bench/, benchmarks/) and local scratch
(.claude/) are NEVER published either.

Two surfaces, two checks:
  1. The BUILT SDIST (the real published artifact) — every member must be in an explicit
     allowlist.  v2.58.1 P3 rewrite: the prior guard checked `MANIFEST.in` (inert under
     scikit-build-core) + `git ls-files` (git-tracked, NOT the built tarball) — which is why
     `.claude/settings.local.json` (untracked, not-gitignored) shipped in the 2.58.0 sdist
     undetected.  This now builds the sdist and asserts against the tarball itself.
  2. The TRACKED REPO TREE (git ls-files) — no journal docs on the public tree (D-addendum).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"

# M-04 FIX (audit, 2026-06-21): in a RELEASE context the guard must NOT be able
# to go green without inspecting a real artifact.  When `MFA_RELEASE_GATE=1`
# (set by the pre-tag flow), missing build tooling or a failed sdist build is
# FATAL instead of a skip.  Outside the gate (ordinary/offline CI) it still
# skips so the suite stays green without network.  NOTE: this fixture builds the
# sdist from the WORKING TREE (not a clean `git archive` export) — that is
# deliberate: the leak this guard exists to catch (the 2.58.0
# `.claude/settings.local.json`) was *untracked-and-not-gitignored*, which a
# tracked-only export would silently drop.
def _skip_or_fail(reason: str) -> None:
    if os.environ.get("MFA_RELEASE_GATE") not in (None, "", "0"):
        pytest.fail("MFA_RELEASE_GATE set — " + reason)
    pytest.skip(reason)

# ── The explicit published-sdist allowlist ────────────────────────────────────
# CC-07 (volet E3, maintainer signoff 2026-06-23): CLAUDE.md / CLAUDE_V6_NAX.md are
# internal agent-process / sprint engineering docs, NOT user-facing — excluded from
# the sdist (see pyproject [tool.scikit-build.sdist].exclude). Removed from this
# allowlist so a future re-introduction would be flagged as a leak.
# (TST-15, review 2026-09: renamed — the tree guard below used to rebind the same name.)
_ALLOWED_SDIST_ROOT_DOCS = {
    "README.md", "CHANGELOG.md", "RESULTS.md", "ENV_VARS.md", "NAMING.md",
    "CONTRIBUTING.md",
    "LICENSE", "LICENSE-DRAWTHINGS", "THIRD_PARTY_LICENSES",
}
# Permitted top-level directories in the sdist (code + tests + scripts).
_ALLOWED_TOP_DIRS = {"csrc", "mlx_mfa", "examples", "tests", "scripts"}
# Permitted root files = the docs + build/metadata files sdist always emits.
_ALLOWED_ROOT_FILES = _ALLOWED_SDIST_ROOT_DOCS | {"CMakeLists.txt", "pyproject.toml", "PKG-INFO"}
# Files the sdist GENERATES (never tracked).
_SDIST_GENERATED = {"PKG-INFO"}

# 2.62.2 (review 2026-09, A10 — maintainer decision): per-FILE dev-only set — kept in
# the repository, never published (sdist nor wheel).  Mirror of the per-file entries of
# pyproject [tool.scikit-build.sdist].exclude (cross-checked below).
_DEV_ONLY_FILES = (
    # historical, not loaded (async_v2 retired in 2.62.2)
    "csrc/async_v2_kernel.metal", "scripts/build_async_metallib.sh",
    # compiled / runnable only with -DMFA_BUILD_PROBES=ON (BLD-14)
    "csrc/mpp_int8_bench.mm", "csrc/v6_nax_primitives_probe.cpp",
    "csrc/v6_nax_toolchain_probe.cpp", "tests/test_varlen_source_generation_guards.py",
    # campaign / dev scripts (no shipped test imports them)
    "scripts/audit_dispatch_grouping_k0.py", "scripts/audit_dit_dispatch.py",
    "scripts/build_metallibs.sh",
    # own kernels kept for the record, never wired into dispatch, + their tests
    "mlx_mfa/gqa_decode_cider.py", "mlx_mfa/topk_stream.py",
    "tests/test_phase2_ii11_gqa_decode_cider.py", "tests/test_phase2_ii3_topk_stream.py",
)


def _strip_prefix(name: str) -> str:
    """`mlx_mfa-2.58.0/csrc/x.cpp` → `csrc/x.cpp`."""
    return name.split("/", 1)[1] if "/" in name else ""


def _disallowed_members(members):
    """Return sdist members outside the explicit publication allowlist.

    docs/ is allowed ONLY under docs/reference/.  Anything else at the top level
    (e.g. .claude/, bench/, benchmarks/, *.log, .gitignore) is a leak.
    """
    bad = []
    for raw in members:
        rel = _strip_prefix(raw)
        if not rel or rel.endswith("/"):
            continue
        if rel in _DEV_ONLY_FILES:
            bad.append(rel)
            continue
        parts = rel.split("/")
        top = parts[0]
        if top in _ALLOWED_TOP_DIRS:
            continue
        if top == "docs":
            if rel.startswith("docs/reference/"):
                continue
            bad.append(rel)
            continue
        if len(parts) == 1 and top in _ALLOWED_ROOT_FILES:
            continue
        bad.append(rel)
    return bad


@pytest.fixture(scope="module")
def _built_sdist(tmp_path_factory):
    """Build the real sdist into a module temp dir; return (tarball path, members)."""
    try:
        import build  # noqa: F401
    except Exception:
        _skip_or_fail("`build` not installed — cannot assert against the built sdist")
    td = tmp_path_factory.mktemp("sdist")
    r = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "-o", str(td)],
        cwd=_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        _skip_or_fail(f"sdist build failed (network/offline?):\n{r.stderr[-800:]}")
    tars = list(Path(td).glob("*.tar.gz"))
    assert tars, "no sdist tarball produced"
    with tarfile.open(tars[0]) as t:
        return tars[0], t.getnames()


@pytest.fixture(scope="module")
def _built_sdist_members(_built_sdist):
    return _built_sdist[1]


def test_built_sdist_only_publication_surface(_built_sdist_members):
    """The REAL built sdist must contain ONLY the explicit publication allowlist —
    no .claude/, bench/, benchmarks/, *.log, journal docs (the 2.58.0 leak class)."""
    bad = _disallowed_members(_built_sdist_members)
    assert not bad, (
        "built sdist ships files OUTSIDE the publication allowlist (cruft/journal leak): "
        f"{sorted(bad)[:25]}{' …' if len(bad) > 25 else ''}")
    # sanity: the intended set is actually present (not an over-aggressive exclude)
    rels = {_strip_prefix(m) for m in _built_sdist_members}
    for required in ("README.md", "CHANGELOG.md", "LICENSE", "pyproject.toml",
                     "mlx_mfa/attention.py", "csrc/bindings.cpp",
                     "docs/reference/dispatch-map.md"):
        assert required in rels, f"intended publication file missing from sdist: {required}"


def test_tracked_but_excluded_dirs_absent_from_sdist(_built_sdist_members):
    """Volet F: the tracked-but-sdist-excluded dirs (audit/ enumeration artifacts,
    release-gate/ M5 receipts) must NOT ship to PyPI users — 0 members each."""
    rels = {_strip_prefix(m) for m in _built_sdist_members}
    for excluded in ("audit/", "release-gate/"):
        leaked = sorted(r for r in rels if r.startswith(excluded))
        assert not leaked, (
            f"{excluded} leaked into the published sdist (must be sdist-excluded): {leaked}")


def test_sdist_guard_catches_a_synthetic_stray():
    """Self-test: the allowlist checker trips on planted strays (the .claude/-class leak,
    bench/, a log, a stray root file, a journal doc) and passes a clean member set."""
    pfx = "mlx_mfa-2.58.0/"
    strays = [pfx + p for p in (
        ".claude/settings.local.json", "bench/foo.py", "benchmarks/x.log",
        "autoresearch_kernel.log", "docs/v50/campaign-2026-06/x.md",
    )]
    caught = _disallowed_members(strays)
    assert len(caught) == 5, f"sdist guard missed a planted stray: caught {caught}"
    clean = [pfx + p for p in (
        "README.md", "LICENSE", "pyproject.toml", "CMakeLists.txt", "PKG-INFO",
        "mlx_mfa/attention.py", "csrc/bindings.cpp", "tests/test_x.py",
        "examples/y.py", "scripts/check_venv.sh", "docs/reference/INDEX.md",
    )]
    assert _disallowed_members(clean) == [], "sdist guard false-positived a clean member set"
    assert _disallowed_members([pfx + f for f in _DEV_ONLY_FILES]) == list(_DEV_ONLY_FILES)


# ── BLD-13 (review 2026-09): the sdist is checked PER FILE, not per directory ─────
# The directory allowlist above let untracked scratch INSIDE csrc/ mlx_mfa/ tests/
# scripts/ examples/ ship unflagged (scikit-build-core packs untracked, non-ignored
# files).  Per file: every sdist member must be a tracked, publishable, non-dev-only
# file (or sdist-generated), and every such tracked file must be in the sdist.
def _per_file_diff(sdist_rels, tracked):
    """-> (leaked, missing): sdist members not justified by the tracked tree, and
    publishable tracked files absent from the sdist."""
    expected = {p for p in tracked if not _disallowed_members(["x/" + p])}
    actual = set(sdist_rels) - _SDIST_GENERATED
    return sorted(actual - expected), sorted(expected - actual)


def test_built_sdist_matches_tracked_tree_per_file(_built_sdist_members):
    tracked = [p for p in _tracked_files() if (_ROOT / p).is_file()]
    rels = [r for r in (_strip_prefix(m) for m in _built_sdist_members)
            if r and not r.endswith("/")]
    leaked, missing = _per_file_diff(rels, tracked)
    assert not leaked, (
        "the sdist ships files that are untracked or dev-only (git add/gitignore them, "
        f"or exclude them in pyproject): {leaked[:25]}")
    assert not missing, f"publishable tracked files missing from the sdist: {missing[:25]}"


def test_dev_only_files_kept_in_repo_excluded_from_sdist(_built_sdist_members):
    tracked = set(_tracked_files())
    rels = {_strip_prefix(m) for m in _built_sdist_members}
    text = _PYPROJECT.read_text(encoding="utf-8")
    for f in _DEV_ONLY_FILES:
        assert f in tracked, f"dev-only file must stay in the repository: {f}"
        assert f not in rels, f"dev-only file shipped in the sdist: {f}"
        assert f'"{f}"' in text, f"dev-only file not listed in pyproject sdist.exclude: {f}"


def test_per_file_guard_catches_untracked_scratch():
    """Self-test: untracked scratch inside an allowed dir and a dev-only file are
    leaks; a tracked publishable file missing from the sdist is reported."""
    tracked = ["csrc/a.cpp", "mlx_mfa/b.py", "README.md", "CLAUDE.md",
               "mlx_mfa/topk_stream.py", "devnotes/x.md"]
    sdist = ["csrc/a.cpp", "csrc/scratch_tmp.cpp", "mlx_mfa/topk_stream.py", "PKG-INFO"]
    leaked, missing = _per_file_diff(sdist, tracked)
    assert leaked == ["csrc/scratch_tmp.cpp", "mlx_mfa/topk_stream.py"], leaked
    assert missing == ["README.md", "mlx_mfa/b.py"], missing


def test_pyproject_wheel_packages_is_mlx_mfa_only():
    """Static pre-check (the artifact itself is checked below)."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    assert 'wheel.packages = ["mlx_mfa"]' in text, "wheel.packages changed — re-verify no journal in the wheel"


# ── TST-15 (review 2026-09): inspect a REAL wheel, not a pyproject string ─────────
# The release model is sdist-only: users get the wheel pip builds FROM the sdist, so
# that is the wheel checked.  MFA_WHEEL_PATH=<wheel> inspects a given artifact; under
# MFA_RELEASE_GATE the guard builds it from the built sdist itself; otherwise it skips
# (compiling _ext takes minutes — too slow for the ordinary suite).
@pytest.fixture(scope="module")
def _real_wheel_members(request, tmp_path_factory):
    path = os.environ.get("MFA_WHEEL_PATH")
    if not path:
        if os.environ.get("MFA_RELEASE_GATE") in (None, "", "0"):
            pytest.skip("real-wheel check: set MFA_WHEEL_PATH=<wheel built from the sdist> "
                        "(the release gate builds it itself)")
        sdist_path = request.getfixturevalue("_built_sdist")[0]
        td = tmp_path_factory.mktemp("wheel")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "-w", str(td), str(sdist_path)], capture_output=True, text=True)
        if r.returncode != 0:
            pytest.fail(f"wheel build from the sdist failed:\n{r.stderr[-1500:]}")
        wheels = list(Path(td).glob("*.whl"))
        assert len(wheels) == 1, wheels
        path = str(wheels[0])
    with zipfile.ZipFile(path) as z:
        return z.namelist()


def test_real_wheel_ships_only_the_package(_real_wheel_members):
    names = _real_wheel_members
    dist_info = {n.split("/", 1)[0] for n in names if ".dist-info/" in n}
    assert len(dist_info) == 1, dist_info
    stray = [n for n in names if not (n.startswith("mlx_mfa/") or n.split("/", 1)[0] in dist_info)]
    assert not stray, f"the wheel ships files outside the mlx_mfa package: {stray[:25]}"
    dev = [f for f in _DEV_ONLY_FILES if f in names]
    assert not dev, f"dev-only files shipped in the wheel: {dev}"
    assert any(re.fullmatch(r"mlx_mfa/_ext\..*\.so", n) for n in names), "no compiled _ext in the wheel"
    for required in ("mlx_mfa/__init__.py", "mlx_mfa/attention.py"):
        assert required in names, f"package module missing from the wheel: {required}"


# ── D-addendum: extend the guard from the WHEEL surface to the TRACKED REPO TREE ──
# Marco's decision: the journal is OFF the public tracked tree too (not merely
# wheel-excluded), retained for provenance via git history + the gitignored
# `.doc-archive/`. The tracked tree must show ONLY current-state: code + tests +
# the permitted docs (root current-state + docs/reference/). A `git add` that
# re-tracks a journal path (devnotes/, docs/ outside docs/reference/, an
# AUTORESEARCH task-plan, an autoresearch log) fails CI.

# A tracked DOC path is permitted only if it is a root current-state doc or lives
# under docs/reference/.  Journal doc trees (devnotes/, docs/<anything-but-reference>)
# are forbidden on the tracked tree.  (Documentary: CLAUDE*.md are tracked but
# sdist-excluded — contrast _ALLOWED_SDIST_ROOT_DOCS.)
_ALLOWED_TREE_ROOT_DOCS = {
    "README.md", "CHANGELOG.md", "RESULTS.md", "ENV_VARS.md", "NAMING.md",
    "CLAUDE.md", "CLAUDE_V6_NAX.md", "CONTRIBUTING.md",
    "LICENSE", "LICENSE-DRAWTHINGS", "THIRD_PARTY_LICENSES",
}


def _tracked_files():
    if not (_ROOT / ".git").exists():
        pytest.skip("not a git checkout (source archive / CI) — tree-guard is git-only")
    out = subprocess.run(
        ["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.splitlines() if p]


def _tree_journal_violations(paths):
    """Return tracked paths that are RETAIN-class journal (must be off the tree)."""
    bad = []
    for p in paths:
        # devnotes/ is journal in its entirety.
        if p == "devnotes" or p.startswith("devnotes/"):
            bad.append(p)
            continue
        # docs/ is journal EXCEPT the current-state reference home docs/reference/.
        if p.startswith("docs/") and not p.startswith("docs/reference/"):
            bad.append(p)
            continue
        # root research-task plans + their logs are journal.
        if re.fullmatch(r"AUTORESEARCH.*\.md", p) or re.fullmatch(r"autoresearch.*\.log", p):
            bad.append(p)
    return bad


def test_tracked_tree_contains_no_journal():
    """The public tracked tree must carry current-state docs only — no journal.

    docs/ is allowed ONLY under docs/reference/; devnotes/ and AUTORESEARCH task
    plans must be archived (git history + .doc-archive/), not tracked.
    """
    bad = _tree_journal_violations(_tracked_files())
    assert not bad, (
        "journal paths re-appeared on the tracked tree (archive them to .doc-archive/ "
        f"+ git rm): {bad[:20]}{' …' if len(bad) > 20 else ''}")


def test_tracked_docs_reference_is_present():
    """The relocated current-state reference must be tracked under docs/reference/."""
    tracked = set(_tracked_files())
    for required in (
        "docs/reference/dispatch-map.md",
        "docs/reference/sparse-family-spec.md",
        "docs/reference/API_MANUAL.md",
        "docs/reference/doc-claim-lock-map.md",
    ):
        assert required in tracked, f"current-state reference missing from tracked tree: {required}"


def test_tree_guard_catches_a_planted_journal_file():
    """Self-test: the tree detector trips on planted tracked journal paths, and a
    legitimate tracked-tree listing does NOT trip it."""
    planted = [
        "devnotes/SESSION_LOG.md",
        "docs/v6-nax/sparse-bug-investigation.md",
        "AUTORESEARCH.md",
        "autoresearch_kernel.log",
    ]
    assert len(_tree_journal_violations(planted)) == 4, "tree guard missed a planted journal path"
    legit = [
        "mlx_mfa/attention.py", "tests/test_attention.py", "csrc/bindings.cpp",
        "README.md", "CLAUDE_V6_NAX.md", "docs/reference/API_MANUAL.md",
        "docs/reference/dispatch-map.md", "examples/cross_attention.py",
    ]
    assert _tree_journal_violations(legit) == [], "tree guard false-positived a legitimate path"
