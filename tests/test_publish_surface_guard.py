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

import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _ROOT / "pyproject.toml"

# ── The explicit published-sdist allowlist ────────────────────────────────────
_ALLOWED_ROOT_DOCS = {
    "README.md", "CHANGELOG.md", "RESULTS.md", "ENV_VARS.md", "NAMING.md",
    "CLAUDE.md", "CLAUDE_V6_NAX.md",
    "LICENSE", "LICENSE-DRAWTHINGS", "THIRD_PARTY_LICENSES",
}
# Permitted top-level directories in the sdist (code + tests + scripts).
_ALLOWED_TOP_DIRS = {"csrc", "mlx_mfa", "examples", "tests", "scripts"}
# Permitted root files = the docs + build/metadata files sdist always emits.
_ALLOWED_ROOT_FILES = _ALLOWED_ROOT_DOCS | {"CMakeLists.txt", "pyproject.toml", "PKG-INFO"}


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
def _built_sdist_members():
    """Build the real sdist into a temp dir (Rule 12 cleanup) and return its members."""
    try:
        import build  # noqa: F401
    except Exception:
        pytest.skip("`build` not installed — cannot assert against the built sdist")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "-o", td],
            cwd=_ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            pytest.skip(f"sdist build failed (network/offline?):\n{r.stderr[-800:]}")
        tars = list(Path(td).glob("*.tar.gz"))
        assert tars, "no sdist tarball produced"
        with tarfile.open(tars[0]) as t:
            return t.getnames()


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


def test_wheel_ships_only_the_package():
    """The wheel must ship only the mlx_mfa package (no docs/devnotes in the wheel)."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    assert 'wheel.packages = ["mlx_mfa"]' in text, "wheel.packages changed — re-verify no journal in the wheel"


# ── D-addendum: extend the guard from the WHEEL surface to the TRACKED REPO TREE ──
# Marco's decision: the journal is OFF the public tracked tree too (not merely
# wheel-excluded), retained for provenance via git history + the gitignored
# `.doc-archive/`. The tracked tree must show ONLY current-state: code + tests +
# the permitted docs (root current-state + docs/reference/). A `git add` that
# re-tracks a journal path (devnotes/, docs/ outside docs/reference/, an
# AUTORESEARCH task-plan, an autoresearch log) fails CI.

# A tracked DOC path is permitted only if it is a root current-state doc or lives
# under docs/reference/.  Journal doc trees (devnotes/, docs/<anything-but-reference>)
# are forbidden on the tracked tree.
_ALLOWED_ROOT_DOCS = {
    "README.md", "CHANGELOG.md", "RESULTS.md", "ENV_VARS.md", "NAMING.md",
    "CLAUDE.md", "CLAUDE_V6_NAX.md",
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
