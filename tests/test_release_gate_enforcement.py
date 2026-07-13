"""Volet D — release-gate enforcement + CI coverage locks.

Closes CX-07 (publish.yml bypassed the gates), CC-14 (M5 locks skip on the M1
runner, uncounted), CX-10 (CI was Python 3.11 only). These are
source/workflow-structure locks + a subprocess test of the M5 fingerprint
precondition; they bite if the enforcement regresses.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_WF = _ROOT / ".github" / "workflows"
_CHECK = _ROOT / "scripts" / "check_m5_gate_fingerprint.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ── CC-14: M5/NAX lock skip-count is frozen (a lock silently dropping = FAIL) ──

# Host-independent count of skipif decorator sites whose condition gates on M5/NAX
# hardware. Frozen so removing/adding an M5 lock changes the suite's M5 surface
# visibly (the marker is attached regardless of whether it skips on this host).
# 69 → 70: added tests/test_sparse_bt_aware_routing.py (BT-aware sparse dispatcher
# routing lock; M5-gated — the NAX sparse kernel is M5+). 70 → 72: added
# tests/test_qmm_nax_lock.py (expert V6 NAX qmm correctness + reject locks).
# 72 -> 73: added tests/test_gna_nax_lock.py (expert GNA V6 NAX locks).
# 73 -> 75: added tests/test_ffn_nax_lock.py (expert linear/GELU correctness
# and malformed-input locks). 75 -> 77: added tests/test_conv3d_lq_nax_lock.py
# (direct LQ-envelope correctness and public non-routing locks). 77 -> 78:
# added tests/test_gna_range_precompute_lock.py (byte-identity lock for the
# per-Q-tile range optimization). 78 -> 79: added the SeedVR2 VAE spatial
# pad-and-slice M5 correctness lock. 79 -> 80: added the public default-off
# byte-identity lock for that same opt-in hook.
# 80 -> 81: added tests/test_gna_residency_swizzle_lock.py (default-vs-grid-
# swizzle byte-identity lock for the opt-in GNA residency probe). 81 -> 82:
# added the hardened sparse beta-3 gate boundary/which-binary lock.
# M5 surface still covered.
_EXPECTED_M5_SKIP_SITES = 82


def _count_m5_skip_sites() -> int:
    pat = re.compile(
        r"is_m5_plus|_HAS_NAX|_M5_PLUS|_IS_M5\b|m5_plus|requires M5|M5\+ NAX|NAX hardware",
        re.I)
    total = 0
    for f in sorted((_ROOT / "tests").rglob("test_*.py")):
        txt = f.read_text(encoding="utf-8")
        for m in re.finditer(r"skipif\s*\(", txt):
            if pat.search(txt[m.start():m.start() + 160]):
                total += 1
    return total


def test_m5_skip_marker_count():
    """CC-14: the count of M5-gated locks must equal the frozen baseline. A lock
    silently dropping out of the suite (deleted / un-gated) changes this count
    even though the locks merely *skip* on a hosted M1 runner."""
    actual = _count_m5_skip_sites()
    assert actual == _EXPECTED_M5_SKIP_SITES, (
        f"M5/NAX lock count changed: {actual} vs frozen {_EXPECTED_M5_SKIP_SITES}. "
        f"If you intentionally added/removed an M5 lock, update the baseline AND "
        f"re-confirm the M5 surface is still covered.")


def test_m5_skip_count_bites():
    """Prove the count lock bites: removing one marker drops the GLOBAL count."""
    pat = re.compile(
        r"is_m5_plus|_HAS_NAX|_M5_PLUS|_IS_M5\b|m5_plus|requires M5|M5\+ NAX|NAX hardware",
        re.I)
    texts = {f: f.read_text(encoding="utf-8")
             for f in sorted((_ROOT / "tests").rglob("test_*.py"))}
    # mutate ONE M5 skipif site (simulate a lock dropping out)
    mutated_one = False
    for f, txt in texts.items():
        ms = [m for m in re.finditer(r"skipif\s*\(", txt)
              if pat.search(txt[m.start():m.start() + 160])]
        if ms:
            i = ms[0].start()
            texts[f] = txt[:i] + "if_removed(" + txt[i + len("skipif("):]
            mutated_one = True
            break
    assert mutated_one, "no M5 skip site found to mutate — counter mis-targeted"
    # recount GLOBALLY across all files (one site now removed)
    total = 0
    for txt in texts.values():
        for m in re.finditer(r"skipif\s*\(", txt):
            if pat.search(txt[m.start():m.start() + 160]):
                total += 1
    assert total == _EXPECTED_M5_SKIP_SITES - 1, (
        f"dropping one M5 lock should change the count to "
        f"{_EXPECTED_M5_SKIP_SITES - 1}, got {total}")


# ── CX-07: publish.yml gates every upload ────────────────────────────────────

def test_publish_yml_enforces_gates():
    y = _read(_WF / "publish.yml")
    assert re.search(r"^\s*gates:", y, re.M), "publish.yml has no `gates` job"
    # the gate job runs the suite, collection floor, surface guard, release-audit,
    # and the M5 fingerprint precondition
    for needle in (
        "pytest tests/",                              # GATE 1 full suite
        "collection",                                  # GATE 2 floor
        "test_publish_surface_guard.py",               # GATE 3 surface guard
        "check_m5_gate_fingerprint.py",                # GATE 5 M5 precondition
    ):
        assert needle in y, f"publish.yml gates job missing: {needle!r}"
    # build depends on gates; both publish jobs depend on build
    assert re.search(r"build:\s*\n\s*name:.*\n\s*needs:\s*gates", y), \
        "publish.yml `build` does not `needs: gates`"
    assert y.count("needs: build") >= 2, \
        "publish-testpypi / publish-pypi must `needs: build` (transitively gates)"


def test_publish_yml_precondition_before_upload():
    """The M5 precondition must be in the gates job that build/publish depend on,
    not after the upload."""
    y = _read(_WF / "publish.yml")
    gate_pos = y.index("check_m5_gate_fingerprint.py")
    upload_pos = y.index("gh-action-pypi-publish")
    assert gate_pos < upload_pos, "M5 precondition appears after the publish step"


# ── CX-10: CI Python matrix covers 3.10–3.14 ─────────────────────────────────

def test_ci_python_matrix_full_range():
    y = _read(_WF / "ci.yml")
    versions = ["3.10", "3.11", "3.12", "3.13", "3.14"]
    # the fallback + packaging jobs must enumerate the full declared range
    full = '["3.10", "3.11", "3.12", "3.13", "3.14"]'
    assert y.count(full) >= 2, (
        "ci.yml must run the 3.10–3.14 matrix on the fallback + packaging jobs "
        f"(found {y.count(full)} full-range matrices)")
    for v in versions:
        assert v in y, f"ci.yml missing Python {v}"
    # the per-interpreter sdist build+import contract step exists
    assert "Install the built sdist" in y and "ext-load-failed" in y, \
        "ci.yml lost the per-version sdist compile-at-install + import check"


# ── M5 fingerprint precondition behaviour (subprocess) ───────────────────────

def _valid_receipt(git_sha: str) -> dict:
    import hashlib
    fps = {"dense_D128_auto_nax": 1.2e-4, "dense_D64_auto_sdpa": 0.0}
    return {
        "release_version": _pyproject_version(), "git_sha": git_sha,
        "mlx_version": "0.31.2", "device": "M5", "chip": "M5",
        "is_m5_plus": True, "date_utc": "2026-06-22", "has_nax": True,
        "nax_reason": "available", "fingerprints": fps,
        "fingerprints_sha256": hashlib.sha256(
            json.dumps(fps, sort_keys=True).encode()).hexdigest(),
        "gate": "PASS",
    }


def _pyproject_version() -> str:
    return re.search(r'(?m)^version\s*=\s*"([^"]+)"',
                     (_ROOT / "pyproject.toml").read_text()).group(1)


def _run_check(receipt_path) -> int:
    return subprocess.run(
        [sys.executable, str(_CHECK), "--receipt", str(receipt_path)],
        capture_output=True, text=True, cwd=str(_ROOT)).returncode


@pytest.mark.skipif(not (_ROOT / ".git").exists(), reason="needs a git checkout")
class TestM5FingerprintPrecondition:
    def _head(self):
        return subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()

    def test_valid_receipt_passes(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps(_valid_receipt(self._head())))
        assert _run_check(p) == 0, "a fresh, authentic receipt must pass"

    def test_missing_receipt_blocks(self, tmp_path):
        assert _run_check(tmp_path / "nope.json") != 0

    def test_stale_receipt_blocks(self, tmp_path):
        # an ancestor commit where csrc/ differs from HEAD (volet C tip, before
        # volet B's csrc changes). If unavailable, skip.
        anc = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-list", "--max-count=8", "HEAD"],
            capture_output=True, text=True).stdout.split()
        stale_sha = None
        for sha in anc[1:]:
            d = subprocess.run(
                ["git", "-C", str(_ROOT), "diff", "--name-only", sha, "HEAD",
                 "--", "csrc", "mlx_mfa"], capture_output=True, text=True).stdout.strip()
            if d:
                stale_sha = sha
                break
        if not stale_sha:
            pytest.skip("no ancestor with csrc/mlx_mfa drift in recent history")
        p = tmp_path / "r.json"
        p.write_text(json.dumps(_valid_receipt(stale_sha)))
        assert _run_check(p) != 0, "a stale receipt (source changed since) must block"

    def test_tampered_fingerprints_block(self, tmp_path):
        r = _valid_receipt(self._head())
        r["fingerprints"]["dense_D128_auto_nax"] = 0.999  # break the hash binding
        p = tmp_path / "r.json"
        p.write_text(json.dumps(r))
        assert _run_check(p) != 0, "a tampered receipt must block"

    def test_not_pass_blocks(self, tmp_path):
        r = _valid_receipt(self._head())
        r["gate"] = "FAIL"
        p = tmp_path / "r.json"
        p.write_text(json.dumps(r))
        assert _run_check(p) != 0
