"""Volet E — docs/knobs reconciliation locks.

Closes CC-08..13, CC-16, CX-09 with bite-proven guards:
- knob registry: every KNOWN_KNOBS entry appears in real (non-comment) code;
  the removed ghosts are gone; ENV_VARS-removed knobs warn "removed" not "typo".
- perf/routing claims: the reconciled README/doc numbers are present and the
  stale/mis-denominated ones are absent.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from mlx_mfa import _knobs

_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


# ── Knob registry coverage (CC-12) ───────────────────────────────────────────

# Read in _knobs.py itself (excluded from the code scan), so allowlisted.
_KNOB_READ_ALLOWLIST = {"MFA_KNOB_STRICT"}
_LIT = re.compile(r"(MFA_[A-Z0-9_]+|MLX_MFA_[A-Z0-9_]+)")


def _names_in_noncomment_code() -> set:
    """Every MFA_*/MLX_MFA_* token appearing in a NON-comment line across
    csrc/ + mlx_mfa/ (excluding _knobs.py, the registry itself). A comment-only
    knob (a ghost) never appears here."""
    out: set = set()
    for d in ("csrc", "mlx_mfa"):
        for f in (_ROOT / d).rglob("*"):
            if f.suffix not in (".cpp", ".hpp", ".mm", ".py") or f.name == "_knobs.py":
                continue
            for line in f.read_text(errors="ignore").splitlines():
                code = line.split("#", 1)[0] if f.suffix == ".py" else line.split("//", 1)[0]
                out.update(_LIT.findall(code))
    return out


def test_every_known_knob_appears_in_real_code():
    """CC-12: no comment-only ghost may sit in KNOWN_KNOBS — every entry must
    appear in a non-comment line (a real read/alias/dispatch site)."""
    appears = _names_in_noncomment_code()
    prefixes = tuple(_knobs.PREFIX_KNOBS)
    missing = sorted(
        k for k in _knobs.KNOWN_KNOBS
        if k not in appears and k not in _KNOB_READ_ALLOWLIST
        and not any(k.startswith(p) for p in prefixes))
    assert not missing, (
        f"KNOWN_KNOBS entries with NO non-comment appearance (ghost knobs — they "
        f"advertise tuning DOF that does not exist): {missing}. Remove them or add "
        f"a real read site.")


def test_knob_coverage_bites():
    """Prove the ghost check bites: a knob that appears nowhere is flagged."""
    appears = _names_in_noncomment_code()
    fake = "MFA_NONEXISTENT_GHOST_XYZ"
    assert fake not in appears
    checked = set(_knobs.KNOWN_KNOBS) | {fake}
    prefixes = tuple(_knobs.PREFIX_KNOBS)
    missing = [k for k in checked
               if k not in appears and k not in _KNOB_READ_ALLOWLIST
               and not any(k.startswith(p) for p in prefixes)]
    assert fake in missing, "the ghost-knob check failed to flag a no-appearance knob"


@pytest.mark.parametrize("ghost", [
    "MFA_CONV3D_MPP", "MFA_V6_BHND", "MFA_V6_MATMUL_EXEC_SG",
    "MFA_REQUIRE_MSL4", "MFA_SUPPORTED_DTYPES", "MFA_SUPPORTED_HDIMS",
])
def test_removed_ghosts_absent_from_registry(ghost):
    """CC-12: the six non-env ghosts are no longer accepted as valid knobs."""
    assert ghost not in _knobs.KNOWN_KNOBS


# ── ENV_VARS removed-knob parity (CC-13) ─────────────────────────────────────

@pytest.mark.parametrize("knob", [
    "MFA_ENABLE_V4", "MFA_ENABLE_V5",
    "MFA_V5_FORCE_BK", "MFA_V5_FORCE_BD_TILE", "MFA_V5_FORCE_BQ", "MFA_V5_FORCE_WM",
])
def test_env_vars_removed_knob_in_removed_registry(knob):
    """CC-13: every V4/V5 knob documented as removed in ENV_VARS.md is in
    REMOVED_KNOBS (so strict-validate says 'removed', not 'typo')."""
    assert knob in _read("ENV_VARS.md"), f"{knob} should be documented in ENV_VARS.md"
    assert knob in _knobs.REMOVED_KNOBS, (
        f"{knob} is documented removed in ENV_VARS.md but missing from REMOVED_KNOBS "
        f"→ it would miswarn as a typo instead of 'removed'.")


def test_removed_knob_warns_removed_not_typo(monkeypatch):
    """CC-13 behavioural: MFA_ENABLE_V5 warns 'removed', not 'typo'."""
    monkeypatch.setenv("MFA_KNOB_STRICT", "1")
    monkeypatch.setenv("MFA_ENABLE_V5", "1")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _knobs.validate_env(strict=True)
    msgs = " ".join(str(x.message) for x in w)
    assert "MFA_ENABLE_V5" in msgs and "REMOVED" in msgs.upper(), msgs
    assert "typo" not in msgs.lower().split("mfa_enable_v5")[1][:80] if "mfa_enable_v5" in msgs.lower() else True


# ── Perf / routing claim reconciliation (CC-08/09/10/11, CX-09) ──────────────

def test_conv3d_denominator_reconciled():
    """CC-08/09: README must give the public conv3d win vs mx.conv_general
    (1.64× SeedVR2) and must NOT attribute the 2.3-2.5× im2col number to
    mx.conv_general."""
    r = _read("README.md")
    assert "1.64×" in r and "conv_general" in r, "lost the provenanced 1.64× vs conv_general"
    # the false denominator pairing "2.3–2.5×, bf16 1.4–2.7× vs `mx.conv_general`" must be gone
    assert "1.4–2.7× vs\n`mx.conv_general`" not in r and \
           "2.3–2.5×, bf16 1.4–2.7× vs `mx.conv_general`" not in r, \
        "README still attributes the 2.3-2.5× im2col figure to mx.conv_general (CC-08)"


def test_v6_backward_block_reconciled():
    """CC-10: the stale '1.81-1.82× via MFA_ENABLE_V6_BACKWARD for D=64' block is
    replaced by the default-on 2.16-3.05× truth."""
    r = _read("README.md")
    assert "2.16–3.05×" in r, "lost the canonical default-on V6-backward number"
    assert "1.81-1.82× faster end-to-end\nbackward" not in r, \
        "README still presents the withdrawn 1.81-1.82× D=64-via-env figure (CC-10)"


def test_window_21x_provenanced():
    """CC-11: the bare '~21×' is replaced by the measured 20.8×/18.4× cell."""
    r = _read("README.md")
    fc = _read("docs/reference/FEATURE_COVERAGE.md")
    assert "20.8×" in r and "18.4×" in r, "README sliding-window claim not stamped"
    assert "up to ~**21x**" not in r, "bare unprovenanced ~21x remains in README"
    assert "up to 21x speedup" not in fc, "bare unprovenanced 21x remains in FEATURE_COVERAGE"


def test_flash_attention_docstring_routing_matches_dispatch_map():
    """CX-09: the docstring no longer claims dense D=64/128 'routes to MFA' — it
    matches the dispatch-map (D=64 → SDPA, D=128 N<2048 → SDPA)."""
    import inspect
    import mlx_mfa
    doc = inspect.getdoc(mlx_mfa.flash_attention) or ""
    assert "Dense causal D=64/128 routes to MFA" not in doc, \
        "docstring still overstates MFA routing (CX-09)"
    assert "D=64" in doc and "SDPA" in doc and "dispatch-map" in doc
