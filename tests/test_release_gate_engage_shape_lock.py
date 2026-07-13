"""Anti-recurrence lock for the 2.62.0 Phase-3 release incident.

The release M5/NAX gate (scripts/release_m5_nax_gate.py) has a sparse ENGAGEMENT
fingerprint that MUST engage the NAX sparse kernel (byteΔ > 0 vs masked SDPA). Its
shape is single-sourced as `SPARSE_ENGAGE_SPEC`. This lock asserts that shape still
satisfies the CURRENT routing predicate `_nax_sparse_route_viable` — so if a future
routing/gate narrowing (like `d3836d3 "narrow sparse routing to measured envelope"`,
which moved the OLD N=2048/density-0.51 fingerprint out of the NAX region → silent
SDPA fallback → gate byteΔ==0) excludes the engagement shape, THIS lock fails first,
on any host (it is a pure Python predicate check — no kernel, no M5), BEFORE a Phase-3
gate run, with a message telling the maintainer exactly what to re-choose.

Deliberately NOT M5-gated: it must bite on the M1 CI runner too (the M5 gate itself
runs only on the maintainer's host), so a routing narrowing is caught in ordinary CI.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx

from mlx_mfa.lcsa_nax import _nax_sparse_route_viable

_GATE = Path(__file__).resolve().parent.parent / "scripts" / "release_m5_nax_gate.py"


def _load_engage_spec() -> dict:
    """Single-source the shape from the gate script (no duplication)."""
    spec = importlib.util.spec_from_file_location("_release_gate_mod", _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs imports + defs only; main() is __main__-guarded
    return mod.SPARSE_ENGAGE_SPEC


def test_release_gate_engagement_shape_routes_to_nax():
    S = _load_engage_spec()
    q = mx.zeros((S["B"], S["H"], S["N"], S["D"]), dtype=mx.float16)
    ok = _nax_sparse_route_viable(q, q, S["BT"], S["density"], causal=S["causal"])
    assert ok, (
        f"Release-gate ENGAGEMENT fingerprint {S} no longer routes to NAX "
        f"(_nax_sparse_route_viable returned False) — a routing/gate narrowing excluded it. "
        f"Re-choose the engagement shape to the CENTER of a CURRENT measured winning region "
        f"(see SPARSE_NAX_* constants in mlx_mfa/lcsa_nax.py) and update SPARSE_ENGAGE_SPEC in "
        f"scripts/release_m5_nax_gate.py BEFORE running the Phase-3 gate — otherwise Phase 3 "
        f"would FAIL with a silent SDPA fallback (byteΔ==0), the 2.62.0 incident this lock prevents."
    )
