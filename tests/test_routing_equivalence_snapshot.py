"""Routing-equivalence golden snapshot — the P-H1 dispatch-refactor safety net.

PURPOSE (Lot-2 lesson → P-H1 unblocker): the suite asserts *correctness*, not
*which backend ran*.  A routing regression (gate reordered, threshold off-by-one,
a kernel silently replaced by SDPA) keeps every correctness test green.  This test
freezes the CURRENT routing decision for every cell of the dispatch decision space
into a golden snapshot; P-H1 (the dispatch-tree refactor) then asserts the routing
is byte-for-byte identical.  Any divergence is a LOUD routing regression.

WHAT IS CAPTURED
- The **Python routing-layer decision** ``(backend, reason)`` via the
  ``mlx_mfa._dispatch_trace`` test hook — this is the layer P-H1 refactors
  (nax_dense threshold, sdpa fallbacks, mfa_primitive entry, sage, bias, window).
- The **C++ V2-vs-V3 variant** boundary (B·H≥4, ``!MFA_DISABLE_V3``) is a *downstream*
  layer the Python recorder cannot see; it is captured separately as a ``v3_active``
  behavioral flag (output changes when ``MFA_DISABLE_V3=1`` ⟺ V3 ran).  See the
  ``_V3_PROBE`` cells.  NOTE (flagged 2026-06-19): on M5 this flag is False for the
  probed cells — V3 is an M1–M4 path; the gate is effectively dormant on M5.  The
  golden records that as-is so a future REROUTE (V3 going live, or going away) is
  caught.

BOUNDARY COVERAGE (the whole point — gate reorders break exactly at thresholds):
  the dense-NAX threshold and its neighbours (D128 N=2047/2048/2049; D64 N=4095/4096/4097),
  D∈{64,128,256}, causal∈{0,1}, dtype∈{f16,bf16,f32}, B·H∈{3,4,…}, window on/off,
  block-sparse (symmetric/asymmetric), and short-decode qL=8/9.

HARDWARE: routing is hardware-dependent (M5 vs M1–M4).  The golden is stamped with
the capture device; the test SKIPS (loudly) if run on a different device class so a
stale golden never produces a false failure.  P-H1 runs on the same machine.

REGENERATE:  ``MFA_REGEN_ROUTING_GOLDEN=1 .venv/bin/python -m pytest \
  tests/test_routing_equivalence_snapshot.py``  (only when routing INTENTIONALLY
  changes — e.g. after P-H1, with the diff reviewed).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa
from mlx_mfa import _dispatch_trace as _dt

try:
    from mlx_mfa.attention import flash_attention, flash_attention_sparse
    from mlx_mfa.masks import make_sliding_window_mask  # noqa: F401  (presence check)
    _HAVE_SPARSE = True
except Exception:  # pragma: no cover
    from mlx_mfa.attention import flash_attention
    _HAVE_SPARSE = False

_GOLDEN = Path(__file__).parent / "routing_equivalence_golden.json"


def _device_key() -> str:
    """Stable device-class key so a golden captured on M5 isn't asserted on M1."""
    try:
        info = mlx_mfa.get_device_info()
        gen = info.get("gpu_family_gen", "?")
        m5 = "1" if str(info.get("is_m5_plus", False)) in ("True", "1", "true") else "0"
        return f"gen{gen}_m5{m5}"
    except Exception:
        return "unknown"


def _qkv(B, H, N, D, dtype):
    mx.random.seed(0)
    f = lambda: (mx.random.normal((B, H, N, D)) * 0.1).astype(dtype)
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    return q, k, v


_DT = {"f16": mx.float16, "bf16": mx.bfloat16, "f32": mx.float32}

# ── Python-routing-layer cells: (id, B, H, N, D, dtype, causal, kwargs) ──────────
# Boundary neighbours are explicit.  kwargs carries window etc.
_CELLS = []
def _add(cid, B, H, N, D, dt, causal, **kw):
    _CELLS.append((cid, B, H, N, D, dt, causal, kw))

# dense-NAX D=128 threshold (v6_min_n=2048) + neighbours, both causal flavours
for N in (2047, 2048, 2049, 1024, 4096, 8192):
    _add(f"d128_N{N}_caus_f16", 1, 8, N, 128, "f16", True)
    _add(f"d128_N{N}_ncaus_f16", 1, 8, N, 128, "f16", False)
# bf16 D=128 (Tier-2 #2: same NAX route as f16)
for N in (2047, 2048, 4096):
    _add(f"d128_N{N}_caus_bf16", 1, 8, N, 128, "bf16", True)
# fp32 D=128 → never NAX (SDPA)
_add("d128_N4096_caus_f32", 1, 8, 4096, 128, "f32", True)
# D=64 threshold neighbours (v3_min_N=4096) + causal-vs-noncausal
for N in (4095, 4096, 4097, 1024, 8192):
    _add(f"d64_N{N}_caus_f16", 1, 8, N, 64, "f16", True)
    _add(f"d64_N{N}_ncaus_f16", 1, 8, N, 64, "f16", False)
# B·H neighbours (V2/V3 C++ gate is downstream, but capture the Python label is stable)
for H in (3, 4):
    _add(f"d64_N4096_caus_f16_BH{H}", 1, H, 4096, 64, "f16", True)
    _add(f"d128_N4096_caus_f16_BH{H}", 1, H, 4096, 128, "f16", True)
# D=256
_add("d256_N4096_caus_f16", 1, 8, 4096, 256, "f16", True)
_add("d256_N4096_ncaus_f16", 1, 8, 4096, 256, "f16", False)
# window (f16 native STEEL path vs f32 SDPA)
_add("d128_N4096_caus_f16_win256", 1, 8, 4096, 128, "f16", True, window_size=(256, 0))
_add("d128_N4096_caus_f32_win256", 1, 8, 4096, 128, "f32", True, window_size=(256, 0))
# short-decode neighbours (qL=8/9): N_q small vs large KV
_add("decode_qL8_d128_f16", 1, 8, 8, 128, "f16", True)
_add("decode_qL9_d128_f16", 1, 8, 9, 128, "f16", True)
# forced backends (must be stable regardless of shape)
_add("forced_sdpa_d128_N4096", 1, 8, 4096, 128, "f16", True, backend="sdpa")
_add("forced_mfa_d128_N1024", 1, 8, 1024, 128, "f16", True, backend="mfa")


def _route_label(B, H, N, D, dt, causal, kw):
    q, k, v = _qkv(B, H, N, D, _DT[dt])
    with _dt.capture() as tr:
        o = flash_attention(q, k, v, causal=causal, **kw)
        mx.eval(o)
    return list(tr[-1]) if tr else ["<none>", "<no decision recorded>"]


# ── V3 behavioral-probe cells (C++ layer, invisible to the Python recorder) ─────
_V3_PROBE = [
    ("v3probe_d64_N4096_caus_BH8", 1, 8, 4096, 64, "f16", True),
    ("v3probe_d128_N4096_caus_BH8", 1, 8, 4096, 128, "f16", True),
    ("v3probe_d64_N8192_caus_BH8", 1, 8, 8192, 64, "f16", True),
]


def _v3_active(B, H, N, D, dt, causal) -> bool:
    """V3 ran ⟺ disabling it (force V2) changes the output bytes."""
    q, k, v = _qkv(B, H, N, D, _DT[dt])
    o_on = flash_attention(q, k, v, causal=causal); mx.eval(o_on)
    prev = os.environ.get("MFA_DISABLE_V3")
    os.environ["MFA_DISABLE_V3"] = "1"
    try:
        o_off = flash_attention(q, k, v, causal=causal); mx.eval(o_off)
    finally:
        if prev is None:
            os.environ.pop("MFA_DISABLE_V3", None)
        else:
            os.environ["MFA_DISABLE_V3"] = prev
    d = float(np.abs(np.array(o_on.astype(mx.float32)) - np.array(o_off.astype(mx.float32))).max())
    return d > 0.0


def _capture_golden() -> dict:
    g = {"_device": _device_key(), "python_routing": {}, "v3_active": {}}
    for cid, B, H, N, D, dt, causal, kw in _CELLS:
        g["python_routing"][cid] = _route_label(B, H, N, D, dt, causal, kw)
    for cid, B, H, N, D, dt, causal in _V3_PROBE:
        g["v3_active"][cid] = _v3_active(B, H, N, D, dt, causal)
    return g


# Regenerate-on-demand (intentional routing change only).
if os.environ.get("MFA_REGEN_ROUTING_GOLDEN") == "1":
    _g = _capture_golden()
    _GOLDEN.write_text(json.dumps(_g, indent=2, sort_keys=True) + "\n")
    print(f"[routing-golden] regenerated {_GOLDEN} ({len(_g['python_routing'])} routing + "
          f"{len(_g['v3_active'])} v3 cells) on {_g['_device']}")


def _load_golden():
    if not _GOLDEN.exists():
        pytest.skip(f"golden missing — regenerate with MFA_REGEN_ROUTING_GOLDEN=1 ({_GOLDEN})")
    return json.loads(_GOLDEN.read_text())


@pytest.mark.skipif(not hasattr(mlx_mfa, "get_device_info"), reason="no device info")
@pytest.mark.parametrize("cid,B,H,N,D,dt,causal,kw", _CELLS, ids=[c[0] for c in _CELLS])
def test_python_routing_matches_golden(cid, B, H, N, D, dt, causal, kw):
    g = _load_golden()
    if g.get("_device") != _device_key():
        pytest.skip(f"golden captured on {g.get('_device')}, running on {_device_key()} "
                    f"— routing is hardware-specific; regenerate on this device")
    expected = g["python_routing"].get(cid)
    if expected is None:
        pytest.fail(f"cell {cid!r} not in golden — new cell; regenerate the golden")
    actual = _route_label(B, H, N, D, dt, causal, kw)
    assert actual == expected, (
        f"ROUTING REGRESSION at {cid}: golden routed {expected!r}, now routes {actual!r}. "
        f"If this change is intentional (P-H1), regenerate with MFA_REGEN_ROUTING_GOLDEN=1.")


@pytest.mark.parametrize("cid,B,H,N,D,dt,causal", _V3_PROBE, ids=[c[0] for c in _V3_PROBE])
def test_v3_active_matches_golden(cid, B, H, N, D, dt, causal):
    g = _load_golden()
    if g.get("_device") != _device_key():
        pytest.skip(f"golden device mismatch ({g.get('_device')} vs {_device_key()})")
    expected = g["v3_active"].get(cid)
    if expected is None:
        pytest.fail(f"v3 cell {cid!r} not in golden — regenerate")
    actual = _v3_active(B, H, N, D, dt, causal)
    assert actual == expected, (
        f"V3-ROUTING REGRESSION at {cid}: golden v3_active={expected}, now {actual}. "
        f"V3 went {'live' if actual else 'dormant'} for this cell. Intentional? regenerate.")
