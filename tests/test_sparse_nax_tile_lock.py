"""Sparse V2/NAX tile-config LOCK (sparse-NAX-autotune, M5 Max, 2026-06-18).

The block-sparse V2 (cooperative-tensor / matmul2d) kernel's tile is STRUCTURALLY
PINNED — not a tunable — to BQ=BK=32, WM=2:

  * BQ=BK=32 is fixed by mask-block faithfulness: one 32-wide Q/K block maps to
    exactly one `block_mask` entry (V2 eligibility forces block_tile==32).
  * WM=2 is fixed by the cooperative-tensor inner GEMM + cross-simdgroup reduction
    (a latent 2-SG assumption). The divisibility rule BQ%(WM*16)==0 also admits
    WM=1, but a measured WM=1 sweep was BOTH ~3-4x slower at high density AND
    incorrect (err up to 3.0e-2 — a silent Category-A wrong-but-finite result).

Why a *source-fingerprint* lock and not just the correctness lock:
`test_sparse_family_correctness_lock.py` checks V2 vs an fp32 oracle at TOL=3e-2.
The broken WM=1 config lands AT ~3.0e-2 — right on that boundary — so the math lock
alone would NOT reliably catch a WM regression. This test asserts the pinned tile
constants directly in the generator source, so any drift fails CI and forces a
re-justification (with a fresh autotune) rather than silently shipping a slow/wrong
tile. See docs/lcsa-nax/sparse-nax-autotune-results.md.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

_SRC = Path(__file__).resolve().parent.parent / "csrc" / "mfa_sparse_attention.cpp"


def _gen_block() -> str:
    """The body of sparse_kernel_source_v2 (where the V2 tile constants live)."""
    text = _SRC.read_text()
    start = text.index("std::string sparse_kernel_source_v2(")
    end = text.index("\n}", start)
    return text[start:end]


@pytest.mark.parametrize("const,expected", [("V2_BQ", 32), ("V2_BK", 32), ("V2_WM", 2)])
def test_sparse_v2_tile_is_pinned(const, expected):
    """Config-fingerprint: the V2 sparse generator bakes the pinned tile constants.
    A drift (e.g. WM back to a tunable, or BK away from 32) fails CI."""
    block = _gen_block()
    m = re.search(rf"const int {const}\s*=\s*(\d+);", block)
    assert m, f"could not find `const int {const} = ...;` in sparse_kernel_source_v2"
    got = int(m.group(1))
    assert got == expected, (
        f"sparse V2 tile drifted: {const}={got}, expected {expected} "
        f"(tile is structurally pinned — re-run the sparse-NAX-autotune before changing)")


def test_sparse_v2_no_wm_override_knob():
    """No env knob may re-expose WM: the only valid value is 2, and the divisibility
    rule alone (which admits WM=1) is necessary-but-not-sufficient. A re-introduced
    MFA_LCSA_V2_WM override would let users select the broken WM=1 path."""
    assert "MFA_LCSA_V2_WM" not in _SRC.read_text(), (
        "MFA_LCSA_V2_WM override re-appeared — WM=1 is incorrect (err ~3e-2); "
        "do not expose WM as a knob")


def test_sparse_v2_default_is_correct_binary():
    """Runtime check: the dispatched default V2 kernel produces the correct (WM=2)
    binary — faithful to an fp32 SDPA-mask oracle, NOT the WM=1 ~1e-2 regime."""
    try:
        from mlx_mfa import get_device_info
        from mlx_mfa._ext import sparse_attention_forward
        if not bool(get_device_info().get("is_m5_plus")):
            pytest.skip("sparse V2 NAX kernel is M5+-only")
    except Exception:
        pytest.skip("extension / M5 detection unavailable")

    mx.random.seed(0)
    B, H, L, D = 1, 8, 2048, 128
    sc = 1.0 / math.sqrt(D)
    f = lambda: (mx.random.uniform(-1, 1, (B, H, L, D)) * 0.1).astype(mx.float16)
    q, k, v = f(), f(), f()
    rng = np.random.default_rng(0)
    m = rng.random((L // 32, L // 32)) < 0.2
    m[:, 0] = True
    mask = mx.array(m)
    mx.eval(q, k, v, mask)

    full = np.repeat(np.repeat(np.array(mask), 32, 0), 32, 1)
    add = mx.array(np.where(full, 0.0, -1e30).astype(np.float32))[None, None]
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    ref = mx.softmax((qf @ kf.transpose(0, 1, 3, 2)) * sc + add, axis=-1) @ vf
    O = sparse_attention_forward(q, k, v, mask, 32, False, sc, "v2")
    mx.eval(ref, O)
    err = float(np.abs(np.array(O.astype(mx.float32)) - np.array(ref)).max())
    assert err < 1e-3, (
        f"default sparse V2 err={err:.2e} — expected ~7e-6 (correct WM=2). "
        f"An err near ~1e-2 means the broken WM=1 tile is live.")
