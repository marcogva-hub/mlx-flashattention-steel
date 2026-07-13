"""Sparse V6NAX tile-config LOCK (sparse-NAX-autotune, M5 Max, 2026-06-18).

The block-sparse V6NAX cooperative-tensor / matmul2d kernel's tile is
STRUCTURALLY PINNED — not a tunable — to BQ=BK=32, WM=2:

  * BQ=BK=32 is fixed by mask-block faithfulness: one 32-wide Q/K block maps to
    exactly one `block_mask` entry (V6NAX sparse eligibility forces block_tile==32).
  * WM=2 is fixed by the cooperative-tensor inner GEMM + cross-simdgroup reduction
    (a latent 2-SG assumption). The divisibility rule BQ%(WM*16)==0 also admits
    WM=1, but a measured WM=1 sweep was BOTH ~3-4x slower at high density AND
    incorrect (err up to 3.0e-2 — a silent Category-A wrong-but-finite result).

Why a *source-fingerprint* lock and not just the correctness lock:
`test_sparse_family_correctness_lock.py` checks V6NAX sparse vs an fp32 oracle at TOL=3e-2.
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
    """The body of sparse_kernel_source_v6nax (where pinned tile constants live)."""
    text = _SRC.read_text()
    start = text.index("std::string sparse_kernel_source_v6nax(")
    end = text.index("\n}", start)
    return text[start:end]


@pytest.mark.parametrize(
    "const,expected",
    [("V6NAX_SPARSE_BQ", 32), ("V6NAX_SPARSE_BK", 32), ("V6NAX_SPARSE_WM", 2)],
)
def test_sparse_v6nax_tile_is_pinned(const, expected):
    """Config-fingerprint: the V6NAX sparse generator bakes the pinned tile constants.
    A drift (e.g. WM back to a tunable, or BK away from 32) fails CI."""
    text = _SRC.read_text()
    shared = {
        "V6NAX_SPARSE_BQ": "kV6NAXSparseBQ",
        "V6NAX_SPARSE_BK": "kV6NAXSparseBK",
        "V6NAX_SPARSE_WM": "kV6NAXSparseWM",
    }[const]
    m = re.search(rf"constexpr int {shared}\s*=\s*(\d+);", text)
    assert m, f"could not find shared sparse tile constant {shared}"
    got = int(m.group(1))
    assert got == expected, (
        f"sparse V6NAX tile drifted: {const}={got}, expected {expected} "
        f"(tile is structurally pinned — re-run the sparse-NAX-autotune before changing)")
    block = _gen_block()
    assert re.search(rf"const int {const}\s*=\s*{shared};", block), (
        f"MSL generator no longer consumes shared host/source constant {shared}")


def test_sparse_v6nax_host_dispatch_uses_shared_wm():
    """Both O-only and O+LSE host grids consume the same WM as generated MSL."""
    text = _SRC.read_text()
    assert text.count("kV6NAXSparseWM * 32") == 2
    assert "const int V6NAX_SPARSE_WM = 2" not in text
    assert "constexpr int V6NAX_SPARSE_WM = 2" not in text


def test_sparse_v6nax_no_wm_override_knob():
    """No env knob may re-expose WM: the only valid value is 2, and the divisibility
    rule alone (which admits WM=1) is necessary-but-not-sufficient. A reintroduced
    sparse WM override would let users select the broken WM=1 path."""
    text = _SRC.read_text()
    for knob in ("MFA_LCSA_V2_WM", "MFA_LCSA_V6NAX_SPARSE_WM"):
        assert knob not in text, (
            f"{knob} override appeared — WM=1 is incorrect (err ~3e-2); "
            "do not expose WM as a knob")


def test_sparse_v6nax_default_is_correct_binary():
    """Runtime check: the dispatched default V6NAX kernel produces the correct (WM=2)
    binary — faithful to an fp32 SDPA-mask oracle, NOT the WM=1 ~1e-2 regime."""
    try:
        from mlx_mfa import get_device_info
        from mlx_mfa._ext import sparse_attention_forward
        if not bool(get_device_info().get("is_m5_plus")):
            pytest.skip("sparse V6NAX kernel is M5+-only")
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
        f"default sparse V6NAX err={err:.2e} — expected ~7e-6 (correct WM=2). "
        f"An err near ~1e-2 means the broken WM=1 tile is live.")


def test_sparse_kernel_version_canonical_aliases_match_legacy_names():
    """Public aliases stay compatible at the C++ binding boundary."""
    try:
        from mlx_mfa import get_device_info
        from mlx_mfa._ext import sparse_attention_forward
        if not bool(get_device_info().get("is_m5_plus")):
            pytest.skip("sparse alias lock asserts M5+ kernels")
    except Exception:
        pytest.skip("extension / M5 detection unavailable")

    mx.random.seed(1)
    B, H, L, D = 1, 1, 2048, 64
    sc = 1.0 / math.sqrt(D)
    f = lambda: (mx.random.uniform(-1, 1, (B, H, L, D)) * 0.01).astype(mx.float16)
    q, k, v = f(), f(), f()
    rng = np.random.default_rng(1)
    m = rng.random((L // 32, L // 32)) < 0.2
    m[:, 0] = True
    mask = mx.array(m)
    mx.eval(q, k, v, mask)

    o_v2 = sparse_attention_forward(q, k, v, mask, 32, False, sc, "v2")
    o_v6 = sparse_attention_forward(q, k, v, mask, 32, False, sc, "v6nax_sparse")
    o_v1 = sparse_attention_forward(q, k, v, mask, 32, False, sc, "v1")
    o_scalar = sparse_attention_forward(q, k, v, mask, 32, False, sc, "scalar_fallback")
    mx.eval(o_v2, o_v6, o_v1, o_scalar)

    d_v6 = float(mx.max(mx.abs(o_v2.astype(mx.float32) - o_v6.astype(mx.float32))).item())
    d_scalar = float(mx.max(mx.abs(o_v1.astype(mx.float32) - o_scalar.astype(mx.float32))).item())
    assert d_v6 == 0.0
    assert d_scalar == 0.0
