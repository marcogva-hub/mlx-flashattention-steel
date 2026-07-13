"""bf16 block-sparse → V2/NAX lock (bf16-sparse-v2 fix, M5 Max, 2026-06-18).

Before this fix, the V2 (cooperative-tensor / matmul2d) sparse kernel was gated to
fp16 only (`&& is_f16` in sparse_attention_forward), so bf16 inputs SILENTLY fell
back to the V1 scalar kernel — up to ~50x slower than plain SDPA-with-mask (a bf16
sparse user was strictly worse off than not using the sparse path). The V2 `mma` is
templated on the input dtype with fp32 accumulation (CType=float) and the generator
already emits `using T = bfloat`, so bf16 is just T=bfloat — the gate was a Phase-1.2
deferral, not a kernel limitation.

This lock asserts (lesson #11 independent fp32 oracle + lesson #14 runtime binary
fingerprint):
  1. bf16 V2 is CORRECT vs an independent fp32 SDPA-mask oracle (both D=64,128).
  2. bf16 reaches the REAL V2 kernel via the PUBLIC path (flash_attention_sparse) —
     byteΔ>0 vs SDPA (a real distinct kernel) AND byteΔ>0 vs the forced-V1 binary
     (it is V2, NOT the old V1 fallback). A drift back to V1/SDPA fails CI.
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import flash_attention_sparse
from mlx_mfa.attention import _get_is_m5_plus_cached

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(),
    reason="bf16 sparse V2/NAX lock asserts M5+ kernels")

try:
    from mlx_mfa._ext import sparse_attention_forward
    _HAVE = True
except Exception:
    _HAVE = False


def _delta(a, b):
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _sym_mask(N, density=0.2, bt=32, seed=0):
    """Symmetric NB×NB block mask (bt_q==bt_k==32) → routes to the V2 sparse kernel."""
    NB = N // bt
    rng = np.random.default_rng(seed)
    m = rng.random((NB, NB)) < density
    m[:, 0] = True  # no all-empty Q-row
    return mx.array(m)


def _fp32_oracle(q, k, v, mask, scale, bt=32):
    NQ, NK = mask.shape
    N, S = q.shape[2], k.shape[2]
    full = np.repeat(np.repeat(np.array(mask), N // NQ, 0), S // NK, 1)
    add = mx.array(np.where(full, 0.0, -1e30).astype(np.float32))[None, None]
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    return mx.softmax((qf @ kf.transpose(0, 1, 3, 2)) * scale + add, axis=-1) @ vf


@pytest.mark.skipif(not _HAVE, reason="extension required")
@pytest.mark.parametrize("D", [64, 128])
def test_bf16_v2_correct_vs_fp32_oracle(D):
    """Forced V2, bf16: faithful to an independent fp32 oracle (NOT SDPA, NOT V1)."""
    mx.random.seed(0)
    B, H, N = 1, 8, 2048
    sc = 1.0 / math.sqrt(D)
    f = lambda: (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.bfloat16)
    q, k, v = f(), f(), f()
    mask = _sym_mask(N)
    mx.eval(q, k, v, mask)
    o = sparse_attention_forward(q, k, v, mask, 32, False, sc, "v2")
    ref = _fp32_oracle(q, k, v, mask, sc)
    mx.eval(o, ref)
    assert bool(mx.all(mx.isfinite(o.astype(mx.float32))).item()), "non-finite bf16 V2 output"
    err = _delta(o, ref)
    assert err < 5e-3, f"bf16 V2 err={err:.2e} vs fp32 oracle (expected ~3e-5; bf16 floor)"


@pytest.mark.skipif(not _HAVE, reason="extension required")
@pytest.mark.parametrize("D", [64, 128])
def test_bf16_reaches_real_v2_not_v1_fallback(D):
    """Runtime binary fingerprint: forced-V2 bf16 is a DISTINCT kernel from forced-V1
    (byteΔ>0). Δ==0 would mean V2 silently fell back to V1 — the bug this fix closes."""
    mx.random.seed(1)
    B, H, N = 1, 8, 2048
    sc = 1.0 / math.sqrt(D)
    f = lambda: (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.bfloat16)
    q, k, v = f(), f(), f()
    mask = _sym_mask(N)
    mx.eval(q, k, v, mask)
    o_v2 = sparse_attention_forward(q, k, v, mask, 32, False, sc, "v2")
    o_v1 = sparse_attention_forward(q, k, v, mask, 32, False, sc, "v1")
    d = _delta(o_v2, o_v1)
    assert d > 1e-7, (
        f"bf16 forced-V2 is byte-identical to forced-V1 (Δ={d:.2e}) — V2 eligibility "
        f"silently dropped bf16 to the V1 fallback. The `&& is_f16` gate regressed.")


@pytest.mark.skipif(not _HAVE, reason="extension required")
def test_fp32_does_not_reach_v2_bound():
    """Phase-0 dtype bound: removing `is_f16` from V2 eligibility must NOT let fp32
    leak into the fp16/bf16-only V2 kernel. fp32 is bounded BEFORE eligibility
    (the dtype-validity throw), at BOTH the C++ and public levels (Rule 8)."""
    mx.random.seed(3)
    B, H, N, D = 1, 4, 2048, 128
    sc = 1.0 / math.sqrt(D)
    f = lambda: (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float32)
    q, k, v = f(), f(), f()
    mask = _sym_mask(N)
    mx.eval(q, k, v, mask)
    # C++ level: the _ext entry raises on fp32 (cannot reach V2 eligibility)
    with pytest.raises((RuntimeError, ValueError)):
        o = sparse_attention_forward(q, k, v, mask, 32, False, sc, "v2")
        mx.eval(o)
    # public level: flash_attention_sparse raises on fp32 (Rule 8 bound)
    with pytest.raises((RuntimeError, ValueError)):
        o = flash_attention_sparse(q, k, v, mask, scale=sc)
        mx.eval(o)


def test_lse_variant_reaches_v6nax_for_both_dtypes():
    """The BT32 D64/128 LSE surface must engage V6NAX for both dtypes."""
    try:
        from mlx_mfa.lcsa_nax import sparse_attention_nax_with_lse
    except Exception:
        pytest.skip("sparse LSE variant unavailable")
    D, N = 128, 2048
    sc = 1.0 / math.sqrt(D)
    for dt in (mx.float16, mx.bfloat16):
        mx.random.seed(4)
        f = lambda: (mx.random.uniform(-1, 1, (1, 8, N, D)) * 0.1).astype(dt)
        q, k, v = f(), f(), f()
        mask = _sym_mask(N)
        mx.eval(q, k, v, mask)
        from mlx_mfa import _dispatch_trace as dtrace
        with dtrace.capture() as trace:
            o, L = sparse_attention_nax_with_lse(
                q, k, v, mask, block_tile=32, scale=sc)
        ref = _fp32_oracle(q, k, v, mask, sc)
        mx.eval(o, L, ref)
        err = _delta(o, ref)
        assert err < 5e-3, f"{dt} LSE-variant O wrong vs fp32 oracle (Δ={err:.2e})"
        assert trace and trace[-1][0] == "v6nax_sparse_lse", trace


@pytest.mark.parametrize(
    "D,H,N,density,expected",
    [
        (64, 4, 2048, 0.25, "sdpa"),
        (128, 12, 4096, 0.15, "v6nax_sparse"),
    ],
)
def test_bf16_public_path_matches_hardened_gate(D, H, N, density, expected):
    """Public bf16 routes only its measured winning region to V6NAX."""
    mx.random.seed(2)
    B = 1
    sc = 1.0 / math.sqrt(D)
    f = lambda: (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.bfloat16)
    q, k, v = f(), f(), f()
    mask = _sym_mask(N, density=density)
    mx.eval(q, k, v, mask)
    with dtrace.capture() as trace:
        o = flash_attention_sparse(q, k, v, mask, scale=sc)
        mx.eval(o)
    ref = _fp32_oracle(q, k, v, mask, sc)
    d_oracle = _delta(o, ref)
    full = np.repeat(np.repeat(np.array(mask), N // mask.shape[0], 0), N // mask.shape[1], 1)
    bias = mx.array(np.where(full, 0.0, -1e9).astype(np.float32))[None, None].astype(mx.bfloat16)
    sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=bias)
    d_sdpa = _delta(o, sdpa)
    assert d_oracle < 5e-3, f"bf16 public-path sparse wrong vs fp32 oracle (Δ={d_oracle:.2e})"
    terminal = [event for event in trace if not event[1].startswith("[reentrant]")]
    assert terminal and terminal[-1][0] == expected, trace
    if expected == "v6nax_sparse":
        assert d_sdpa > 0.0, "eligible bf16 cell did not run a distinct V6NAX binary"
    else:
        assert d_sdpa == 0.0, "excluded bf16 cell did not delegate byte-identically"
