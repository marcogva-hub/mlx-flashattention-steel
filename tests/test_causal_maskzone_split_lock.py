"""Engaged regression lock for the causal mask-zone (RC-A) and flash-decode
empty-split NaN (RC-B) fixes — the converged correctness defect surfaced by the
forensic diagnosis and escalated to a **default-path CRITICAL**.

Root causes (both kernel-math, fixed in csrc/):
  * RC-A — the per-element causal mask zone gated on `kb >= kb_lim - (BQ+BK-1)/BK`
    masked only the FINAL K-tile (integer division collapses to 1).  Correct only
    when the causal diagonal sits at the sequence tail (`N==S` or `qL_off%BK==0`);
    for `qL_off%BK != 0` the diagonal spans two K-tiles and the second-to-last
    went unmasked → silently-wrong attention.  Shared by flash-decode, V2, V1; now
    uses `mfa_causal_mask_zone_gate` = `(tile*BQ+qL_off)/BK` (qL_off-aware, exact).
  * RC-B — `compute_num_splits` / `NK_per_split` could leave an empty trailing
    split whose 0/0 pO normalization + the reduce's 0*NaN produced all-NaN output
    for decode tails with odd `NK=ceil(S/32)`.  Now: no empty splits + reduce
    guards (skip non-finite LSE / zero-weight splits).

This lock is ENGAGED: it forces `backend="mfa"` and asserts the MFA primitive
actually ran (`_dispatch_trace` terminal == "mfa_primitive", never an SDPA
fallback — closing the CX-06/CC-17 engagement gap), then compares to an
INDEPENDENT fp64 softmax oracle.  Pre-fix these shapes erred 0.9–3.2 or NaN; the
asserts bite hard.

Default-path CRITICAL re-verified directly: D=128 causal fp16 S=4096, N≈S with
`qL_off%32 != 0`, via `backend="auto"`.
"""
import math
import numpy as np
import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa.attention import flash_attention
from mlx_mfa import _dispatch_trace as dt

pytestmark = pytest.mark.skipif(
    not mlx_mfa.has_nax(),
    reason="STEEL/flash-decode kernels require NAX/_ext (has_nax() False -> SDPA fallback)",
)

_DT = {mx.float16: "f16", mx.bfloat16: "bf16"}
_TOL = {mx.float16: 1.5e-2, mx.bfloat16: 3.5e-2}


def _oracle(q, k, v, causal):
    """Independent fp64 reference. Decode/cross convention: query i attends keys
    [0 .. (S-N)+i] when causal (qL_off = S-N)."""
    q = np.asarray(q, np.float64); k = np.asarray(k, np.float64); v = np.asarray(v, np.float64)
    B, H, N, D = q.shape; S = k.shape[2]
    sc = 1.0 / math.sqrt(D)
    out = np.zeros((B, H, N, D), np.float64)
    for b in range(B):
        for h in range(H):
            s = (q[b, h] @ k[b, h].T) * sc
            if causal:
                off = S - N
                for i in range(N):
                    s[i, off + i + 1:] = -1e30
            m = s.max(1, keepdims=True)
            p = np.exp(s - m); p /= p.sum(1, keepdims=True)
            out[b, h] = p @ v[b, h]
    return out


def _run_forced(N, S, D, dtype, causal, *, seed=0, H=8, B=1):
    rng = np.random.default_rng(seed)
    qn = rng.standard_normal((B, H, N, D)).astype(np.float32)
    kn = rng.standard_normal((B, H, S, D)).astype(np.float32)
    vn = rng.standard_normal((B, H, S, D)).astype(np.float32)
    q, k, v = (mx.array(x).astype(dtype) for x in (qn, kn, vn))
    with dt.capture() as tr:
        o = flash_attention(q, k, v, causal=causal, backend="mfa"); mx.eval(o)
    term = tr[-1][0] if tr else None
    ref = _oracle(qn, kn, vn, causal)
    o_np = np.asarray(o.astype(mx.float32))
    return o_np, ref, term


def _assert_engaged_and_correct(N, S, D, dtype, causal, **kw):
    o, ref, term = _run_forced(N, S, D, dtype, causal, **kw)
    # (b) path-entered: the MFA primitive ran (not an SDPA fallback) — engagement.
    assert term == "mfa_primitive", f"expected mfa_primitive, got {term} (N={N},S={S},D={D})"
    # (a) output sanity vs independent fp64 oracle.
    assert np.all(np.isfinite(o)), f"non-finite output N={N} S={S} D={D} {_DT[dtype]} caus={causal}"
    e = np.abs(o - ref).max()
    assert e < _TOL[dtype], f"err {e:.3e} > tol N={N} S={S} D={D} {_DT[dtype]} caus={causal}"
    return e


# ── RC-A: small-Nq aligned (flash-decode) — was 5.7e-3..2e-2; controls N=1,5,8 ──
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("D", [64, 128])
def test_rca_small_nq_aligned_causal(N, D, dtype):
    _assert_engaged_and_correct(N, 1024, D, dtype, True)


# ── RC-A: cross-attention N<S non-aligned — was 0.9..1.3 ──
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N,S", [(37, 53), (44, 77), (100, 777), (53, 53)])  # 53/53 = N==S control
@pytest.mark.parametrize("D", [64, 128])
def test_rca_cross_attention_causal(N, S, D, dtype):
    _assert_engaged_and_correct(N, S, D, dtype, True)


# ── RC-A: full qL_off % BK sweep at S=4096 (the exact trigger axis) ──
@pytest.mark.parametrize("r", list(range(0, 32)))  # qL_off = r ; r%32==0 is the aligned control
def test_rca_qloff_mod_bk_sweep(r):
    N = 4096 - r
    _assert_engaged_and_correct(N, 4096, 128, mx.float16, True, H=4)


# ── RC-B: decode odd-NK tails (were all-NaN) + even-NK controls ──
@pytest.mark.parametrize("N", [1, 2, 3, 4])
@pytest.mark.parametrize("S", [257, 288, 513, 319, 512, 640])  # odd-NK (NaN) + even-NK controls
def test_rcb_decode_tails_no_nan(N, S):
    _assert_engaged_and_correct(N, S, 128, mx.float16, True)


# ── Edge cases preserved: non-causal decode must stay correct (no over-masking) ──
@pytest.mark.parametrize("N", [1, 2, 4])
@pytest.mark.parametrize("S", [1024, 257, 513])
@pytest.mark.parametrize("D", [64, 128])
def test_noncausal_decode_preserved(N, S, D):
    _assert_engaged_and_correct(N, S, D, mx.float16, False)


# ── Default-path CRITICAL: backend="auto" on the escalated shape must be correct ──
@pytest.mark.parametrize("N", [4095, 4094, 4093, 4090, 4064])
def test_default_path_critical_auto(N):
    S, D = 4096, 128
    rng = np.random.default_rng(7)
    qn = rng.standard_normal((1, 8, N, D)).astype(np.float32)
    kn = rng.standard_normal((1, 8, S, D)).astype(np.float32)
    vn = rng.standard_normal((1, 8, S, D)).astype(np.float32)
    q, k, v = (mx.array(x).astype(mx.float16) for x in (qn, kn, vn))
    o = flash_attention(q, k, v, causal=True, backend="auto"); mx.eval(o)
    ref = _oracle(qn, kn, vn, True)
    o_np = np.asarray(o.astype(mx.float32))
    assert np.all(np.isfinite(o_np))
    # pre-fix: err 2-3 for qL_off%32 != 0.  post-fix: ~1e-3.
    assert np.abs(o_np - ref).max() < 1.5e-2, f"default-path CRITICAL N={N} qLoff={S-N}"
