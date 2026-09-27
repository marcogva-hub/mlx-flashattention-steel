"""R1 / DSP-09 / CRIT-02 (code review 2026-09, P0 PUBLISHED) — `flash_attention` auto must
never hand fp32 to the MFA primitive's legacy fp32 (ccv) kernel.

That kernel (csrc/mfa_attention.cpp dtype_code==2 branch) has NO causal ``qL_off``:
causal N<S is TOP-LEFT aligned -> silently wrong (err ~3) for decode / chunked prefill.
`backend='mfa'` already refused fp32 for exactly this reason, but `auto` reached it
through two side doors:
  (a) mixed dtype (q fp32, k/v f16): ``if _mixed_dtype: use_mfa = True``;
  (b) ``return_lse and _mfa_capable -> use_mfa`` (fp32 is "capable").
Heirs of (b), confirmed by test: `flash_attention_speculative_verify` and
`flash_attention_paged(return_lse=True)` in fp32.

Oracle = fp32 math with the bottom-right causal mask (key j visible to row i iff
j <= i + (S - N)), i.e. SDPA's convention for N <= S. Which-binary = dispatch trace.

N > S (causal): decided 2026-09 — every route follows NAMING.md's canonical
zero-clamp (TestNgtSZeroClamp below; full matrix in test_causal_zero_clamp_convention.py).
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_mfa import (
    flash_attention,
    flash_attention_paged,
    flash_attention_speculative_verify,
    is_mfa_available,
)
from mlx_mfa import _dispatch_trace as dt

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")

_LN2 = math.log(2.0)


def _mask(N, S, qL_off):
    qi = mx.arange(N)[:, None]
    kj = mx.arange(S)[None, :]
    return mx.where(kj <= qi + qL_off, 0.0, float("-inf"))


# fp32 on the M5 GPU is NOT IEEE-fp32 accurate (VERIFIED 2026-09-27, M5 Max, MLX 0.31.2,
# macOS 27.2): vs float64 NumPy, GPU fp32 matmul ~6e-4..1.2e-3 and mx.fast SDPA fp32
# ~4e-4..8e-4, while CPU fp32 is ~5e-7.  vs this CPU oracle, raw
# mx.fast.scaled_dot_product_attention(mask="causal") itself sits at ~2.0-2.5e-3 on O
# (unit-variance inputs, N,S<=513) and the fixed fp32 routes match it exactly (same O
# as the no-lse route); log2-LSE ~2-3.5e-3.  So the oracle runs on the CPU stream and
# fp32 routes are held to MLX's own fp32-GPU floor.  5e-3 still separates the R1 bug
# (err ~2.5-3.4) by ~500x.
FP32_GPU_TOL = 5e-3


def _oracle(q, k, v, causal, *, zero_clamp=False):
    """fp32 O and log2-LSE on the CPU. Bottom-right for N<=S; `zero_clamp` = max(0,S-N)."""
    with mx.stream(mx.cpu):
        q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
        N, S, D = q.shape[2], k.shape[2], q.shape[3]
        scores = (q32 @ mx.swapaxes(k32, -1, -2)) * (D ** -0.5)
        if causal:
            off = max(0, S - N) if zero_clamp else S - N
            scores = scores + _mask(N, S, off)
        O = mx.softmax(scores, axis=-1) @ v32
        L = mx.logsumexp(scores, axis=-1) / _LN2
        mx.eval(O, L)
    return O, L


def _qkv(N, S, D, q_dtype=mx.float32, kv_dtype=mx.float32, B=1, H=2, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H, N, D)).astype(q_dtype)
    k = mx.random.normal((B, H, S, D)).astype(kv_dtype)
    v = mx.random.normal((B, H, S, D)).astype(kv_dtype)
    return q, k, v


def _err(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _terminals(tr):
    return [t[0] for t in tr if not t[1].startswith("[reentrant]")]


SHAPES_N_LE_S = [(40, 100, 64), (1, 100, 64), (1, 513, 128), (33, 64, 128), (64, 64, 64)]


@pytest.mark.parametrize("N,S,D", SHAPES_N_LE_S)
def test_mixed_fp32_q_f16_kv_causal_matches_oracle(N, S, D):
    """DSP-09: q fp32 against an f16 KV cache (prefill AND N=1 decode)."""
    q, k, v = _qkv(N, S, D, q_dtype=mx.float32, kv_dtype=mx.float16, seed=3)
    with dt.capture() as tr:
        out = flash_attention(q, k, v, causal=True)
        mx.eval(out)
    O, _ = _oracle(q, k, v, causal=True)
    assert out.dtype == mx.float32
    assert _err(out, O) < FP32_GPU_TOL, _err(out, O)
    assert "mfa_primitive" not in _terminals(tr), _terminals(tr)


@pytest.mark.parametrize("N,S,D", SHAPES_N_LE_S)
def test_fp32_return_lse_causal_matches_oracle(N, S, D):
    """CRIT-02: fp32 + return_lse, causal N<=S -> O and log2-LSE exact."""
    q, k, v = _qkv(N, S, D, seed=4)
    with dt.capture() as tr:
        out, lse = flash_attention(q, k, v, causal=True, return_lse=True)
        mx.eval(out, lse)
    O, L = _oracle(q, k, v, causal=True)
    assert _err(out, O) < FP32_GPU_TOL and _err(lse, L) < FP32_GPU_TOL, (_err(out, O), _err(lse, L))
    assert "mfa_primitive" not in _terminals(tr), _terminals(tr)


def test_fp32_speculative_verify_matches_oracle():
    """CRIT-02 heir (was DEDUCED): speculative verify always uses return_lse, causal N_draft<S."""
    q, k, v = _qkv(8, 48, 64, seed=5)
    out, lse, _lp = flash_attention_speculative_verify(q, k, v, mx.zeros((1, 8), dtype=mx.int32))
    mx.eval(out, lse)
    O, L = _oracle(q, k, v, causal=True)
    assert _err(out, O) < FP32_GPU_TOL and _err(lse, L) < FP32_GPU_TOL, (_err(out, O), _err(lse, L))


def test_fp32_paged_return_lse_matches_oracle():
    """CRIT-02 heir (was DEDUCED): paged per-sequence return_lse path in fp32."""
    B, H, N, S, D, BS = 1, 2, 8, 48, 64, 16
    q, k, v = _qkv(N, S, D, seed=5)
    kp = k[0].transpose(1, 0, 2).reshape(S // BS, BS, H, D)
    vp = v[0].transpose(1, 0, 2).reshape(S // BS, BS, H, D)
    out, lse = flash_attention_paged(
        q, kp, vp, mx.array([[0, 1, 2]], dtype=mx.int32), mx.array([S], dtype=mx.int32),
        causal=True, block_size=BS, return_lse=True)
    mx.eval(out, lse)
    O, L = _oracle(q, k, v, causal=True)
    assert _err(out, O) < FP32_GPU_TOL and _err(lse, L) < FP32_GPU_TOL, (_err(out, O), _err(lse, L))


@pytest.mark.parametrize("kwargs", [
    dict(causal=True), dict(causal=False), dict(causal=True, return_lse=True),
    dict(causal=True, softcap=30.0),
])
def test_fp32_auto_never_reaches_legacy_primitive(kwargs):
    """Which-binary: no fp32 auto call terminates on the MFA primitive."""
    q, k, v = _qkv(40, 100, 64, seed=6)
    with dt.capture() as tr:
        r = flash_attention(q, k, v, **kwargs)
        mx.eval(r)
    assert "mfa_primitive" not in _terminals(tr), _terminals(tr)


def test_fp32_softcap_preserved():
    """The fp32 guard must not drop softcap (the ccv kernel has none)."""
    q, k, v = _qkv(40, 100, 64, seed=7)
    cap = 5.0
    out = flash_attention(q, k, v, causal=True, softcap=cap)
    with mx.stream(mx.cpu):
        scores = (q @ mx.swapaxes(k, -1, -2)) * (64 ** -0.5)
        scores = cap * mx.tanh(scores / cap) + _mask(40, 100, 60)
        ref = mx.softmax(scores, axis=-1) @ v
        mx.eval(ref)
    assert _err(out, ref) < FP32_GPU_TOL, _err(out, ref)


@pytest.mark.parametrize("q_dt,kv_dt", [(mx.float16, mx.bfloat16), (mx.bfloat16, mx.float16)])
@pytest.mark.parametrize("N,S,causal", [(2048, 2048, False), (40, 100, True), (1, 513, True)])
def test_mixed_low_precision_equals_uniform_call(q_dt, kv_dt, N, S, causal):
    """Mixed dtype must route exactly like the same call with k/v pre-cast to q.dtype."""
    q, k, v = _qkv(N, S, 128, q_dtype=q_dt, kv_dtype=kv_dt, seed=8)
    with dt.capture() as tr_m:
        o_mixed = flash_attention(q, k, v, causal=causal)
        mx.eval(o_mixed)
    with dt.capture() as tr_u:
        o_unif = flash_attention(q, k.astype(q_dt), v.astype(q_dt), causal=causal)
        mx.eval(o_unif)
    assert _terminals(tr_m) == _terminals(tr_u), (_terminals(tr_m), _terminals(tr_u))
    assert _err(o_mixed, o_unif) == 0.0


def test_raw_forward_with_lse_refuses_fp32_causal_n_lt_s():
    """Source-level guard (raw expert entry): fp32 causal N<S is the silent-wrong domain."""
    from mlx_mfa._ext import mfa_forward_with_lse
    q, k, v = _qkv(40, 100, 64, seed=9)
    with pytest.raises(ValueError, match="float32"):
        mfa_forward_with_lse(q, k, v, 64 ** -0.5, True)


@pytest.mark.parametrize("N,S,causal", [(64, 64, True), (40, 100, False)])
def test_raw_forward_with_lse_fp32_outside_domain_still_allowed(N, S, causal):
    """fp32 stays valid on the raw entry outside causal N<S (documented expert surface)."""
    from mlx_mfa._ext import mfa_forward_with_lse
    q, k, v = _qkv(N, S, 64, seed=10)
    O, L = mfa_forward_with_lse(q, k, v, 64 ** -0.5, causal)
    mx.eval(O, L)
    Oref, _ = _oracle(q, k, v, causal=causal)
    assert _err(O, Oref) < 1e-2


class TestNgtSZeroClamp:
    """R1'/CRIT-01 — DECIDED 2026-09 (Marco): N>S causal follows NAMING.md's canonical
    zero-clamp on every route (full matrix: tests/test_causal_zero_clamp_convention.py)."""

    N, S, D = 256, 128, 64

    def test_f16_no_lse_is_zero_clamp(self):
        q, k, v = _qkv(self.N, self.S, self.D, mx.float16, mx.float16, seed=11)
        out = flash_attention(q, k, v, causal=True)
        O, _ = _oracle(q, k, v, causal=True, zero_clamp=True)
        assert _err(out, O) < 1e-2

    def test_f16_return_lse_is_zero_clamp(self):
        q, k, v = _qkv(self.N, self.S, self.D, mx.float16, mx.float16, seed=11)
        out, _ = flash_attention(q, k, v, causal=True, return_lse=True)
        O, _ = _oracle(q, k, v, causal=True, zero_clamp=True)
        assert _err(out, O) < 1e-2

    def test_fp32_return_lse_is_zero_clamp(self):
        q, k, v = _qkv(self.N, self.S, self.D, seed=11)
        out, lse = flash_attention(q, k, v, causal=True, return_lse=True)
        O, L = _oracle(q, k, v, causal=True, zero_clamp=True)
        assert _err(out, O) < FP32_GPU_TOL and _err(lse, L) < FP32_GPU_TOL, (_err(out, O), _err(lse, L))
