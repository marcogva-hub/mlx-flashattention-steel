"""Volet P7 — LocalHost host-store is dtype-agnostic (fp32 hybrid/offload).

Re-run #4: the persistence axis table is complete and enforced, but the dtype
axis was OVER-restricted on one surface. `LocalHostKVStoreAdapter.put` is a host
BYTE store — the reload path is `fetch()` → `primary.append()` with no cast, no
fp16/bf16 assumption — yet P6 gave it an `(fp16, bf16)` accepted-set. So
`create_decode_runtime(dtype=float32, hybrid_cache=True, hybrid_enable_offload=
True)` — a legal-if-off-spec config that constructs + runs (SDPA fallback) —
failed LATE at `offload` (far from the cause).

P7 fix: the host store enforces ONLY K↔V dtype consistency (the universal rule),
not an accepted-input-dtype set, so it round-trips any consistent dtype. The
fixed-dtype STORAGE caches keep their configured-dtype restriction (they have a
real fixed-dtype buffer; the host store does not).
"""
import warnings
import numpy as np
import mlx.core as mx
import pytest
from mlx_mfa.external_cache import LocalHostKVStoreAdapter
from mlx_mfa.attention import DenseKVCache


def _kv(dk, dv, b=1, h=2, n=1, d=64):
    k = mx.zeros((b, h, n, d), dk)
    v = mx.zeros((b, h, n, d), dv)
    mx.eval(k, v)
    return k, v


# ── fp32 hybrid + offload runs end-to-end with NO late raise ─────────────────────
def test_fp32_hybrid_offload_end_to_end():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # off-spec fp32 → SDPA fallback
        import mlx_mfa
        rt = mlx_mfa.create_decode_runtime(
            backend="dense", B=1, H_q=8, H_kv=8, D=64, dtype=mx.float32,
            max_seq_len=128, hybrid_cache=True, hybrid_enable_offload=True)

        def mk(n):
            a = mx.random.normal((1, 8, n, 64)).astype(mx.float32)
            mx.eval(a)
            return a

        rt.prefill(mk(16), mk(16), mk(16))
        rt.hybrid_offload([0])                    # <- previously raised LATE here
        rt.hybrid_reload([0])
        o = rt.step(mk(1), mk(1), mk(1))
        mx.eval(o)
        assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


# ── host store now accepts any CONSISTENT dtype (fp16/bf16/fp32) ─────────────────
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16, mx.float32])
def test_localhost_accepts_any_consistent_dtype(dt):
    LocalHostKVStoreAdapter().put(0, *_kv(dt, dt))


# ── but K↔V consistency is STILL enforced (only the accepted-set was wrong) ──────
def test_localhost_still_rejects_kv_dtype_mismatch():
    with pytest.raises((ValueError, Exception)):
        LocalHostKVStoreAdapter().put(0, *_kv(mx.float16, mx.bfloat16))


# ── no regression: fixed-dtype STORAGE caches still reject a mismatched dtype ────
def test_storage_cache_still_rejects_mismatched_dtype():
    with pytest.raises((ValueError, Exception)):
        DenseKVCache(1, 2, 64, 32, dtype=mx.float16).append(*_kv(mx.float32, mx.float32))
    DenseKVCache(1, 2, 64, 32, dtype=mx.float16).append(*_kv(mx.float16, mx.float16))
