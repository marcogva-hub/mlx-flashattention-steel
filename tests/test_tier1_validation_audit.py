"""Tier-1 regression locks for the secondary-surface correctness/validation
audit items (CC-04..09, CX-03, fp32 forced-mfa refusal).

Each test reproduces the previously silent-wrong / unvalidated behaviour as a
loud failure (raise/warn) or — for the quant items — proves correctness with a
round-trip / fp32 oracle (Lesson #11), not isfinite-only.
"""
import math
import numpy as np
import mlx.core as mx
import pytest

import mlx_mfa


# ── CC-04: HybridKVCache unsupported policy raises ─────────────────────────
class _DummyCache:
    def __init__(self):
        self.k = None


def test_cc04_hybrid_unsupported_policy_raises():
    from mlx_mfa.kv_cache import HybridKVCache
    for bad in ("lfu", "fifo", "manual", "mru"):
        with pytest.raises(ValueError, match="policy"):
            HybridKVCache(_DummyCache(), policy=bad)


def test_cc04_hybrid_lru_ok():
    from mlx_mfa.kv_cache import HybridKVCache
    assert HybridKVCache(_DummyCache(), policy="lru").policy == "lru"
    assert HybridKVCache(_DummyCache(), policy="LRU").policy == "lru"  # case-insensitive


# ── CC-05: turboquant pack/unpack round-trip + cross-bits raise ────────────
@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("n", [10, 32, 33, 64, 100])
def test_cc05_pack_unpack_roundtrip(bits, n):
    from mlx_mfa.turboquant import pack_indices, unpack_indices
    rng = np.random.default_rng(bits * 100 + n)
    idx = mx.array(rng.integers(0, 2 ** bits, size=n).astype(np.uint8))
    out = unpack_indices(pack_indices(idx, bits), n, bits)
    mx.eval(out)
    assert np.array_equal(np.asarray(idx), np.asarray(out)[:n])


def test_cc05_cross_bits_raises():
    from mlx_mfa.turboquant import pack_indices, unpack_indices
    rng = np.random.default_rng(0)
    packed = pack_indices(mx.array(rng.integers(0, 4, 64).astype(np.uint8)), 2)
    with pytest.raises(ValueError, match="bit-width|expected"):
        unpack_indices(packed, 64, 4)  # packed at 2 bits, unpacked at 4


# ── CC-08: turboquant unknown dtype raises (no silent fp16) ────────────────
def test_cc08_compress_unsupported_dtype_raises():
    from mlx_mfa.turboquant import turboquant_compress
    x = mx.zeros((1, 1, 32, 64), dtype=mx.int8)  # int8 not in {fp16,bf16,fp32}
    with pytest.raises(ValueError, match="unsupported dtype"):
        turboquant_compress(x, bits=3)


def test_cc08_compress_fp16_ok():
    from mlx_mfa.turboquant import turboquant_compress, turboquant_decompress
    x = mx.random.normal((1, 1, 32, 64)).astype(mx.float16)
    out = turboquant_decompress(turboquant_compress(x, bits=3))
    assert out.dtype == mx.float16


# ── CC-06: SVDQuant input validation ───────────────────────────────────────
def test_cc06_svdquant_valid():
    from mlx_mfa.svdquant.linear import SVDQuantLinear
    m = SVDQuantLinear(256, 512, bits=4, group_size=64, rank=32)
    assert m.weight.shape == (512, 32)


@pytest.mark.parametrize("args,kw", [
    ((100, 16), dict(bits=4, group_size=64)),   # in_features not divisible
    ((8, 8), dict(bits=0)),                       # div-by-zero guard
    ((8, 8), dict(group_size=0)),                 # div-by-zero guard
    ((256, 512), dict(rank=-1)),                  # negative rank
])
def test_cc06_svdquant_invalid_raises(args, kw):
    from mlx_mfa.svdquant.linear import SVDQuantLinear
    with pytest.raises(ValueError):
        SVDQuantLinear(*args, **kw)


# ── CC-07: dequantize block_size consistency + fp32 oracle ─────────────────
def test_cc07_dequantize_correct_blocksize_oracle():
    from mlx_mfa.quantize import quantize_per_block, dequantize
    x = mx.array(np.random.default_rng(0).standard_normal((1, 2, 128, 16)).astype(np.float16))
    xi, sc = quantize_per_block(x, block_size=32)
    deq = dequantize(xi, sc, block_size=32)
    mx.eval(deq)
    # int8 per-block quant error is small; just confirm it tracks the reference.
    assert float(mx.max(mx.abs(deq.astype(mx.float32) - x.astype(mx.float32))).item()) < 0.1


def test_cc07_dequantize_wrong_blocksize_raises():
    from mlx_mfa.quantize import quantize_per_block, dequantize
    x = mx.array(np.random.default_rng(0).standard_normal((1, 2, 128, 16)).astype(np.float16))
    xi, sc = quantize_per_block(x, block_size=32)
    with pytest.raises(ValueError, match="block_size"):
        dequantize(xi, sc, block_size=64)  # inconsistent with scale's N_blocks


# ── CC-09: off-spec inference context warns (loud fallback) ────────────────
def test_cc09_offspec_context_warns():
    import mlx_mfa.inference as inf
    inf._offspec_warned.clear()
    with pytest.warns(RuntimeWarning, match="off-spec"):
        inf.InferenceContext(B=1, H_kv=4, D=77, max_seq_len=32)  # D=77 off-spec


def test_cc09_onspec_context_no_warn():
    import warnings
    import mlx_mfa.inference as inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning would fail
        inf.InferenceContext(B=1, H_kv=4, D=128, max_seq_len=32)  # on-spec, no warn


def test_cc09_strict_raises(monkeypatch):
    import mlx_mfa.inference as inf
    monkeypatch.setenv("MFA_REQUIRE_NAX", "1")
    with pytest.raises(ValueError, match="off-spec"):
        inf.InferenceContext(B=1, H_kv=4, D=77, max_seq_len=32)


# ── CX-03: raw _ext.mfa_forward_with_lse validation ────────────────────────
@pytest.mark.skipif(not mlx_mfa.has_nax(), reason="raw _ext binding needs the kernel")
def test_cx03_raw_binding_rejects_mismatch():
    ext = pytest.importorskip("mlx_mfa._ext")
    q = mx.random.normal((1, 8, 16, 64)).astype(mx.float16)
    k = mx.random.normal((2, 8, 16, 64)).astype(mx.float16)  # batch mismatch
    v = mx.random.normal((2, 8, 16, 64)).astype(mx.float16)
    with pytest.raises(Exception):  # std::invalid_argument -> Python exception
        o, l = ext.mfa_forward_with_lse(q, k, v, 1.0 / 8.0, False)
        mx.eval(o)


@pytest.mark.skipif(not mlx_mfa.has_nax(), reason="raw _ext binding needs the kernel")
def test_cx03_raw_binding_valid_finite():
    ext = pytest.importorskip("mlx_mfa._ext")
    q = mx.random.normal((1, 8, 64, 64)).astype(mx.float16)
    k = mx.random.normal((1, 8, 64, 64)).astype(mx.float16)
    v = mx.random.normal((1, 8, 64, 64)).astype(mx.float16)
    o, l = ext.mfa_forward_with_lse(q, k, v, 1.0 / 8.0, False)
    mx.eval(o, l)
    assert bool(mx.all(mx.isfinite(o)).item())


# ── fp32 forced-mfa refusal; auto+fp32 still correct (SDPA) ─────────────────
def test_fp32_forced_mfa_raises():
    from mlx_mfa.attention import flash_attention
    q = mx.random.normal((1, 4, 8, 64)).astype(mx.float32)
    k = mx.random.normal((1, 4, 8, 64)).astype(mx.float32)
    v = mx.random.normal((1, 4, 8, 64)).astype(mx.float32)
    with pytest.raises(ValueError, match="float32"):
        flash_attention(q, k, v, causal=True, backend="mfa")


def test_fp32_auto_still_correct():
    from mlx_mfa.attention import flash_attention
    rng = np.random.default_rng(0)
    qn = rng.standard_normal((1, 4, 8, 64)).astype(np.float32)
    kn = rng.standard_normal((1, 4, 8, 64)).astype(np.float32)
    vn = rng.standard_normal((1, 4, 8, 64)).astype(np.float32)
    out = flash_attention(mx.array(qn), mx.array(kn), mx.array(vn), causal=True, backend="auto")
    mx.eval(out)
    # fp32 oracle (square causal, top-left)
    D = 64; sc = 1.0 / math.sqrt(D); o = np.zeros_like(qn)
    for h in range(4):
        s = (qn[0, h] @ kn[0, h].T) * sc
        for i in range(8):
            s[i, i + 1:] = -1e30
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        o[0, h] = p @ vn[0, h]
    assert np.abs(np.asarray(out.astype(mx.float32)) - o).max() < 1e-4
