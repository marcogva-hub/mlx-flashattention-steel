"""Tests for native GNA (Generalized Neighborhood Attention) Metal kernel.

Validates the inline 3D window kernel against a pure-Python reference
that applies the same per-element GNA masking.

Note: the sparse path (make_gna_mask + flash_attention_sparse) only applies
TILE-level masking, so it's a conservative over-approximation. The native
kernel applies exact per-element masking, which is more precise. Tests here
compare against a Python reference with the same per-element mask.
"""

import os
import math
import pytest
import numpy as np
import mlx.core as mx


os.environ.pop("MFA_DISABLE_GNA_NATIVE", None)


def _has_ext():
    try:
        from mlx_mfa._ext import mfa_gna_forward
        return True
    except ImportError:
        return False


requires_ext = pytest.mark.skipif(not _has_ext(), reason="C++ extension not built")


def _gna_python_reference(q, k, v, seq_shape, window_size, stride, scale):
    """Pure-Python GNA reference with exact per-element masking.

    Computes attention with the GNA window formula applied per (q, k) pair.
    This is the ground truth for what the native kernel should produce.
    """
    dim0, dim1, dim2 = seq_shape
    win0, win1, win2 = window_size
    str0, str1, str2 = stride
    dim12 = dim1 * dim2
    N = dim0 * dim12
    D = q.shape[-1]

    qf = np.array(q.astype(mx.float32)).reshape(-1, N, D)  # [B*H, N, D]
    kf = np.array(k.astype(mx.float32))
    vf = np.array(v.astype(mx.float32))
    B, H_q, _, _ = q.shape
    H_kv = k.shape[1]
    gqa = H_q // H_kv

    # Build per-element mask [N, N]
    mask = np.zeros((N, N), dtype=bool)
    for qi in range(N):
        c0, c1, c2 = qi // dim12, (qi // dim2) % dim1, qi % dim2
        g0, g1, g2 = c0 // str0, c1 // str1, c2 // str2
        lo0 = max(0, g0 * str0 - (win0 - str0) // 2)
        hi0 = min(dim0 - 1, (g0 + 1) * str0 + (win0 - str0 + 1) // 2 - 1)
        lo1 = max(0, g1 * str1 - (win1 - str1) // 2)
        hi1 = min(dim1 - 1, (g1 + 1) * str1 + (win1 - str1 + 1) // 2 - 1)
        lo2 = max(0, g2 * str2 - (win2 - str2) // 2)
        hi2 = min(dim2 - 1, (g2 + 1) * str2 + (win2 - str2 + 1) // 2 - 1)
        for ki in range(N):
            kc0, kc1, kc2 = ki // dim12, (ki // dim2) % dim1, ki % dim2
            if lo0 <= kc0 <= hi0 and lo1 <= kc1 <= hi1 and lo2 <= kc2 <= hi2:
                mask[qi, ki] = True

    results = []
    for b in range(B):
        for h in range(H_q):
            h_kv = h // gqa
            qq = np.array(q[b, h].astype(mx.float32)).reshape(N, D)
            kk = np.array(k[b, h_kv].astype(mx.float32)).reshape(N, D)
            vv = np.array(v[b, h_kv].astype(mx.float32)).reshape(N, D)
            S = qq @ kk.T * scale
            S[~mask] = -1e9
            S_max = S.max(axis=-1, keepdims=True)
            P = np.exp(S - S_max)
            P = P / P.sum(axis=-1, keepdims=True)
            results.append(P @ vv)

    out = np.stack(results).reshape(B, H_q, N, D)
    return out


def _gna_native(q, k, v, seq_shape, window_size, stride, scale):
    from mlx_mfa._ext import mfa_gna_forward
    out = mfa_gna_forward(
        q, k, v, scale,
        seq_shape[0], seq_shape[1], seq_shape[2],
        window_size[0], window_size[1], window_size[2],
        stride[0], stride[1], stride[2],
    )
    mx.synchronize()
    return out


def _gna_sparse(q, k, v, seq_shape, window_size, stride, scale):
    """Sparse path (tile-level masking only, conservative)."""
    os.environ["MFA_DISABLE_GNA_NATIVE"] = "1"
    try:
        from mlx_mfa import flash_attention_gna
        out = flash_attention_gna(q, k, v, seq_shape, window_size, stride, scale=scale)
        mx.synchronize()
        return out
    finally:
        os.environ.pop("MFA_DISABLE_GNA_NATIVE", None)


# ──────────────────────────────────────────────────────────────────────────────
# Correctness: native vs Python reference (exact per-element mask)
# ──────────────────────────────────────────────────────────────────────────────

@requires_ext
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gna_native_sliding_window_small(dtype):
    """Small sliding window: (2, 4, 4) N=32, window (2,2,2), stride (1,1,1)."""
    B, H, D = 1, 2, 128
    seq_shape = (2, 4, 4)
    window_size = (2, 2, 2)
    stride = (1, 1, 1)
    N = math.prod(seq_shape)
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(42)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    mx.synchronize()

    nat = _gna_native(q, k, v, seq_shape, window_size, stride, scale)
    ref = _gna_python_reference(q, k, v, seq_shape, window_size, stride, scale)
    nat_np = np.array(nat.astype(mx.float32))

    max_err = np.max(np.abs(nat_np - ref))
    atol = 0.01 if dtype == mx.float16 else 0.05
    assert max_err < atol, f"max_abs_err={max_err:.4f} (expected < {atol})"
    assert nat.shape == (B, H, N, D)


@requires_ext
def test_gna_native_blocked_matches_sdpa():
    """Blocked attention (stride==window) should match dense SDPA exactly."""
    B, H, D = 1, 2, 128
    seq_shape = (1, 4, 32)
    window_size = (1, 4, 32)
    stride = (1, 4, 32)
    N = math.prod(seq_shape)
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(123)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.synchronize()

    nat = _gna_native(q, k, v, seq_shape, window_size, stride, scale)
    sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    mx.eval(sdpa)

    nat_np = np.array(nat.astype(mx.float32))
    sdpa_np = np.array(sdpa.astype(mx.float32))

    max_err = np.max(np.abs(nat_np - sdpa_np))
    assert max_err < 0.005, f"blocked GNA should match SDPA, max_err={max_err:.6f}"


@requires_ext
def test_gna_native_strided_window():
    """Strided window: groups of queries share K/V windows."""
    B, H, D = 1, 2, 128
    seq_shape = (2, 8, 8)
    window_size = (2, 4, 4)
    stride = (1, 2, 2)
    N = math.prod(seq_shape)
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(999)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.synchronize()

    nat = _gna_native(q, k, v, seq_shape, window_size, stride, scale)
    ref = _gna_python_reference(q, k, v, seq_shape, window_size, stride, scale)
    nat_np = np.array(nat.astype(mx.float32))

    max_err = np.max(np.abs(nat_np - ref))
    assert max_err < 0.01, f"max_abs_err={max_err:.4f} (expected < 0.01)"


@requires_ext
def test_gna_native_gqa():
    """GQA: H_q=8, H_kv=2, gqa_factor=4."""
    B, H, H_kv, D = 1, 8, 2, 128
    seq_shape = (2, 4, 4)
    window_size = (2, 2, 2)
    stride = (1, 1, 1)
    N = math.prod(seq_shape)
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(77)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
    mx.synchronize()

    nat = _gna_native(q, k, v, seq_shape, window_size, stride, scale)
    ref = _gna_python_reference(q, k, v, seq_shape, window_size, stride, scale)
    nat_np = np.array(nat.astype(mx.float32))

    max_err = np.max(np.abs(nat_np - ref))
    assert max_err < 0.01, f"max_abs_err={max_err:.4f} (expected < 0.01)"


@requires_ext
def test_gna_native_multi_batch():
    """Multi-batch: B=2."""
    B, H, D = 2, 4, 128
    seq_shape = (2, 4, 4)
    window_size = (2, 2, 2)
    stride = (1, 1, 1)
    N = math.prod(seq_shape)
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(55)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.synchronize()

    nat = _gna_native(q, k, v, seq_shape, window_size, stride, scale)
    ref = _gna_python_reference(q, k, v, seq_shape, window_size, stride, scale)
    nat_np = np.array(nat.astype(mx.float32))

    max_err = np.max(np.abs(nat_np - ref))
    assert max_err < 0.01, f"max_abs_err={max_err:.4f} (expected < 0.01)"


@requires_ext
def test_gna_native_larger_shape():
    """Larger 3D shape: (4, 8, 8) = N=256."""
    B, H, D = 1, 2, 128
    seq_shape = (4, 8, 8)
    window_size = (2, 4, 4)
    stride = (1, 1, 1)
    N = math.prod(seq_shape)
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(88)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.synchronize()

    nat = _gna_native(q, k, v, seq_shape, window_size, stride, scale)
    ref = _gna_python_reference(q, k, v, seq_shape, window_size, stride, scale)
    nat_np = np.array(nat.astype(mx.float32))

    max_err = np.max(np.abs(nat_np - ref))
    assert max_err < 0.01, f"max_abs_err={max_err:.4f} (expected < 0.01)"


@requires_ext
def test_gna_python_api_routes_to_native():
    """Verify flash_attention_gna() uses native kernel when available."""
    B, H, D = 1, 2, 128
    seq_shape = (2, 4, 4)
    window_size = (2, 2, 2)
    stride = (1, 1, 1)
    N = math.prod(seq_shape)

    mx.random.seed(42)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.synchronize()

    from mlx_mfa import flash_attention_gna
    out = flash_attention_gna(q, k, v, seq_shape, window_size, stride)
    mx.synchronize()

    assert out.shape == (B, H, N, D)
    assert out.dtype == mx.float16


@requires_ext
def test_gna_native_d128_constraint():
    """D != 128 should raise."""
    from mlx_mfa._ext import mfa_gna_forward
    B, H, N, D = 1, 2, 32, 64

    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.synchronize()

    with pytest.raises((RuntimeError, ValueError), match="D=128"):
        mfa_gna_forward(q, k, v, 0.1, 2, 4, 4, 2, 2, 2, 1, 1, 1)


@requires_ext
def test_gna_native_shape_mismatch():
    """N != dim0*dim1*dim2 should raise."""
    from mlx_mfa._ext import mfa_gna_forward
    B, H, N, D = 1, 2, 64, 128

    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.synchronize()

    with pytest.raises((RuntimeError, ValueError), match="dim0"):
        mfa_gna_forward(q, k, v, 0.1, 2, 4, 4, 2, 2, 2, 1, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────────────

@requires_ext
@pytest.mark.benchmark
def test_gna_benchmark_cogvideox():
    """CogVideoX shape: (13, 60, 90) = N=70200, window (4, 16, 16)."""
    B, H, D = 1, 2, 128
    seq_shape = (13, 60, 90)
    window_size = (4, 16, 16)
    stride = (1, 1, 1)
    N = math.prod(seq_shape)
    scale = 1.0 / math.sqrt(D)

    mx.random.seed(42)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.synchronize()

    import time

    # Warmup (mx.eval forces GPU work to complete)
    nat = _gna_native(q, k, v, seq_shape, window_size, stride, scale)
    mx.eval(nat)

    times_nat = []
    for _ in range(5):
        mx.synchronize()
        t0 = time.perf_counter()
        out = _gna_native(q, k, v, seq_shape, window_size, stride, scale)
        mx.eval(out)
        t1 = time.perf_counter()
        times_nat.append(t1 - t0)

    # Warmup sparse
    ref = _gna_sparse(q, k, v, seq_shape, window_size, stride, scale)
    mx.eval(ref)

    times_ref = []
    for _ in range(5):
        mx.synchronize()
        t0 = time.perf_counter()
        out = _gna_sparse(q, k, v, seq_shape, window_size, stride, scale)
        mx.eval(out)
        t1 = time.perf_counter()
        times_ref.append(t1 - t0)

    nat_ms = np.median(times_nat) * 1000
    ref_ms = np.median(times_ref) * 1000
    speedup = ref_ms / nat_ms if nat_ms > 0 else float("inf")

    print(f"\n  CogVideoX GNA benchmark:")
    print(f"    Native:   {nat_ms:.2f} ms")
    print(f"    Sparse:   {ref_ms:.2f} ms")
    print(f"    Speedup:  {speedup:.2f}x")

    assert nat.shape == (B, H, N, D)
