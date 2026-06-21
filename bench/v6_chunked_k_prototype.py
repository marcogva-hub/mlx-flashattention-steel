"""Python-level chunked-K prototype.

Tests whether per-chunk K dispatching with LSE-weighted combination
yields any measurable speedup on V6 NAX (which already streams K via
register-resident cooperative tensors).

If gain < 3% → V6's streaming architecture already captures the
benefit, no need for C++ infrastructure. Document and skip.
"""
import math, time, statistics
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward
# audit H7/H-09 phantom-bench gate (run-at-import bench)
from _bench_guard import require_accel_or_die as _phantom_gate
_phantom_gate(__file__)

# SeedVR2-large: the candidate shape (N=111375 > 65536 threshold)
B, H, N, D = 1, 20, 111375, 128
WARMUP, ITERS = 2, 5  # tight budget

mx.random.seed(0)
q = mx.random.normal((B, H, N, D)).astype(mx.float16)
k = mx.random.normal((B, H, N, D)).astype(mx.float16)
v = mx.random.normal((B, H, N, D)).astype(mx.float16)
mx.async_eval(q, k, v); mx.synchronize()


def baseline_v6(q, k, v):
    o, lse = v6_nax_forward(q, k, v, False)
    return o, lse


def chunked_v6(q, k, v, chunk_size=32768):
    """Streaming chunked-K via per-chunk V6 + LSE-weighted online combine."""
    Nkv = k.shape[2]
    chunks = []
    pos = 0
    while pos < Nkv:
        end = min(pos + chunk_size, Nkv)
        chunks.append((pos, end))
        pos = end

    O_acc = None
    LSE_acc = None
    for (s, e) in chunks:
        # Slice + contiguous K, V for this chunk
        k_c = mx.contiguous(k[:, :, s:e, :])
        v_c = mx.contiguous(v[:, :, s:e, :])
        O_i, LSE_i = v6_nax_forward(q, k_c, v_c, False)
        if O_acc is None:
            O_acc = O_i
            LSE_acc = LSE_i
        else:
            # Streaming combine:
            #   LSE_new = max(LSE_acc, LSE_i) + log(exp(LSE_acc - max) + exp(LSE_i - max))
            #   alpha = exp(LSE_acc - LSE_new)  (rescale accumulated O)
            #   beta  = exp(LSE_i - LSE_new)    (weight new chunk)
            #   O_acc = alpha * O_acc + beta * O_i
            LSE_max = mx.maximum(LSE_acc, LSE_i)
            exp_acc = mx.exp(LSE_acc - LSE_max)
            exp_i = mx.exp(LSE_i - LSE_max)
            Z = exp_acc + exp_i
            LSE_new = LSE_max + mx.log(Z)
            alpha = mx.expand_dims(exp_acc / Z, axis=-1)  # broadcast over D
            beta = mx.expand_dims(exp_i / Z, axis=-1)
            O_acc = alpha.astype(mx.float16) * O_acc + beta.astype(mx.float16) * O_i
            LSE_acc = LSE_new
    return O_acc, LSE_acc


# Warmup
print("Warming up...")
for _ in range(WARMUP):
    o, _ = baseline_v6(q, k, v); mx.async_eval(o); mx.synchronize()
    o, _ = chunked_v6(q, k, v); mx.async_eval(o); mx.synchronize()

# Correctness check
print("Correctness check...")
o_base, lse_base = baseline_v6(q, k, v)
o_chunk, lse_chunk = chunked_v6(q, k, v)
mx.async_eval(o_base, lse_base, o_chunk, lse_chunk); mx.synchronize()
import numpy as np
b = np.asarray(o_base).astype(np.float32)
c = np.asarray(o_chunk).astype(np.float32)
diff = b - c
print(f"  Max abs diff: {np.abs(diff).max():.6e}")
print(f"  RMSE: {float(np.sqrt((diff*diff).mean())):.6e}")

# Timing
print("\nBenchmark (5 iters each)...")
def time_fn(fn):
    times = []
    for _ in range(ITERS):
        mx.synchronize()
        t0 = time.perf_counter()
        o, _ = fn()
        mx.async_eval(o); mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times)//2]

t_base = time_fn(lambda: baseline_v6(q, k, v))
t_chunk32k = time_fn(lambda: chunked_v6(q, k, v, 32768))
t_chunk16k = time_fn(lambda: chunked_v6(q, k, v, 16384))
t_chunk64k = time_fn(lambda: chunked_v6(q, k, v, 65536))
print(f"  baseline V6:           {t_base:.2f} ms")
print(f"  chunked V6 (16K chunk): {t_chunk16k:.2f} ms ({100*(t_chunk16k/t_base-1):+.1f}%)")
print(f"  chunked V6 (32K chunk): {t_chunk32k:.2f} ms ({100*(t_chunk32k/t_base-1):+.1f}%)")
print(f"  chunked V6 (64K chunk): {t_chunk64k:.2f} ms ({100*(t_chunk64k/t_base-1):+.1f}%)")
