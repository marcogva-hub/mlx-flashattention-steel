#!/usr/bin/env python3
"""Profile V6 fixed overhead on FlashVSR-dense.

Goal (Axe 9): identify why V6 tuned (1.48ms) is still 0.62× SDPA (0.91ms)
on the smallest production shape. Fixed overhead probably dominates.
"""
import time
import math
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward, shader_cache_size

# FlashVSR-dense
B, H, N, D = 1, 10, 4096, 64
mx.random.seed(42)
q = mx.random.normal((B, H, N, D)).astype(mx.float16)
k = mx.random.normal((B, H, N, D)).astype(mx.float16)
v = mx.random.normal((B, H, N, D)).astype(mx.float16)
mx.eval(q, k, v)
scale = 1.0 / math.sqrt(D)

# Step 1: warmup so all caches are populated
for _ in range(10):
    out, _ = v6_nax_forward(q, k, v, False)
    mx.eval(out)
print(f"Shader cache size after warmup: {shader_cache_size()}")
print()

# Step 2: time isolated phases
# 2a: just the Python entry + return (allocations only)
import sys
# audit H7/H-09 phantom-bench gate (run-at-import bench)
from _bench_guard import require_accel_or_die as _phantom_gate
_phantom_gate(__file__)

# Time pure dispatch: Python -> C++ -> dispatch -> wait -> return
N_calls = 100
mx.synchronize()
t0 = time.perf_counter()
for _ in range(N_calls):
    out, _ = v6_nax_forward(q, k, v, False)
    mx.eval(out)
mx.synchronize()
t_total = (time.perf_counter() - t0) * 1000 / N_calls
print(f"V6 NAX full call (eval included): {t_total:.4f} ms")

# Time without mx.eval (lazy)
mx.synchronize()
t0 = time.perf_counter()
for _ in range(N_calls):
    out, _ = v6_nax_forward(q, k, v, False)
mx.synchronize()  # final flush
t_lazy = (time.perf_counter() - t0) * 1000 / N_calls
print(f"V6 NAX dispatch only (no per-call eval, final sync): {t_lazy:.4f} ms")

# Compare with SDPA
mx.synchronize()
t0 = time.perf_counter()
for _ in range(N_calls):
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    mx.eval(out)
mx.synchronize()
t_sdpa = (time.perf_counter() - t0) * 1000 / N_calls
print(f"\nSDPA full call (eval included): {t_sdpa:.4f} ms")

# Just SDPA without eval
mx.synchronize()
t0 = time.perf_counter()
for _ in range(N_calls):
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
mx.synchronize()
t_sdpa_lazy = (time.perf_counter() - t0) * 1000 / N_calls
print(f"SDPA dispatch only: {t_sdpa_lazy:.4f} ms")

# Compute overhead
# t_total = t_lazy + per_call_eval_overhead + actual_kernel_compute
# actual_kernel = t_total - some baseline. Hard to isolate without Metal trace.

# Comparison
print(f"\n--- Comparison ---")
print(f"V6 / SDPA full:    {t_total/t_sdpa:.2f}x")
print(f"V6 / SDPA dispatch:{t_lazy/t_sdpa_lazy:.2f}x")

# Time individual operations to understand allocator costs
print("\n--- Sub-operation timing ---")

# Just allocations
mx.synchronize()
t0 = time.perf_counter()
for _ in range(N_calls):
    o = mx.zeros((B, H, N, D), dtype=mx.float16)
    mx.eval(o)
mx.synchronize()
t_alloc = (time.perf_counter() - t0) * 1000 / N_calls
print(f"Just allocate+zero output (B={B} H={H} N={N} D={D} f16): {t_alloc:.4f} ms")
