"""Benchmark: STEEL V2 vs V1 vs MLX SDPA.

Runs three kernel paths across D=64/128/256, causal/non-causal, f16/bf16:
  - SDPA   : mx.fast.scaled_dot_product_attention (MLX baseline)
  - MFA V1 : STEEL V1 (MFA_DISABLE_V2=1 env var forces V1 path)
  - MFA V2 : STEEL V2 (default, sequential K/V phases, 2x BK)

Usage:
    python benchmarks/bench_v2.py
    python benchmarks/bench_v2.py --head-dim 128 --causal
    python benchmarks/bench_v2.py --dtype bf16 --seq-len 4096 8192
"""

import argparse
import math
import os
import time

import mlx.core as mx

from mlx_mfa.attention import _fallback_sdpa, _mfa_forward


def _measure(fn, n_warmup=8, n_iter=20):
    for _ in range(n_warmup):
        mx.eval(fn())
    mx.synchronize()
    times = []
    for _ in range(n_iter):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000


def benchmark_row(B, H, N, D, causal, dtype, n_warmup=8, n_iter=20):
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)
    mx.eval(q, k, v)

    # SDPA baseline
    sdpa_ms = _measure(lambda: _fallback_sdpa(q, k, v, scale, causal), n_warmup, n_iter)

    # V1: set MFA_DISABLE_V2 so eval_gpu() bypasses both V2 blocks
    os.environ["MFA_DISABLE_V2"] = "1"
    v1_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), n_warmup, n_iter)
    del os.environ["MFA_DISABLE_V2"]

    # V2: env unset; eligible configs (f16/bf16, D=64/128/256) route to V2
    v2_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), n_warmup, n_iter)

    return sdpa_ms, v1_ms, v2_ms


def main():
    parser = argparse.ArgumentParser(description="STEEL V2 vs V1 vs SDPA benchmark")
    parser.add_argument("--batch",    type=int, default=2)
    parser.add_argument("--heads",    type=int, default=8)
    parser.add_argument("--head-dim", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--seq-len",  type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--causal",   action="store_true")
    parser.add_argument("--dtype",    choices=["f16", "bf16"], default="f16")
    parser.add_argument("--warmup",   type=int, default=8)
    parser.add_argument("--iters",    type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "f16" else mx.bfloat16
    B, H = args.batch, args.heads

    print(f"\nSTEEL V2 vs V1 vs SDPA  --  "
          f"B={B} H={H} causal={args.causal} dtype={args.dtype}")
    print(f"{'D':>5} {'N':>6}  {'SDPA ms':>9} {'V1 ms':>8} {'V2 ms':>8}  "
          f"{'V1/SDPA':>8} {'V2/SDPA':>8} {'V2/V1':>7}")
    print("-" * 72)

    for D in args.head_dim:
        for N in args.seq_len:
            sdpa, v1, v2 = benchmark_row(
                B, H, N, D, args.causal, dtype, args.warmup, args.iters)
            r_v1   = sdpa / v1 if v1 > 0 else float("inf")
            r_v2   = sdpa / v2 if v2 > 0 else float("inf")
            r_v2v1 = v1   / v2 if v2 > 0 else float("inf")
            print(f"{D:>5} {N:>6}  {sdpa:>9.2f} {v1:>8.2f} {v2:>8.2f}  "
                  f"{r_v1:>7.2f}x {r_v2:>7.2f}x {r_v2v1:>6.2f}x")
        if D != args.head_dim[-1]:
            print()

    print("\nNote: speedup > 1.0x = faster than reference (SDPA or V1).")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
