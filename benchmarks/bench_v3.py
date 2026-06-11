"""Benchmark: STEEL V3 vs V2 vs SDPA.

V3 uses separate K_smem + V_smem (2 barriers/iter) instead of V2's shared
KV_smem (4 barriers/iter).  Eligible: D=64 all gens, D=128 M1/M2 (BK=32).

Usage:
    python benchmarks/bench_v3.py
    python benchmarks/bench_v3.py --causal
    python benchmarks/bench_v3.py --head-dim 64 --seq-len 4096 8192
"""

import argparse
import math
import os
import time

import mlx.core as mx

from mlx_mfa.attention import _fallback_sdpa, _mfa_forward


def _measure(fn, n_warmup=8, n_iter=20):
    for _ in range(n_warmup):
        mx.synchronize()
        fn()
        mx.synchronize()
    times = []
    for _ in range(n_iter):
        mx.synchronize()
        t0 = time.perf_counter()
        result = fn()
        mx.eval(result)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000


def benchmark_row(B, H, N, D, causal, dtype, n_warmup, n_iter):
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)
    mx.eval(q, k, v)

    # SDPA baseline
    sdpa_ms = _measure(
        lambda q=q, k=k, v=v, s=scale, c=causal: _fallback_sdpa(q, k, v, s, c),
        n_warmup, n_iter)

    # V2 path (disable V3 so V2 is chosen)
    # Repo review 2026-05: try/finally — an exception inside _measure
    # previously leaked MFA_DISABLE_V3, silently forcing all remaining
    # rows onto the V2 path (wrong benchmark numbers, no error).
    os.environ["MFA_DISABLE_V3"] = "1"
    try:
        v2_ms = _measure(
            lambda q=q, k=k, v=v, s=scale, c=causal: _mfa_forward(q, k, v, s, c),
            n_warmup, n_iter)
    finally:
        os.environ.pop("MFA_DISABLE_V3", None)

    # V3 path (default — V3 checked before V2)
    v3_ms = _measure(
        lambda q=q, k=k, v=v, s=scale, c=causal: _mfa_forward(q, k, v, s, c),
        n_warmup, n_iter)

    return sdpa_ms, v2_ms, v3_ms


def main():
    parser = argparse.ArgumentParser(description="STEEL V3 vs V2 vs SDPA benchmark")
    parser.add_argument("--batch",    type=int, default=2)
    parser.add_argument("--heads",    type=int, default=8)
    parser.add_argument("--head-dim", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seq-len",  type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--causal",   action="store_true")
    parser.add_argument("--dtype",    choices=["f16", "bf16"], default="f16")
    parser.add_argument("--warmup",   type=int, default=8)
    parser.add_argument("--iters",    type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "f16" else mx.bfloat16
    B, H = args.batch, args.heads

    print(f"\nSTEEL V3 vs V2 vs SDPA  --  "
          f"B={B} H={H} causal={args.causal} dtype={args.dtype}")
    print(f"{'D':>5} {'N':>6}  {'SDPA ms':>9} {'V2 ms':>8} {'V3 ms':>8}  "
          f"{'V2/SDPA':>8} {'V3/SDPA':>8} {'V3/V2':>7}")
    print("-" * 72)

    for D in args.head_dim:
        for N in args.seq_len:
            sdpa, v2, v3 = benchmark_row(
                B, H, N, D, args.causal, dtype, args.warmup, args.iters)
            r_v2   = sdpa / v2 if v2 > 0 else float("inf")
            r_v3   = sdpa / v3 if v3 > 0 else float("inf")
            r_v3v2 = v2   / v3 if v3 > 0 else float("inf")
            print(f"{D:>5} {N:>6}  {sdpa:>9.2f} {v2:>8.2f} {v3:>8.2f}  "
                  f"{r_v2:>7.2f}x {r_v3:>7.2f}x {r_v3v2:>6.2f}x")
        if D != args.head_dim[-1]:
            print()

    print("\nNote: speedup > 1.0x = faster than SDPA/V2. V3 ineligible for D=256.")


if __name__ == "__main__":
    main()
