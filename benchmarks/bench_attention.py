"""Benchmark: mlx-mfa vs MLX SDPA."""

import argparse
import math
import time

import mlx.core as mx

from mlx_mfa import flash_attention
from mlx_mfa.attention import _fallback_sdpa


def benchmark_one(B, H, N, D, causal, dtype, n_warmup=5, n_iter=20):
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)

    results = {}
    for name, fn in [
        ("mlx_sdpa", lambda: _fallback_sdpa(q, k, v, scale, causal)),
        ("mlx_mfa", lambda: flash_attention(q, k, v, scale=scale, causal=causal)),
    ]:
        for _ in range(n_warmup):
            out = fn()
            mx.eval(out)

        times = []
        for _ in range(n_iter):
            mx.synchronize()
            t0 = time.perf_counter()
            out = fn()
            mx.eval(out)
            mx.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        median_ms = sorted(times)[len(times) // 2] * 1000
        results[name] = median_ms

    return results


def main():
    parser = argparse.ArgumentParser(description="mlx-mfa benchmark")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--seq-len", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=["f16", "bf16", "f32"], default="f16")
    args = parser.parse_args()

    dtype_map = {"f16": mx.float16, "bf16": mx.bfloat16, "f32": mx.float32}
    dtype = dtype_map[args.dtype]

    print(f"B={args.batch} H={args.heads} causal={args.causal} dtype={args.dtype}")
    print(f"{'D':>5} {'N':>6} {'SDPA ms':>9} {'MFA ms':>9} {'Speedup':>8}")
    print("-" * 45)

    for D in args.head_dim:
        for N in args.seq_len:
            r = benchmark_one(args.batch, args.heads, N, D,
                              args.causal, dtype)
            sdpa = r["mlx_sdpa"]
            mfa = r["mlx_mfa"]
            speedup = sdpa / mfa if mfa > 0 else float("inf")
            print(f"{D:>5} {N:>6} {sdpa:>9.2f} {mfa:>9.2f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
