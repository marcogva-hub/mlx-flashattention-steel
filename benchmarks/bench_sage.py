"""Benchmark: SageAttention (int8 Q/K) vs fp16 flash_attention.

Usage:
    .venv/bin/python benchmarks/bench_sage.py
    .venv/bin/python benchmarks/bench_sage.py --D 256 --N 4096
"""

import argparse
import math
import time

import mlx.core as mx

from mlx_mfa import flash_attention, sage_attention
from mlx_mfa.attention import _ext_available


def benchmark_one(label, fn, warmup, n_iter):
    # Warmup
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    # Timed
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn()
        mx.eval(out)
    t1 = time.perf_counter()
    ms = (t1 - t0) / n_iter * 1000
    print(f"  {label:<30s}  {ms:7.3f} ms")
    return ms


def run(B=1, H=8, N=2048, D=128, causal=True, dtype=mx.float16,
        n_warmup=5, n_iter=20):
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)

    dtype_str = "f16" if dtype == mx.float16 else "bf16"
    print(f"\nB={B} H={H} N={N} D={D} causal={causal} {dtype_str}")

    ms_fa = benchmark_one(
        "flash_attention (fp16)",
        lambda: flash_attention(q, k, v, scale=scale, causal=causal),
        n_warmup, n_iter,
    )

    if _ext_available():
        ms_sage = benchmark_one(
            "sage_attention (int8 Q/K)",
            lambda: sage_attention(q, k, v, scale=scale, causal=causal),
            n_warmup, n_iter,
        )
        ms_sage_ns = benchmark_one(
            "sage_attention (no smooth)",
            lambda: sage_attention(q, k, v, scale=scale, causal=causal,
                                   apply_smooth_k=False),
            n_warmup, n_iter,
        )
        print(f"  => Sage/FA speedup: {ms_fa/ms_sage:.2f}x  "
              f"(no-smooth: {ms_fa/ms_sage_ns:.2f}x)")
    else:
        print("  [MFA extension not available — sage benchmark skipped]")


def main():
    parser = argparse.ArgumentParser(description="bench_sage")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--N", type=int, default=None,
                        help="sequence length (default: sweep 512..4096)")
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    warmup = 2 if args.no_warmup else 5

    if args.N is not None:
        run(B=args.B, H=args.H, N=args.N, D=args.D, causal=args.causal,
            n_warmup=warmup)
    else:
        for N in [512, 1024, 2048, 4096]:
            run(B=args.B, H=args.H, N=N, D=args.D, causal=args.causal,
                n_warmup=warmup)


if __name__ == "__main__":
    main()
