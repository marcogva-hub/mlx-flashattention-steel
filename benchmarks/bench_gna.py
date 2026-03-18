"""Benchmark: GNA native kernel vs sparse+mask vs dense attention.

Usage:
    .venv/bin/python benchmarks/bench_gna.py
    .venv/bin/python benchmarks/bench_gna.py --save  # save JSON results
"""

import argparse
import json
import math
import os
import time

import mlx.core as mx
import numpy as np


def _bench(fn, n_warmup: int = 5, n_iter: int = 20) -> float:
    """Return median time in ms."""
    for _ in range(n_warmup):
        mx.eval(fn())
    mx.synchronize()

    times = []
    for _ in range(n_iter):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)[len(times) // 2] * 1000  # median ms


def bench_gna():
    from mlx_mfa import flash_attention, flash_attention_gna, flash_attention_sparse
    from mlx_mfa.masks import make_gna_mask

    B, H, D = 1, 8, 128
    dtype = mx.float16
    scale = 1.0 / math.sqrt(D)

    configs = [
        # (T, pH, pW, window, stride, label)
        (4, 8, 8, (3, 5, 5), (1, 1, 1), "small-sliding"),
        (4, 16, 16, (3, 5, 5), (1, 1, 1), "medium-sliding"),
        (4, 16, 16, (2, 8, 8), (2, 8, 8), "medium-blocked"),
        (4, 16, 16, (2, 8, 8), (1, 4, 4), "medium-strided"),
        (8, 32, 32, (4, 8, 8), (1, 1, 1), "large-sliding"),
        (8, 32, 32, (4, 8, 8), (2, 4, 4), "large-strided"),
        (8, 32, 32, (4, 8, 8), (4, 8, 8), "large-blocked"),
    ]

    results = []

    header = (
        f"{'Config':<22} | {'N':>6} | {'Sparsity':>8} | "
        f"{'GNA native':>10} | {'Sparse+mask':>11} | {'Dense':>8} | "
        f"{'GNA/Sparse':>10} | {'GNA/Dense':>9}"
    )
    print(header)
    print("-" * len(header))

    for T, pH, pW, window, stride, label in configs:
        N = T * pH * pW
        seq_shape = (T, pH, pW)

        mx.random.seed(42)
        q = mx.random.normal((B, H, N, D)).astype(dtype)
        k = mx.random.normal((B, H, N, D)).astype(dtype)
        v = mx.random.normal((B, H, N, D)).astype(dtype)
        mx.eval(q, k, v)

        # Compute mask sparsity
        mask = make_gna_mask(seq_shape, window, stride, head_dim=D)
        mx.eval(mask)
        mask_np = np.array(mask)
        sparsity = 1.0 - mask_np.mean()

        # GNA native kernel
        t_gna = _bench(lambda: flash_attention_gna(
            q, k, v, seq_shape, window, stride, scale=scale))

        # Sparse + mask baseline
        t_sparse = _bench(lambda: flash_attention_sparse(
            q, k, v, mask, scale=scale))

        # Dense baseline (skip for very large N)
        if N <= 16384:
            t_dense = _bench(lambda: flash_attention(
                q, k, v, scale=scale, backend="sdpa"))
        else:
            t_dense = float("nan")

        gna_vs_sparse = t_sparse / t_gna if t_gna > 0 else float("nan")
        gna_vs_dense = t_dense / t_gna if t_gna > 0 and not math.isnan(t_dense) else float("nan")

        row = {
            "label": label,
            "N": N,
            "seq_shape": list(seq_shape),
            "window": list(window),
            "stride": list(stride),
            "sparsity": round(sparsity, 4),
            "gna_ms": round(t_gna, 3),
            "sparse_ms": round(t_sparse, 3),
            "dense_ms": round(t_dense, 3) if not math.isnan(t_dense) else None,
            "gna_vs_sparse": round(gna_vs_sparse, 2),
            "gna_vs_dense": round(gna_vs_dense, 2) if not math.isnan(gna_vs_dense) else None,
        }
        results.append(row)

        dense_str = f"{t_dense:>7.2f}ms" if not math.isnan(t_dense) else "    skip"
        gna_dense_str = f"{gna_vs_dense:>8.2f}x" if not math.isnan(gna_vs_dense) else "     N/A"
        print(
            f"{label:<22} | {N:>6} | {sparsity:>7.1%} | "
            f"{t_gna:>9.2f}ms | {t_sparse:>10.2f}ms | {dense_str} | "
            f"{gna_vs_sparse:>9.2f}x | {gna_dense_str}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark GNA attention")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    print(f"\nGNA Forward Benchmark -- B=1 H=8 D=128 f16")
    print(f"{'='*100}\n")

    results = bench_gna()

    if args.save:
        out_dir = "devnotes"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "gna_forward_bench_initial.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
