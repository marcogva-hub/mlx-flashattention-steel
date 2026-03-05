#!/usr/bin/env python3
"""bench_segment.py — Segment mask benchmarks.

Compares segment-masked sparse attention vs running each segment independently
vs dense SDPA.

Usage:
    python benchmarks/bench_segment.py
"""
from __future__ import annotations
import math, os, sys, time
import numpy as np
sys.path.insert(0, ".")
import mlx.core as mx
from mlx_mfa import flash_attention, flash_attention_sparse, make_segment_mask, is_mfa_available

WARMUP, TIMED = 5, 20

def timed_run(fn, warmup=WARMUP, n=TIMED):
    for _ in range(warmup):
        fn(); mx.eval()
    times = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); mx.eval()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))

def sparsity_pct(mask):
    m = np.array(mask)
    return 100.0 * m.sum() / m.size

SCENARIOS = [
    # (label, segment_lengths, H_heads, D)
    ("2_segs",    [2048, 2048],                     4, 128),
    ("4_segs",    [1024, 1024, 1024, 1024],         4, 128),
    ("8_segs",    [512] * 8,                        4, 128),
    ("mixed",     [256, 1024, 512, 2048],           4, 128),
    ("16_segs",   [256] * 16,                       4, 128),
]

def main():
    if not is_mfa_available():
        print("[INFO] MFA extension not available. Using MLX fallback.")
    print("mlx-mfa segment mask benchmarks")
    print(f"{'Scenario':<15} {'N':>6} {'Spar%':>6} {'Mask(ms)':>10}"
          f" {'Sparse(ms)':>12} {'PerSeg(ms)':>12} {'Dense(ms)':>11}")
    print("-" * 80)
    all_rows = []
    for label, seg_lens, H, D in SCENARIOS:
        N = sum(seg_lens)
        B = 1; scale = 1.0 / math.sqrt(D); dtype = mx.float16
        q = mx.random.normal((B, H, N, D)).astype(dtype)
        k = mx.random.normal((B, H, N, D)).astype(dtype)
        v = mx.random.normal((B, H, N, D)).astype(dtype)

        t0 = time.perf_counter()
        mask = make_segment_mask(seg_lens, head_dim=D)
        mx.eval(mask)
        mask_ms = (time.perf_counter() - t0) * 1000.0
        sp = sparsity_pct(mask)

        def sparse_fn():
            return flash_attention_sparse(q, k, v, mask, scale=scale, causal=False)
        sparse_ms = timed_run(sparse_fn)

        offs = [0] + [int(x) for x in np.cumsum(seg_lens)]
        seg_qs = [q[:, :, offs[i]:offs[i+1], :] for i in range(len(seg_lens))]
        seg_ks = [k[:, :, offs[i]:offs[i+1], :] for i in range(len(seg_lens))]
        seg_vs = [v[:, :, offs[i]:offs[i+1], :] for i in range(len(seg_lens))]
        def perseg_fn():
            outs = [flash_attention(seg_qs[i], seg_ks[i], seg_vs[i], scale=scale)
                    for i in range(len(seg_lens))]
            return mx.concatenate(outs, axis=2)
        perseg_ms = timed_run(perseg_fn)

        def dense_fn():
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        dense_ms = timed_run(dense_fn)

        print(f"{label:<15} {N:>6,} {sp:>5.1f}% {mask_ms:>9.2f}ms"
              f" {sparse_ms:>11.2f}ms {perseg_ms:>11.2f}ms {dense_ms:>10.2f}ms")
        all_rows.append({"label": label, "N": N, "sparsity": sp,
                          "mask_ms": mask_ms, "sparse_ms": sparse_ms,
                          "perseg_ms": perseg_ms, "dense_ms": dense_ms})

    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RESULTS.md")
    section = "\n\n## v0.7.0 — Segment Mask Benchmarks\n\n"
    section += "| Scenario | N | Sparsity | Mask (ms) | Sparse (ms) | Per-segment (ms) | Dense (ms) |\n"
    section += "|----------|---|----------|-----------|-------------|------------------|------------|\n"
    for r in all_rows:
        section += (f"| {r['label']} | {r['N']:,} | {r['sparsity']:.1f}% |"
                    f" {r['mask_ms']:.2f} | {r['sparse_ms']:.2f} | {r['perseg_ms']:.2f} | {r['dense_ms']:.2f} |\n")
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("# mlx-mfa Benchmark Results\n")
    with open(results_path, "a") as f:
        f.write(section)
    print(f"\nResults appended to {results_path}")

if __name__ == "__main__":
    main()
