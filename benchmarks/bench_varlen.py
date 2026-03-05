#!/usr/bin/env python3
"""bench_varlen.py — Variable-length batching benchmarks.

Compares:
  A) flash_attention_varlen (packed sequences)
  B) Padded: max-length tensor (simulates naive batching)
  C) Sequential: one sequence at a time then concatenate

Usage:
    python benchmarks/bench_varlen.py
"""
from __future__ import annotations
import math, os, sys, time
import numpy as np
sys.path.insert(0, ".")
import mlx.core as mx
from mlx_mfa import flash_attention, flash_attention_varlen, is_mfa_available

WARMUP, TIMED = 5, 20

def timed_run(fn, warmup=WARMUP, n=TIMED):
    for _ in range(warmup):
        fn(); mx.eval()
    times = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); mx.eval()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))

SCENARIOS = [
    ("uniform_10x1024",  [1024]*10,           8, 128),
    ("varied",           [256, 512, 1024, 2048, 512], 8, 128),
    ("extreme",          [64, 64, 64, 4096, 64],      8, 128),
    ("short_many",       [64]*32,              8, 128),
    ("two_long",         [2048, 2048],         8, 128),
]

def run_scenario(label, lengths, H, D):
    N_total = sum(lengths)
    N_max   = max(lengths)
    n_seqs  = len(lengths)
    scale   = 1.0 / math.sqrt(D)
    dtype   = mx.float16

    q = mx.random.normal((1, H, N_total, D)).astype(dtype)
    k = mx.random.normal((1, H, N_total, D)).astype(dtype)
    v = mx.random.normal((1, H, N_total, D)).astype(dtype)

    cu = mx.array([0] + [int(x) for x in np.cumsum(lengths)])

    def varlen_fn():
        return flash_attention_varlen(q, k, v, cu, cu, N_max, N_max, scale=scale)
    varlen_ms = timed_run(varlen_fn)

    q_s = q[:, :, :N_max, :]; k_s = k[:, :, :N_max, :]; v_s = v[:, :, :N_max, :]
    def padded_fn():
        return flash_attention(q_s, k_s, v_s, scale=scale)
    padded_ms = timed_run(padded_fn)

    offs = [0] + [int(x) for x in np.cumsum(lengths)]
    qs_list = [q[:, :, offs[i]:offs[i+1], :] for i in range(n_seqs)]
    ks_list = [k[:, :, offs[i]:offs[i+1], :] for i in range(n_seqs)]
    vs_list = [v[:, :, offs[i]:offs[i+1], :] for i in range(n_seqs)]
    def sequential_fn():
        outs = [flash_attention(qs_list[i], ks_list[i], vs_list[i], scale=scale)
                for i in range(n_seqs)]
        return mx.concatenate(outs, axis=2)
    sequential_ms = timed_run(sequential_fn)

    return {"label": label, "num_seqs": n_seqs, "N_total": N_total, "N_max": N_max,
            "varlen_ms": varlen_ms, "padded_ms": padded_ms, "sequential_ms": sequential_ms}

def main():
    if not is_mfa_available():
        print("[INFO] MFA extension not available. Using MLX fallback.")
    print("mlx-mfa varlen benchmarks")
    print(f"{'Scenario':<22} {'Seqs':>5} {'TotalN':>8} {'Varlen(ms)':>12}"
          f" {'Padded(ms)':>12} {'Seq(ms)':>10} {'vsPadded':>10} {'vsSeq':>8}")
    print("-" * 95)
    all_rows = []
    for label, lengths, H, D in SCENARIOS:
        r = run_scenario(label, lengths, H, D)
        all_rows.append(r)
        rp = r["padded_ms"] / r["varlen_ms"] if r["varlen_ms"] > 0 else float("nan")
        rs = r["sequential_ms"] / r["varlen_ms"] if r["varlen_ms"] > 0 else float("nan")
        print(f"{label:<22} {r['num_seqs']:>5} {r['N_total']:>8,}"
              f" {r['varlen_ms']:>11.2f}ms {r['padded_ms']:>11.2f}ms {r['sequential_ms']:>9.2f}ms"
              f" {rp:>9.2f}x {rs:>7.2f}x")

    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RESULTS.md")
    section = "\n\n## v0.7.0 — Varlen Attention Benchmarks\n\n"
    section += "| Scenario | Seqs | Total N | Varlen (ms) | Padded (ms) | Sequential (ms) |\n"
    section += "|----------|------|---------|-------------|-------------|------------------|\n"
    for r in all_rows:
        section += (f"| {r['label']} | {r['num_seqs']} | {r['N_total']:,} |"
                    f" {r['varlen_ms']:.2f} | {r['padded_ms']:.2f} | {r['sequential_ms']:.2f} |\n")
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("# mlx-mfa Benchmark Results\n")
    with open(results_path, "a") as f:
        f.write(section)
    print(f"\nResults appended to {results_path}")

if __name__ == "__main__":
    main()
