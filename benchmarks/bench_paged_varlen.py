#!/usr/bin/env python3
"""bench_paged_varlen.py — Benchmark fused PagedVarlenForward vs bridge.

Measures the performance gain of the fused Metal kernel (single dispatch)
vs the per-sequence Python bridge (N dispatches) for packed varlen queries
over paged KV.

Usage:
    python benchmarks/bench_paged_varlen.py
"""

import json
import math
import sys
import time

sys.path.insert(0, ".")

import mlx.core as mx

from mlx_mfa import flash_attention_paged, flash_attention_paged_varlen, is_mfa_available

WARMUP = 5
TIMED = 20


def make_paged_kv(kv_lens, H_kv, D, block_size, dtype):
    """Build paged KV pool, block table, seq_lens from per-seq kv lengths."""
    B = len(kv_lens)
    max_blocks_per_seq = max((kl + block_size - 1) // block_size for kl in kv_lens)
    total_pages = sum((kl + block_size - 1) // block_size for kl in kv_lens)

    k_pool = mx.random.normal((total_pages, block_size, H_kv, D)).astype(dtype)
    v_pool = mx.random.normal((total_pages, block_size, H_kv, D)).astype(dtype)

    bt_np = [[-1] * max_blocks_per_seq for _ in range(B)]
    page = 0
    for i, kl in enumerate(kv_lens):
        n_blks = (kl + block_size - 1) // block_size
        for j in range(n_blks):
            bt_np[i][j] = page
            page += 1
    bt = mx.array(bt_np, dtype=mx.int32)
    seq_lens = mx.array(kv_lens, dtype=mx.int32)
    return k_pool, v_pool, bt, seq_lens


def bench_fused(q, k_pool, v_pool, bt, seq_lens, cu_seqlens_q, scale, block_size):
    """Benchmark the fused kernel (production default)."""
    for _ in range(WARMUP):
        out = flash_attention_paged_varlen(
            q, k_pool, v_pool, bt, seq_lens, cu_seqlens_q,
            scale=scale, causal=True, block_size=block_size,
        )
        mx.synchronize()

    times = []
    for _ in range(TIMED):
        mx.synchronize()
        t0 = time.perf_counter()
        out = flash_attention_paged_varlen(
            q, k_pool, v_pool, bt, seq_lens, cu_seqlens_q,
            scale=scale, causal=True, block_size=block_size,
        )
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000  # median ms


def bench_bridge(q, k_pool, v_pool, bt, seq_lens, q_lens, cu_q, scale, block_size):
    """Benchmark the per-sequence bridge (old path)."""
    B = len(q_lens)

    def run_bridge():
        parts = []
        for i in range(B):
            qs, qe = cu_q[i], cu_q[i + 1]
            if qe == qs:
                continue
            out_i = flash_attention_paged(
                q[:, :, qs:qe, :],
                k_pool, v_pool,
                bt[i : i + 1],
                seq_lens[i : i + 1],
                scale=scale,
                causal=True,
                block_size=block_size,
            )
            parts.append(out_i)
        if parts:
            return mx.concatenate(parts, axis=2)
        return mx.zeros_like(q[:, :, :0, :])

    for _ in range(WARMUP):
        run_bridge()
        mx.synchronize()

    times = []
    for _ in range(TIMED):
        mx.synchronize()
        t0 = time.perf_counter()
        run_bridge()
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000  # median ms


CONFIGS = [
    # (B, q_lens, kv_lens, label)
    # Decode (N_q=1 per seq)
    (1, [1], [256], "B=1  decode  kv=256"),
    (1, [1], [1024], "B=1  decode  kv=1024"),
    (1, [1], [4096], "B=1  decode  kv=4096"),
    (4, [1] * 4, [256] * 4, "B=4  decode  kv=256 uniform"),
    (4, [1] * 4, [128, 256, 512, 1024], "B=4  decode  kv hetero"),
    (8, [1] * 8, [256] * 8, "B=8  decode  kv=256 uniform"),
    (8, [1] * 8, [128, 256, 384, 512, 640, 768, 896, 1024], "B=8  decode  kv hetero"),
    (16, [1] * 16, [256] * 16, "B=16 decode  kv=256 uniform"),
    (16, [1] * 16, [128 + 64 * i for i in range(16)], "B=16 decode  kv hetero"),
    # Prefill (N_q > 1)
    (4, [32] * 4, [256] * 4, "B=4  prefill q=32 kv=256"),
    (4, [8, 16, 32, 64], [128, 256, 512, 1024], "B=4  prefill q+kv hetero"),
    (8, [1, 4, 1, 8, 1, 2, 1, 16], [256] * 8, "B=8  mixed q hetero"),
]


def main():
    if not is_mfa_available():
        print("MFA not available — cannot benchmark fused kernel")
        return

    H_q, H_kv, D = 32, 8, 128
    block_size = 16
    dtype = mx.float16
    scale = 1.0 / math.sqrt(D)

    print(f"PagedVarlenForward Benchmark — H_q={H_q} H_kv={H_kv} D={D} f16")
    print(f"{'Config':<45} | {'Fused':>8} | {'Bridge':>8} | {'Speedup':>8}")
    print("-" * 80)

    results = []
    for B, q_lens, kv_lens, label in CONFIGS:
        mx.random.seed(42)
        total_q = sum(q_lens)
        q = mx.random.normal((1, H_q, total_q, D)).astype(dtype)
        k_pool, v_pool, bt, seq_lens = make_paged_kv(kv_lens, H_kv, D, block_size, dtype)

        cu_q = [0]
        for ql in q_lens:
            cu_q.append(cu_q[-1] + ql)
        cu_seqlens_q = mx.array(cu_q, dtype=mx.int32)

        t_fused = bench_fused(q, k_pool, v_pool, bt, seq_lens, cu_seqlens_q, scale, block_size)
        t_bridge = bench_bridge(q, k_pool, v_pool, bt, seq_lens, q_lens, cu_q, scale, block_size)

        speedup = t_bridge / t_fused if t_fused > 0 else float("inf")
        print(f"{label:<45} | {t_fused:7.2f}ms | {t_bridge:7.2f}ms | {speedup:7.2f}x")

        results.append(
            {
                "label": label,
                "B": B,
                "q_lens": q_lens,
                "kv_lens": kv_lens,
                "fused_ms": round(t_fused, 3),
                "bridge_ms": round(t_bridge, 3),
                "speedup": round(speedup, 2),
            }
        )

    out_path = "devnotes/paged_varlen_fused_bench.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
