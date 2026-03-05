#!/usr/bin/env python3
"""bench_rope_3d.py — 3D RoPE table construction + flash_attention_rope benchmarks.

Compares:
  A) make_rope_3d_tables + flash_attention_rope (tables once, reused)
  B) Recompute tables each call (worst-case)
  C) flash_attention with pre-applied RoPE in Python (separate pass)

Usage:
    python benchmarks/bench_rope_3d.py
"""
from __future__ import annotations
import math, os, sys, time
import numpy as np
sys.path.insert(0, ".")
import mlx.core as mx
from mlx_mfa import flash_attention, flash_attention_rope, make_rope_3d_tables, is_mfa_available

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
    # (label, grid_h, grid_w, T, H_heads, D)
    ("dit_tiny",    8,  8,  4,  8, 128),
    ("dit_small",  16, 16,  8,  8, 128),
    ("dit_medium", 32, 32, 16,  8, 128),
    ("dit_large",  64, 64,  8,  8, 128),
]

def apply_rope_python(x, cos, sin):
    """Apply RoPE from [N, D/2] tables in Python."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    rot0 = x0 * cos - x1 * sin
    rot1 = x0 * sin + x1 * cos
    B, H, N, D = x.shape
    out = mx.concatenate([rot0, rot1], axis=-1)
    out = out.reshape(B, H, N, 2, D // 2)
    out = mx.transpose(out, (0, 1, 2, 4, 3)).reshape(B, H, N, D)
    return out

def main():
    if not is_mfa_available():
        print("[INFO] MFA extension not available. Using MLX fallback.")
    print("mlx-mfa 3D RoPE benchmarks")
    print(f"{'Scenario':<18} {'N':>8} {'TableBuild(ms)':>15} {'RopeFwd(ms)':>13}"
          f" {'PlainFwd(ms)':>13} {'PythonRope(ms)':>15}")
    print("-" * 90)
    all_rows = []
    for label, gh, gw, T, H, D in SCENARIOS:
        N = gh * gw * T
        dtype = mx.float16
        B = 1
        scale = 1.0 / math.sqrt(D)
        q = mx.random.normal((B, H, N, D)).astype(dtype)
        k = mx.random.normal((B, H, N, D)).astype(dtype)
        v = mx.random.normal((B, H, N, D)).astype(dtype)

        # Table build time
        t0 = time.perf_counter()
        cos, sin = make_rope_3d_tables(gh, gw, T, head_dim=D)
        mx.eval(cos, sin)
        table_ms = (time.perf_counter() - t0) * 1000.0

        # A: rope fwd (tables pre-built, reused)
        def rope_fwd():
            return flash_attention_rope(q, k, v, cos, sin, scale=scale)
        rope_ms = timed_run(rope_fwd)

        # B: plain flash_attention (no RoPE — baseline)
        def plain_fwd():
            return flash_attention(q, k, v, scale=scale)
        plain_ms = timed_run(plain_fwd)

        # C: Python RoPE + flash_attention
        def python_rope_fwd():
            q_r = apply_rope_python(q, cos, sin)
            k_r = apply_rope_python(k, cos, sin)
            return flash_attention(q_r, k_r, v, scale=scale)
        pyrop_ms = timed_run(python_rope_fwd)

        print(f"{label:<18} {N:>8,} {table_ms:>14.2f}ms {rope_ms:>12.2f}ms"
              f" {plain_ms:>12.2f}ms {pyrop_ms:>14.2f}ms")
        all_rows.append({"label": label, "N": N, "table_ms": table_ms,
                          "rope_ms": rope_ms, "plain_ms": plain_ms, "pyrop_ms": pyrop_ms})

    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RESULTS.md")
    section = "\n\n## v0.7.0 — 3D RoPE Benchmarks\n\n"
    section += "| Scenario | N | Table build (ms) | RopeFwd (ms) | PlainFwd (ms) | PyRope+Fwd (ms) |\n"
    section += "|----------|---|-----------------|--------------|---------------|------------------|\n"
    for r in all_rows:
        section += (f"| {r['label']} | {r['N']:,} | {r['table_ms']:.2f} |"
                    f" {r['rope_ms']:.2f} | {r['plain_ms']:.2f} | {r['pyrop_ms']:.2f} |\n")
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("# mlx-mfa Benchmark Results\n")
    with open(results_path, "a") as f:
        f.write(section)
    print(f"\nResults appended to {results_path}")

if __name__ == "__main__":
    main()
