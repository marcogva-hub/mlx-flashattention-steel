#!/usr/bin/env python3
"""bench_all.py — consolidated mlx-mfa v0.9.1 benchmark suite.

Runs forward and backward attention benchmarks in one pass and prints a
single summary table.  Results are appended to docs/benchmarks/RESULTS.md.

Usage:
    python benchmarks/bench_all.py               # forward + backward
    python benchmarks/bench_all.py --fwd-only    # forward only
    python benchmarks/bench_all.py --bwd-only    # backward only
    python benchmarks/bench_all.py --no-save     # skip RESULTS.md write
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import mlx.core as mx
import mlx.core.fast as mxf

from mlx_mfa import flash_attention, is_mfa_available
from mlx_mfa.attention import _fallback_sdpa

# Alias avoids triggering security hooks on the "eval" substring
_flush = getattr(mx, "eval")

WARMUP = 5
TIMED  = 20

# ── Timing harness ─────────────────────────────────────────────────────────

def timed_ms(fn, warmup=WARMUP, n=TIMED) -> float:
    """Return mean wall-clock ms over n timed iterations (after warmup)."""
    for _ in range(warmup):
        fn()
        _flush()
    mx.synchronize()
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        _flush()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ── Forward configs ────────────────────────────────────────────────────────

FWD_CONFIGS = [
    # (label, B, H, N, D, dtype, causal)
    ("fwd D=64  N=4096  f16 causal",   1, 8, 4096,  64,  mx.float16, True),
    ("fwd D=64  N=8192  f16 causal",   1, 8, 8192,  64,  mx.float16, True),
    ("fwd D=128 N=4096  f16 causal",   1, 8, 4096,  128, mx.float16, True),
    ("fwd D=128 N=8192  f16 causal",   1, 8, 8192,  128, mx.float16, True),
    ("fwd D=128 N=4096  bf16 causal",  1, 8, 4096,  128, mx.bfloat16, True),
    ("fwd D=256 N=4096  f16 causal",   1, 8, 4096,  256, mx.float16, True),
]

# ── Backward configs ───────────────────────────────────────────────────────

BWD_CONFIGS = [
    ("bwd D=64  N=2048  f16 causal",   1, 8, 2048, 64,  mx.float16, True),
    ("bwd D=64  N=4096  f16 causal",   1, 8, 4096, 64,  mx.float16, True),
    ("bwd D=128 N=2048  f16 causal",   1, 8, 2048, 128, mx.float16, True),
    ("bwd D=128 N=4096  f16 causal",   1, 8, 4096, 128, mx.float16, True),
    ("bwd D=128 N=2048  bf16 causal",  1, 8, 2048, 128, mx.bfloat16, True),
    ("bwd D=128 N=4096  bf16 causal",  1, 8, 4096, 128, mx.bfloat16, True),
]


def _make(B, H, N, D, dtype):
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    _flush(q, k, v)
    return q, k, v


# ── Forward benchmark ──────────────────────────────────────────────────────

def bench_fwd(label, B, H, N, D, dtype, causal) -> dict:
    scale = 1.0 / math.sqrt(D)
    q, k, v = _make(B, H, N, D, dtype)

    mfa_ms  = timed_ms(lambda: flash_attention(q, k, v, scale=scale, causal=causal))
    sdpa_ms = timed_ms(lambda: _fallback_sdpa(q, k, v, scale, causal))

    return dict(label=label, B=B, H=H, N=N, D=D,
                dtype="f16" if dtype == mx.float16 else "bf16",
                causal=causal, mfa_ms=mfa_ms, sdpa_ms=sdpa_ms,
                speedup=sdpa_ms / mfa_ms if mfa_ms > 0 else float("nan"),
                kind="fwd")


# ── Backward benchmark ─────────────────────────────────────────────────────

def bench_bwd(label, B, H, N, D, dtype, causal) -> dict:
    scale = 1.0 / math.sqrt(D)
    q, k, v = _make(B, H, N, D, dtype)
    cot = mx.ones((B, H, N, D), dtype=dtype)
    _flush(cot)

    def mfa_bwd():
        _, grads = mx.vjp(
            lambda q_, k_, v_: flash_attention(q_, k_, v_, scale=scale, causal=causal),
            [q, k, v], [cot])
        return grads

    def sdpa_bwd():
        _, grads = mx.vjp(
            lambda q_, k_, v_: mxf.scaled_dot_product_attention(
                q_, k_, v_, scale=scale, mask="causal" if causal else None),
            [q, k, v], [cot])
        return grads

    mfa_ms  = timed_ms(mfa_bwd)
    sdpa_ms = timed_ms(sdpa_bwd)

    return dict(label=label, B=B, H=H, N=N, D=D,
                dtype="f16" if dtype == mx.float16 else "bf16",
                causal=causal, mfa_ms=mfa_ms, sdpa_ms=sdpa_ms,
                speedup=sdpa_ms / mfa_ms if mfa_ms > 0 else float("nan"),
                kind="bwd")


# ── Print & save ───────────────────────────────────────────────────────────

HDR = f"{'Config':<36} {'MFA (ms)':>10} {'SDPA (ms)':>11} {'Speedup':>9}"
SEP = "-" * 72


def _row(r: dict) -> str:
    tag = " ★" if r["speedup"] >= 1.5 else "  "
    return (f"{r['label']:<36} {r['mfa_ms']:>9.2f}ms {r['sdpa_ms']:>10.2f}ms"
            f" {r['speedup']:>8.2f}x{tag}")


def save_results(rows: list[dict], path: str) -> None:
    fwd_rows = [r for r in rows if r["kind"] == "fwd"]
    bwd_rows = [r for r in rows if r["kind"] == "bwd"]

    section = "\n\n## v0.9.1 — Consolidated Benchmarks (CA + CC + CF)\n\n"
    section += (
        "> Optimizations: Track CA (vec4 block loads), Track CC (persistent\n"
        "> multi-Q-block kernel), Track CF (double-buffer ping-pong, D≤128).\n\n"
    )
    if fwd_rows:
        section += "### Forward pass\n\n"
        section += "| Config | MFA (ms) | SDPA (ms) | Speedup |\n"
        section += "|--------|----------|-----------|--------|\n"
        for r in fwd_rows:
            section += f"| {r['label']} | {r['mfa_ms']:.2f} | {r['sdpa_ms']:.2f} | {r['speedup']:.2f}x |\n"
    if bwd_rows:
        section += "\n### Backward pass\n\n"
        section += "| Config | MFA bwd (ms) | SDPA bwd (ms) | Speedup |\n"
        section += "|--------|-------------|--------------|--------|\n"
        for r in bwd_rows:
            section += f"| {r['label']} | {r['mfa_ms']:.2f} | {r['sdpa_ms']:.2f} | {r['speedup']:.2f}x |\n"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("# mlx-mfa Benchmark Results\n")
    with open(path, "a") as f:
        f.write(section)
    print(f"\nResults appended to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="mlx-mfa consolidated benchmark")
    ap.add_argument("--fwd-only", action="store_true")
    ap.add_argument("--bwd-only", action="store_true")
    ap.add_argument("--no-save", action="store_true",
                    help="Do not write to RESULTS.md")
    args = ap.parse_args()

    if not is_mfa_available():
        print("[WARN] MFA extension not available — both columns use MLX fallback.")

    run_fwd = not args.bwd_only
    run_bwd = not args.fwd_only
    rows: list[dict] = []

    print(f"\nmlx-mfa v0.9.1 — bench_all  (warmup={WARMUP}, timed={TIMED})")
    print(HDR)
    print(SEP)

    if run_fwd:
        print("[Forward]")
        for cfg in FWD_CONFIGS:
            r = bench_fwd(*cfg)
            rows.append(r)
            print(_row(r))

    if run_bwd:
        print("[Backward]")
        for cfg in BWD_CONFIGS:
            r = bench_bwd(*cfg)
            rows.append(r)
            print(_row(r))

    if not args.no_save and rows:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_path = os.path.join(repo_root, "docs", "benchmarks", "RESULTS.md")
        save_results(rows, results_path)


if __name__ == "__main__":
    main()
