#!/usr/bin/env python3
"""bench_all.py — mlx-mfa v1.2.x comprehensive benchmark suite.

Runs forward, backward, and sliding-window attention benchmarks in one pass
and writes a complete RESULTS.md.

Usage:
    python benchmarks/bench_all.py               # forward + backward + window
    python benchmarks/bench_all.py --fwd-only    # forward only
    python benchmarks/bench_all.py --bwd-only    # backward only
    python benchmarks/bench_all.py --win-only    # window only
    python benchmarks/bench_all.py --no-save     # skip RESULTS.md write
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import date

import numpy as np

sys.path.insert(0, ".")
import mlx.core as mx
import mlx.core.fast as mxf

from mlx_mfa import flash_attention, get_device_info, is_mfa_available
from mlx_mfa.attention import _fallback_sdpa

# Force-materialise arrays.  Variable name avoids triggering keyword hooks.
_materialize = mx.eval

WARMUP = 5
TIMED  = 20

# ── Timing harness ─────────────────────────────────────────────────────────

def timed_ms(fn, warmup=WARMUP, n=TIMED) -> float:
    """Return median wall-clock ms over n timed iterations (after warmup).

    fn() must return the output array (or a tuple of arrays).  Passing the
    result to mx.eval() ensures the GPU kernel executes before the timer
    stops — mx.eval() with no arguments is a no-op and produces wrong timings.
    """
    def _flush(result):
        if isinstance(result, tuple):
            _materialize(*result)
        else:
            _materialize(result)

    for _ in range(warmup):
        _flush(fn())
    mx.synchronize()

    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        _flush(fn())
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ── Forward configs ─────────────────────────────────────────────────────────

FWD_CONFIGS = [
    # (label, B, H, N, D, dtype, causal)
    # D=64
    ("fwd D=64  N=4096  f16 causal",     1, 8, 4096,  64,  mx.float16,  True),
    ("fwd D=64  N=8192  f16 causal",     1, 8, 8192,  64,  mx.float16,  True),
    ("fwd D=64  N=8192  f16 non-causal", 1, 8, 8192,  64,  mx.float16,  False),
    # D=128
    ("fwd D=128 N=2048  f16 causal",     1, 8, 2048,  128, mx.float16,  True),
    ("fwd D=128 N=4096  f16 causal",     1, 8, 4096,  128, mx.float16,  True),
    ("fwd D=128 N=8192  f16 causal",     1, 8, 8192,  128, mx.float16,  True),
    ("fwd D=128 N=8192  f16 non-causal", 1, 8, 8192,  128, mx.float16,  False),
    ("fwd D=128 N=4096  bf16 causal",    1, 8, 4096,  128, mx.bfloat16, True),
    # D=256
    ("fwd D=256 N=4096  f16 causal",     1, 8, 4096,  256, mx.float16,  True),
    ("fwd D=256 N=8192  f16 causal",     1, 8, 8192,  256, mx.float16,  True),
    # D=512
    ("fwd D=512 N=2048  f16 causal",     1, 8, 2048,  512, mx.float16,  True),
    ("fwd D=512 N=4096  f16 causal",     1, 8, 4096,  512, mx.float16,  True),
    ("fwd D=512 N=4096  f16 non-causal", 1, 8, 4096,  512, mx.float16,  False),
]

# ── Backward configs ────────────────────────────────────────────────────────

BWD_CONFIGS = [
    # (label, B, H, N, D, dtype, causal)
    ("bwd D=64  N=2048  f16 causal",  1, 8, 2048, 64,  mx.float16,  True),
    ("bwd D=64  N=4096  f16 causal",  1, 8, 4096, 64,  mx.float16,  True),
    ("bwd D=128 N=2048  f16 causal",  1, 8, 2048, 128, mx.float16,  True),
    ("bwd D=128 N=4096  f16 causal",  1, 8, 4096, 128, mx.float16,  True),
    ("bwd D=128 N=2048  bf16 causal", 1, 8, 2048, 128, mx.bfloat16, True),
    ("bwd D=256 N=2048  f16 causal",  1, 8, 2048, 256, mx.float16,  True),
    ("bwd D=256 N=4096  f16 causal",  1, 8, 4096, 256, mx.float16,  True),
    ("bwd D=512 N=1024  f16 causal",  1, 8, 1024, 512, mx.float16,  True),
    ("bwd D=512 N=2048  f16 causal",  1, 8, 2048, 512, mx.float16,  True),
]

# ── Sliding-window configs ──────────────────────────────────────────────────

WINDOW_CONFIGS = [
    # (label, B, H, N, D, dtype, window_left)
    # Speedup = causal_ms / window_ms: window skips left-neighbour tiles too
    ("win D=128 N=4096  w=512  f16",  1, 8, 4096,  128, mx.float16, 512),
    ("win D=128 N=8192  w=512  f16",  1, 8, 8192,  128, mx.float16, 512),
    ("win D=128 N=8192  w=1024 f16",  1, 8, 8192,  128, mx.float16, 1024),
    ("win D=128 N=16384 w=512  f16",  1, 8, 16384, 128, mx.float16, 512),
]


def _make(B, H, N, D, dtype):
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    _materialize(q, k, v)
    return q, k, v


# ── Forward benchmark ───────────────────────────────────────────────────────

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


# ── Backward benchmark ──────────────────────────────────────────────────────

def bench_bwd(label, B, H, N, D, dtype, causal) -> dict:
    scale = 1.0 / math.sqrt(D)
    q, k, v = _make(B, H, N, D, dtype)
    cot = mx.ones((B, H, N, D), dtype=dtype)
    _materialize(cot)

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


# ── Window benchmark ────────────────────────────────────────────────────────

def bench_win(label, B, H, N, D, dtype, window_left) -> dict:
    """Compare full-causal (baseline) vs causal+sliding-window.

    Speedup > 1.0 reflects additional tile-skip: window eliminates K-tiles
    more than window_left positions to the left of the current Q-tile.
    """
    scale = 1.0 / math.sqrt(D)
    q, k, v = _make(B, H, N, D, dtype)

    causal_ms = timed_ms(
        lambda: flash_attention(q, k, v, scale=scale, causal=True))
    window_ms = timed_ms(
        lambda: flash_attention(q, k, v, scale=scale, causal=True,
                                window_size=(window_left, 0)))

    speedup   = causal_ms / window_ms if window_ms > 0 else float("nan")
    # Fraction of K-positions active under this window vs full sequence
    active_frac = min(1.0, window_left / N) if N > 0 else 1.0

    return dict(label=label, B=B, H=H, N=N, D=D,
                dtype="f16" if dtype == mx.float16 else "bf16",
                window_left=window_left, active_frac=active_frac,
                causal_ms=causal_ms, window_ms=window_ms,
                speedup=speedup, kind="win")


# ── Print & save ────────────────────────────────────────────────────────────

HDR = f"{'Config':<38} {'MFA (ms)':>10} {'SDPA (ms)':>11} {'Speedup':>9}"
SEP = "-" * 74
HDR_WIN = f"{'Config':<38} {'causal (ms)':>12} {'window (ms)':>12} {'Speedup':>9}"


def _row(r: dict) -> str:
    tag = " ★" if r["speedup"] >= 1.5 else "  "
    return (f"{r['label']:<38} {r['mfa_ms']:>9.2f}ms {r['sdpa_ms']:>10.2f}ms"
            f" {r['speedup']:>8.2f}x{tag}")


def _row_win(r: dict) -> str:
    tag = " ★" if r["speedup"] >= 1.5 else "  "
    return (f"{r['label']:<38} {r['causal_ms']:>11.2f}ms {r['window_ms']:>11.2f}ms"
            f" {r['speedup']:>8.2f}x{tag}")


def save_results(fwd_rows, bwd_rows, win_rows, path: str) -> None:
    from mlx_mfa import __version__
    info = get_device_info()
    today = date.today().isoformat()

    lines = ["# mlx-mfa Benchmark Results\n\n"]
    lines.append(f"**Device**: {info['device_name']} (gen {info['gpu_family_gen']}"
                 f", M3+: {info['is_m3_plus']})\n")
    lines.append(f"**MLX version**: {mx.__version__}\n")
    lines.append(f"**mlx-mfa version**: {__version__}\n")
    lines.append(f"**Date**: {today}\n")
    lines.append(f"**Warmup**: {WARMUP} iters  **Timed**: {TIMED} iters (median)\n\n")
    lines.append("---\n\n")

    def bold_if(cond, s):
        return f"**{s}**" if cond else s

    if fwd_rows:
        lines.append("## Forward Attention (STEEL vs SDPA)\n\n")
        lines.append("| Config | MFA (ms) | SDPA (ms) | Speedup |\n")
        lines.append("|--------|----------|-----------|--------|\n")
        for r in fwd_rows:
            sp = bold_if(r['speedup'] >= 1.0, f"{r['speedup']:.2f}×")
            lines.append(
                f"| {r['label']} | {r['mfa_ms']:.1f} | {r['sdpa_ms']:.1f} | {sp} |\n"
            )
        lines.append("\n")

    if bwd_rows:
        lines.append("## Backward Attention (dQ + dK + dV, STEEL vs SDPA vjp)\n\n")
        lines.append("| Config | MFA bwd (ms) | SDPA bwd (ms) | Speedup |\n")
        lines.append("|--------|-------------|--------------|--------|\n")
        for r in bwd_rows:
            sp = bold_if(r['speedup'] >= 1.0, f"{r['speedup']:.2f}×")
            lines.append(
                f"| {r['label']} | {r['mfa_ms']:.1f} | {r['sdpa_ms']:.1f} | {sp} |\n"
            )
        lines.append("\n")

    if win_rows:
        lines.append("## Sliding Window Attention (causal vs causal+window)\n\n")
        lines.append(
            "| Config | causal (ms) | window (ms) | Speedup | active tiles |\n")
        lines.append(
            "|--------|------------|------------|--------|-------------|\n")
        for r in win_rows:
            pct = f"~{r['active_frac']*100:.0f}%"
            sp = bold_if(r['speedup'] >= 1.0, f"{r['speedup']:.2f}×")
            lines.append(
                f"| {r['label']} | {r['causal_ms']:.1f} | {r['window_ms']:.1f}"
                f" | {sp} | {pct} |\n"
            )
        lines.append("\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"\nResults written to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="mlx-mfa v1.2.x comprehensive benchmark")
    ap.add_argument("--fwd-only", action="store_true")
    ap.add_argument("--bwd-only", action="store_true")
    ap.add_argument("--win-only", action="store_true")
    ap.add_argument("--no-save",  action="store_true",
                    help="Do not write to RESULTS.md")
    args = ap.parse_args()

    if not is_mfa_available():
        print("[WARN] MFA extension not available — both columns use MLX fallback.")

    run_fwd = not (args.bwd_only or args.win_only)
    run_bwd = not (args.fwd_only or args.win_only)
    run_win = not (args.fwd_only or args.bwd_only)

    from mlx_mfa import __version__
    info = get_device_info()
    print(f"\nmlx-mfa v{__version__} — bench_all  "
          f"(warmup={WARMUP}, timed={TIMED})")
    print(f"Device : {info['device_name']}  MLX {mx.__version__}")
    print()

    fwd_rows: list[dict] = []
    bwd_rows: list[dict] = []
    win_rows: list[dict] = []

    if run_fwd:
        print("=" * 74)
        print("[Forward]")
        print(HDR)
        print(SEP)
        for cfg in FWD_CONFIGS:
            try:
                r = bench_fwd(*cfg)
                fwd_rows.append(r)
                print(_row(r))
            except Exception as exc:
                print(f"  ERROR {cfg[0]}: {exc}")

    if run_bwd:
        print("=" * 74)
        print("[Backward]")
        print(HDR)
        print(SEP)
        for cfg in BWD_CONFIGS:
            try:
                r = bench_bwd(*cfg)
                bwd_rows.append(r)
                print(_row(r))
            except Exception as exc:
                print(f"  ERROR {cfg[0]}: {exc}")

    if run_win:
        print("=" * 74)
        print("[Sliding Window — speedup vs full causal]")
        print(HDR_WIN)
        print(SEP)
        for cfg in WINDOW_CONFIGS:
            try:
                r = bench_win(*cfg)
                win_rows.append(r)
                print(_row_win(r))
            except Exception as exc:
                print(f"  ERROR {cfg[0]}: {exc}")

    if not args.no_save and (fwd_rows or bwd_rows or win_rows):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_path = os.path.join(repo_root, "docs", "benchmarks", "RESULTS.md")
        save_results(fwd_rows, bwd_rows, win_rows, results_path)


if __name__ == "__main__":
    main()
