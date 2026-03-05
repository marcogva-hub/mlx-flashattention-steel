#!/usr/bin/env python3
"""bench_spatial_masks.py — Spatial mask construction + end-to-end attention benchmarks.

Measures:
  1. Mask construction time (make_spatial_2d_mask, 3D, segment, adaptive)
  2. End-to-end: mask + flash_attention_sparse vs dense SDPA
  3. Sparsity % and speedup

Usage:
    python benchmarks/bench_spatial_masks.py [--skip-attention]

Memory safety: scenarios where N > 100K tokens are automatically SKIPPED for
the full-attention benchmark.  Mask construction timing is always reported.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

# Allow running from project root or benchmarks/
sys.path.insert(0, ".")

import mlx.core as mx
from mlx_mfa import (
    flash_attention_sparse,
    make_spatial_2d_mask,
    make_spatial_3d_mask,
    make_segment_mask,
    make_adaptive_window_mask,
    is_mfa_available,
)

WARMUP = 5
TIMED = 20
MAX_N_FOR_ATTN = 100_000  # skip full attention above this token count


def timed_run(fn, warmup: int = WARMUP, n: int = TIMED) -> float:
    """Return mean wall-clock ms for fn()."""
    for _ in range(warmup):
        fn()
        mx.eval()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        mx.eval()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def dense_sdpa(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def sparsity_pct(mask: mx.array) -> float:
    m = np.array(mask)
    return 100.0 * m.sum() / m.size


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS_2D = [
    # (label, pH, pW, radius, H_heads, D)
    ("flickr_r4",    16,  16,   4, 4, 128),
    ("image_r8",     32,  32,   8, 4, 128),
    ("image_r16",    32,  32,  16, 4, 128),
    # FlashVSR-like: 360x640 patches (720p / 2x2 patch)
    ("flashvsr_r8",  180, 320,  8, 8, 128),
    ("flashvsr_r16", 180, 320, 16, 8, 128),
    ("flashvsr_r32", 180, 320, 32, 8, 128),
]

SCENARIOS_3D = [
    # (label, pH, pW, T, sp_r, tm_r, H_heads, D)
    ("video_dit_small",  16, 16,  8, 4, 2,  8, 128),
    ("video_dit_medium", 32, 32,  8, 4, 2,  8, 128),
    ("seedvr2_512",      32, 32, 16, 4, 2, 16, 128),
    ("seedvr2_1024",     64, 64, 16, 4, 2, 16, 128),
    ("diffvsr_8frame",   64, 64,  8, 64, 0, 8, 128),
]

SCENARIOS_SEGMENT = [
    # (label, segment_lengths, H_heads, D)
    ("2_segs_2k",  [1024, 1024],        4, 128),
    ("4_segs_1k",  [512, 512, 512, 512], 4, 128),
    ("8_segs_512", [256] * 8,           4, 128),
    ("mixed",      [256, 1024, 512, 2048], 4, 128),
]

SCENARIOS_ADAPTIVE = [
    # (label, H, W, T, bw_h, bw_w, train_res, infer_res, H_heads, D)
    ("adaptive_1x",  32,  32, 4, 16, 16, (256, 256), (256, 256),   8, 128),
    ("adaptive_2x",  64,  64, 4, 16, 16, (256, 256), (512, 512),   8, 128),
    ("adaptive_4x", 128, 128, 4, 16, 16, (256, 256), (1024, 1024), 8, 128),
]


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _run_one(
    label: str,
    N: int,
    mask: mx.array,
    mask_ms: float,
    H_heads: int,
    D: int,
    skip_attention: bool,
    row_type: str,
) -> dict:
    sp = sparsity_pct(mask)
    NQ, NK = mask.shape
    sparse_ms = dense_ms = speedup = float("nan")

    if N > MAX_N_FOR_ATTN:
        skip_note = f"N={N:,}>limit"
    elif skip_attention or not is_mfa_available():
        skip_note = "attention skipped"
    else:
        skip_note = ""
        dtype = mx.float16
        q = mx.random.normal((1, H_heads, N, D)).astype(dtype)
        k = mx.random.normal((1, H_heads, N, D)).astype(dtype)
        v = mx.random.normal((1, H_heads, N, D)).astype(dtype)
        scale = 1.0 / math.sqrt(D)
        frozen_mask = mask  # capture for closures

        def sparse_fn():
            return flash_attention_sparse(q, k, v, frozen_mask,
                                          scale=scale, causal=False)

        def dense_fn():
            return dense_sdpa(q, k, v, scale)

        sparse_ms = timed_run(sparse_fn)
        dense_ms = timed_run(dense_fn)
        speedup = dense_ms / sparse_ms if sparse_ms > 0.0 else float("nan")

    return {
        "type": row_type,
        "label": label,
        "N": N,
        "NQ": NQ,
        "NK": NK,
        "sparsity": sp,
        "mask_ms": mask_ms,
        "sparse_ms": sparse_ms,
        "dense_ms": dense_ms,
        "speedup": speedup,
        "skip_note": skip_note,
    }


def _fmt(val: float, suffix: str = "") -> str:
    return f"{val:.2f}{suffix}" if not math.isnan(val) else "N/A"


def run_2d_benchmarks(skip_attention: bool) -> list[dict]:
    print("\n=== 2D Spatial Mask Benchmarks ===")
    print(f"{'Scenario':<20} {'N':>8} {'Spar%':>6} {'Mask(ms)':>10}"
          f" {'Sparse(ms)':>11} {'Dense(ms)':>11} {'Speedup':>8}")
    print("-" * 80)
    rows: list[dict] = []
    for label, pH, pW, radius, H_heads, D in SCENARIOS_2D:
        N = pH * pW
        t0 = time.perf_counter()
        mask = make_spatial_2d_mask(pH, pW, spatial_radius=radius, head_dim=D)
        mx.eval(mask)
        mask_ms = (time.perf_counter() - t0) * 1000.0
        r = _run_one(label, N, mask, mask_ms, H_heads, D, skip_attention, "2D")
        rows.append(r)
        note = f"  [{r['skip_note']}]" if r["skip_note"] else ""
        print(
            f"{label:<20} {N:>8,} {r['sparsity']:>5.1f}% {mask_ms:>9.2f}ms"
            f" {_fmt(r['sparse_ms'], 'ms'):>11} {_fmt(r['dense_ms'], 'ms'):>11}"
            f" {_fmt(r['speedup'], 'x'):>8}{note}"
        )
    return rows


def run_3d_benchmarks(skip_attention: bool) -> list[dict]:
    print("\n=== 3D Spatio-Temporal Mask Benchmarks ===")
    print(f"{'Scenario':<22} {'N':>8} {'Spar%':>6} {'Mask(ms)':>10}"
          f" {'Sparse(ms)':>11} {'Dense(ms)':>11} {'Speedup':>8}")
    print("-" * 83)
    rows: list[dict] = []
    for label, pH, pW, T, sp_r, tm_r, H_heads, D in SCENARIOS_3D:
        N = pH * pW * T
        t0 = time.perf_counter()
        mask = make_spatial_3d_mask(pH, pW, T,
                                    spatial_radius=sp_r, temporal_radius=tm_r,
                                    head_dim=D)
        mx.eval(mask)
        mask_ms = (time.perf_counter() - t0) * 1000.0
        r = _run_one(label, N, mask, mask_ms, H_heads, D, skip_attention, "3D")
        rows.append(r)
        note = f"  [{r['skip_note']}]" if r["skip_note"] else ""
        print(
            f"{label:<22} {N:>8,} {r['sparsity']:>5.1f}% {mask_ms:>9.2f}ms"
            f" {_fmt(r['sparse_ms'], 'ms'):>11} {_fmt(r['dense_ms'], 'ms'):>11}"
            f" {_fmt(r['speedup'], 'x'):>8}{note}"
        )
    return rows


def run_segment_benchmarks(skip_attention: bool) -> list[dict]:
    print("\n=== Segment Mask Benchmarks ===")
    rows: list[dict] = []
    for label, seg_lengths, H_heads, D in SCENARIOS_SEGMENT:
        N = sum(seg_lengths)
        t0 = time.perf_counter()
        mask = make_segment_mask(seg_lengths, head_dim=D)
        mx.eval(mask)
        mask_ms = (time.perf_counter() - t0) * 1000.0
        r = _run_one(label, N, mask, mask_ms, H_heads, D, skip_attention, "segment")
        rows.append(r)
        note = f"  [{r['skip_note']}]" if r["skip_note"] else ""
        print(
            f"  {label:<20} N={N:>6,}  spar={r['sparsity']:.1f}%"
            f"  mask={mask_ms:.2f}ms  sparse={_fmt(r['sparse_ms'], 'ms')}"
            f"  dense={_fmt(r['dense_ms'], 'ms')}  speedup={_fmt(r['speedup'], 'x')}{note}"
        )
    return rows


def run_adaptive_benchmarks(skip_attention: bool) -> list[dict]:
    print("\n=== Adaptive Window Mask Benchmarks ===")
    rows: list[dict] = []
    for label, H, W, T, bw_h, bw_w, train_res, infer_res, H_heads, D in SCENARIOS_ADAPTIVE:
        N = H * W * T
        t0 = time.perf_counter()
        mask = make_adaptive_window_mask(
            H, W, num_frames=T,
            base_window_h=bw_h, base_window_w=bw_w,
            train_resolution=train_res,
            inference_resolution=infer_res,
            head_dim=D,
        )
        mx.eval(mask)
        mask_ms = (time.perf_counter() - t0) * 1000.0
        r = _run_one(label, N, mask, mask_ms, H_heads, D, skip_attention, "adaptive")
        rows.append(r)
        note = f"  [{r['skip_note']}]" if r["skip_note"] else ""
        print(
            f"  {label:<18} N={N:>6,}  spar={r['sparsity']:.1f}%"
            f"  mask={mask_ms:.2f}ms  sparse={_fmt(r['sparse_ms'], 'ms')}"
            f"  dense={_fmt(r['dense_ms'], 'ms')}  speedup={_fmt(r['speedup'], 'x')}{note}"
        )
    return rows


# ---------------------------------------------------------------------------
# RESULTS.md update
# ---------------------------------------------------------------------------

def update_results_md(all_rows: list[dict]) -> None:
    """Append v0.7.0 spatial mask benchmark results to RESULTS.md."""
    import os
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RESULTS.md"
    )

    section = "\n\n## v0.7.0 — Spatial Mask Benchmarks\n\n"
    section += (
        "| Type | Scenario | N tokens | Sparsity | Mask build (ms) |"
        " Sparse (ms) | Dense SDPA (ms) | Speedup |\n"
        "|------|----------|----------|----------|-----------------|"
        "------------|-----------------|--------|\n"
    )
    for r in all_rows:
        sp = _fmt(r["sparse_ms"])
        dp = _fmt(r["dense_ms"])
        su = _fmt(r["speedup"], "x")
        section += (
            f"| {r['type']} | {r['label']} | {r['N']:,} |"
            f" {r['sparsity']:.1f}% | {r['mask_ms']:.2f} | {sp} | {dp} | {su} |\n"
        )

    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("# mlx-mfa Benchmark Results\n")
    with open(results_path, "a") as f:
        f.write(section)
    print(f"\nResults appended to {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial mask benchmarks")
    parser.add_argument(
        "--skip-attention", action="store_true",
        help="Only benchmark mask construction, skip attention timing"
    )
    args = parser.parse_args()

    if not is_mfa_available():
        print("[INFO] MFA extension not available — attention benchmarks SKIPPED.")
        args.skip_attention = True

    print(f"mlx-mfa spatial mask benchmarks  (MFA: {'ON' if is_mfa_available() else 'OFF'})")
    print(f"N > {MAX_N_FOR_ATTN:,} → attention SKIPPED (mask construction always runs)")

    all_rows: list[dict] = []
    all_rows.extend(run_2d_benchmarks(args.skip_attention))
    all_rows.extend(run_3d_benchmarks(args.skip_attention))
    all_rows.extend(run_segment_benchmarks(args.skip_attention))
    all_rows.extend(run_adaptive_benchmarks(args.skip_attention))

    update_results_md(all_rows)


if __name__ == "__main__":
    main()
