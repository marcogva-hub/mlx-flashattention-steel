#!/usr/bin/env python3
"""Final public packed-varlen closure benchmark.

The candidate arm is the public opt-in route, not a forced raw symbol.  The
other arms are the current STEEL expert and per-segment MLX SDPA.  Run this
script twice in separate foreground processes with the two order values.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import _ext, flash_attention_varlen


GEOMETRIES = {
    "seed_aligned": [3226, 3226, 3226, 3226, 2434, 1642, 1642, 1642, 1642,
                     1246, 1642, 1642, 1642, 1642, 1246, 850, 850, 850, 850, 652],
    "seed_shifted": [454, 850, 850, 850, 850, 256, 850, 1642, 1642, 1642,
                     1642, 454, 850, 1642, 1642, 1642, 1642, 454, 1642, 3226,
                     3226, 3226, 3226, 850],
}
SESSIONS = 5
WARMUPS = 2
TILE = (32, 32, 2)


def prefix(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return values


def tiles(lengths):
    return mx.array(prefix([math.ceil(length / TILE[0]) for length in lengths]), dtype=mx.int32)


def evaluate(value):
    if isinstance(value, (tuple, list)):
        mx.eval(*value)
    else:
        mx.eval(value)
    mx.synchronize()


def cosine(a, b):
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    value = mx.sum(af * bf) / mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    mx.eval(value)
    return float(value.item())


def stats(samples):
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(np.asarray(samples), 95)),
        "samples_ms": samples,
        "n": len(samples),
    }


def time_arm(fn):
    for _ in range(WARMUPS):
        evaluate(fn())
    samples = []
    for _ in range(SESSIONS):
        mx.synchronize()
        started = time.perf_counter()
        evaluate(fn())
        samples.append((time.perf_counter() - started) * 1000.0)
    return stats(samples)


def run_cell(geometry, lengths, gqa, dtype_name, causal, order):
    total = sum(lengths)
    hq, hk, d = 16, 16 // gqa, 128
    dtype = mx.float16 if dtype_name == "fp16" else mx.bfloat16
    mx.random.seed(731000 + total + gqa * 17 + int(causal) * 31 + (1 if dtype_name == "bf16" else 0))
    q = (mx.random.normal((1, hq, total, d)) * 0.05).astype(dtype)
    k = (mx.random.normal((1, hk, total, d)) * 0.05).astype(dtype)
    v = (mx.random.normal((1, hk, total, d)) * 0.05).astype(dtype)
    cu_list = prefix(lengths)
    cu = mx.array(cu_list, dtype=mx.int32)
    tile_arr = tiles(lengths)
    scale = 1.0 / math.sqrt(d)
    evaluate((q, k, v, cu, tile_arr))

    def public_nax():
        return flash_attention_varlen(
            q, k, v, cu, cu, max(lengths), max(lengths),
            scale=scale, causal=causal,
        )

    def steel():
        return _ext.mfa_attention_varlen_forward(
            q, k, v, cu, cu, tile_arr, scale, causal
        )[0]

    def sdpa():
        return mx.concatenate([
            mx.fast.scaled_dot_product_attention(
                q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop],
                scale=scale, mask="causal" if causal else None,
            )
            for start, stop in zip(cu_list[:-1], cu_list[1:])
        ], axis=2)

    # Engagement and correction are checked before timing.
    with dtrace.capture() as trace:
        public_probe = public_nax()
        evaluate(public_probe)
    steel_probe, sdpa_probe = steel(), sdpa()
    oracle = mx.concatenate([
        mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32)[:, :, start:stop],
            k.astype(mx.float32)[:, :, start:stop],
            v.astype(mx.float32)[:, :, start:stop],
            scale=scale, mask="causal" if causal else None,
        )
        for start, stop in zip(cu_list[:-1], cu_list[1:])
    ], axis=2)
    evaluate((public_probe, steel_probe, sdpa_probe, oracle))
    segment_cos = [
        cosine(public_probe[:, :, start:stop], oracle[:, :, start:stop])
        for start, stop in zip(cu_list[:-1], cu_list[1:])
    ]
    correction = {
        "global_cos": cosine(public_probe, oracle),
        "min_segment_cos": min(segment_cos),
        "finite": bool(mx.all(mx.isfinite(public_probe)).item()),
    }
    deltas = {
        "public_vs_steel_max_abs": float(mx.max(mx.abs(public_probe.astype(mx.float32) - steel_probe.astype(mx.float32))).item()),
        "public_vs_sdpa_max_abs": float(mx.max(mx.abs(public_probe.astype(mx.float32) - sdpa_probe.astype(mx.float32))).item()),
    }
    if correction["global_cos"] < 0.999 or correction["min_segment_cos"] < 0.999 or not correction["finite"]:
        raise RuntimeError(f"correction failed: {geometry}/{gqa}/{dtype_name}/{causal}: {correction}")
    if not all(value > 0.0 for value in deltas.values()):
        raise RuntimeError(f"which-binary delta failed: {deltas}")
    if trace != [("varlen_v6nax", "opt-in beta-3 packed V6 NAX (BQ32/BK32/WM2 explicit)")]:
        raise RuntimeError(f"public which-binary failed: {trace}")

    timings = {name: time_arm({"public": public_nax, "steel": steel, "sdpa": sdpa}[name]) for name in order}
    ratio = {
        "sdpa_over_public_nax": timings["sdpa"]["median_ms"] / timings["public"]["median_ms"],
        "steel_over_public_nax": timings["steel"]["median_ms"] / timings["public"]["median_ms"],
    }
    print(f"{geometry} gqa={gqa} {dtype_name} causal={int(causal)} "
          f"SDPA/NAX={ratio['sdpa_over_public_nax']:.3f}x "
          f"STEEL/NAX={ratio['steel_over_public_nax']:.3f}x", flush=True)
    return {
        "geometry": geometry, "total_tokens": total, "segments": len(lengths),
        "gqa": gqa, "Hq": hq, "Hkv": hk, "D": d, "dtype": dtype_name,
        "causal": causal, "tile": {"bq": 32, "bk": 32, "wm": 2},
        "which_binary": {"public_trace": trace, **deltas,
                         "public": "flash_attention_varlen -> v6_nax_varlen_forward",
                         "steel": "_ext.mfa_attention_varlen_forward",
                         "sdpa": "per-segment mx.fast.scaled_dot_product_attention"},
        "correction": correction, "timing": timings, "ratio": ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("public,steel,sdpa", "sdpa,steel,public"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["MFA_ENABLE_VARLEN_NAX"] = "1"
    order = args.order.split(",")
    rows = []
    for geometry, lengths in GEOMETRIES.items():
        for gqa in (2, 8):
            for dtype in ("fp16", "bf16"):
                for causal in (False, True):
                    rows.append(run_cell(geometry, lengths, gqa, dtype, causal, order))
    payload = {
        "schema": "mlx-mfa.final-varlen-public-closure.v1",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": str(Path(__file__).resolve()),
        "mlx": importlib.metadata.version("mlx"), "platform": platform.platform(),
        "order": order, "method": {"sessions": SESSIONS, "warmups": WARMUPS,
                                   "samples_per_session": 1, "sampling_asymmetry": "none"},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
