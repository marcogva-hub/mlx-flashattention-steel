#!/usr/bin/env python3
"""Three-arm packed-varlen benchmark: V6 NAX vs STEEL vs split SDPA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _ext


GEOMETRIES = {
    "seed-aligned": [3226, 3226, 3226, 3226, 2434, 1642, 1642, 1642, 1642,
                     1246, 1642, 1642, 1642, 1642, 1246, 850, 850, 850, 850, 652],
    "seed-shifted": [454, 850, 850, 850, 850, 256, 850, 1642, 1642, 1642,
                     1642, 454, 850, 1642, 1642, 1642, 1642, 454, 1642, 3226,
                     3226, 3226, 3226, 850],
    "equal-256": [256] * 16,
    "equal-1024": [1024] * 8,
    "heterogeneous": [1, 17, 65, 129, 257, 513, 1025, 2049],
    "single-8192": [8192],
}


def prefix(lengths: list[int]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def tile_offsets(lengths: list[int], bq: int) -> mx.array:
    return mx.array(prefix([math.ceil(length / bq) for length in lengths]), dtype=mx.int32)


def evaluate(value) -> None:
    if isinstance(value, (tuple, list)):
        mx.eval(*value)
    else:
        mx.eval(value)
    mx.synchronize()


def stats(samples: list[float]) -> dict:
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(np.asarray(samples), 95)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    return float(mx.sum(af * bf) / (mx.sqrt(mx.sum(af * af)) * mx.sqrt(mx.sum(bf * bf))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=GEOMETRIES, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), required=True)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--order", choices=("nax-steel-sdpa", "sdpa-steel-nax"), required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--cooldown", type=float, default=0.5)
    parser.add_argument("--bq", type=int)
    parser.add_argument("--bk", type=int)
    parser.add_argument("--wm", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    default_bq = 32 if args.head_dim == 64 else 64
    default_wm = 2 if args.head_dim == 64 else 4
    bq = args.bq or default_bq
    bk = args.bk or 32
    wm = args.wm or default_wm
    os.environ["MFA_V6_NAX_BQ"] = str(bq)
    os.environ["MFA_V6_NAX_BK"] = str(bk)
    os.environ["MFA_V6_NAX_WM"] = str(wm)

    lengths = GEOMETRIES[args.geometry]
    total = sum(lengths)
    dtype = mx.float16 if args.dtype == "fp16" else mx.bfloat16
    scale = 1.0 / math.sqrt(args.head_dim)
    mx.random.seed(20260712)
    q = mx.random.normal((1, args.heads, total, args.head_dim)).astype(dtype)
    k = mx.random.normal((1, args.heads, total, args.head_dim)).astype(dtype)
    v = mx.random.normal((1, args.heads, total, args.head_dim)).astype(dtype)
    cu_list = prefix(lengths)
    cu = mx.array(cu_list, dtype=mx.int32)
    nax_tiles = tile_offsets(lengths, bq)
    steel_tiles = tile_offsets(lengths, 32)
    evaluate((q, k, v, cu, nax_tiles, steel_tiles))

    def nax():
        return _ext.v6_nax_varlen_forward(q, k, v, cu, cu, nax_tiles, scale, False)[0]

    def steel():
        return _ext.mfa_attention_varlen_forward(q, k, v, cu, cu, steel_tiles, scale, False)[0]

    def sdpa():
        return mx.concatenate([
            mx.fast.scaled_dot_product_attention(
                q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop], scale=scale
            )
            for start, stop in zip(cu_list[:-1], cu_list[1:])
        ], axis=2)

    outputs = {name: fn() for name, fn in (("nax", nax), ("steel", steel), ("sdpa", sdpa))}
    evaluate(list(outputs.values()))
    segment_cos = []
    for start, stop in zip(cu_list[:-1], cu_list[1:]):
        segment_cos.append(cosine(outputs["nax"][:, :, start:stop], outputs["sdpa"][:, :, start:stop]))
    correctness = {
        "global_cos_nax_sdpa": cosine(outputs["nax"], outputs["sdpa"]),
        "min_segment_cos_nax_sdpa": min(segment_cos),
        "global_cos_nax_steel": cosine(outputs["nax"], outputs["steel"]),
        "max_abs_nax_steel": float(mx.max(mx.abs(outputs["nax"].astype(mx.float32) - outputs["steel"].astype(mx.float32)))),
        "max_abs_nax_sdpa": float(mx.max(mx.abs(outputs["nax"].astype(mx.float32) - outputs["sdpa"].astype(mx.float32)))),
        "nax_has_nan": bool(mx.any(mx.isnan(outputs["nax"]))),
    }
    if correctness["global_cos_nax_sdpa"] < 0.999 or correctness["min_segment_cos_nax_sdpa"] < 0.999:
        raise RuntimeError(f"correctness gate failed: {correctness}")
    if correctness["max_abs_nax_steel"] == 0 or correctness["max_abs_nax_sdpa"] == 0:
        raise RuntimeError(f"engagement delta gate failed: {correctness}")

    functions = {"nax": nax, "steel": steel, "sdpa": sdpa}
    order = args.order.split("-")
    timings = {}
    for name in order:
        fn = functions[name]
        for _ in range(args.warmup):
            evaluate(fn())
        samples = []
        for index in range(args.sessions):
            start = time.perf_counter()
            for _ in range(args.iterations):
                evaluate(fn())
            samples.append((time.perf_counter() - start) * 1000 / args.iterations)
            if index + 1 < args.sessions and args.cooldown:
                time.sleep(args.cooldown)
        timings[name] = stats(samples)
        mx.clear_cache()

    payload = {
        "schema": "mlx-mfa.varlen-packed-nax.v1",
        "date": "2026-07-12",
        "hardware": platform.machine(),
        "macos": platform.mac_ver()[0],
        "mlx": importlib.metadata.version("mlx"),
        "geometry": args.geometry,
        "lengths": lengths,
        "total_tokens": total,
        "num_segments": len(lengths),
        "dtype": args.dtype,
        "head_dim": args.head_dim,
        "heads": args.heads,
        "tile": {"bq": bq, "bk": bk, "wm": wm},
        "order": args.order,
        "iterations_per_sample": args.iterations,
        "correctness": correctness,
        "engagement": {
            "nax": "_ext.v6_nax_varlen_forward",
            "steel": "_ext.mfa_attention_varlen_forward",
            "sdpa": "mx.fast.scaled_dot_product_attention per segment",
        },
        "timings": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
