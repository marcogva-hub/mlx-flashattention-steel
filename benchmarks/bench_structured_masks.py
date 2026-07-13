#!/usr/bin/env python3
"""Measure sparse-mask traffic in isolation.

One process measures one arm over the full grid. The shell launcher runs five
fresh processes per arm and order, so compilation/cache state cannot cross the
comparison. The on-the-fly arm is a probe-only C++ source variant: it retains
the bool mask argument and the V6 NAX tile body, but replaces only the mask
load predicate with make_sliding_window_mask's arithmetic predicate.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _ext
from mlx_mfa.attention import make_sliding_window_mask


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "structured_masks"
GRID = [(n, d, dt) for n in (2048, 4096, 8192)
        for d in (64, 128) for dt in ("fp16", "bf16")]
WINDOWS = {2048: 128, 4096: 256, 8192: 512}
BATCH = 1
HEADS = 16
BLOCK_TILE = 32
DISPATCHES = 20


def _dtype(name: str):
    return mx.float16 if name == "fp16" else mx.bfloat16


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    return float(mx.sum(af * bf) /
                 (mx.sqrt(mx.sum(af * af)) * mx.sqrt(mx.sum(bf * bf))))


def _make_case(n: int, d: int, dtype_name: str, window_size: int):
    dtype = _dtype(dtype_name)
    mx.random.seed(8127 + n + d + (0 if dtype_name == "fp16" else 1))
    q = mx.random.normal((BATCH, HEADS, n, d)).astype(dtype)
    k = mx.random.normal((BATCH, HEADS, n, d)).astype(dtype)
    v = mx.random.normal((BATCH, HEADS, n, d)).astype(dtype)
    mask = make_sliding_window_mask(n, window_size, head_dim=d, causal=False)
    mx.eval(q, k, v, mask)

    # Independent fp32 oracle for the same block-level semantics.
    expanded = mx.repeat(mx.repeat(mask, BLOCK_TILE, axis=-2), BLOCK_TILE, axis=-1)
    bias = mx.where(expanded, mx.array(0.0, mx.float32),
                    mx.array(-float("inf"), mx.float32))
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=1.0 / math.sqrt(d), mask=bias)
    mx.eval(ref)
    return q, k, v, mask, ref


def _call(arm: str, q, k, v, mask, d: int, window_size: int):
    kwargs = dict(
        block_tile=BLOCK_TILE,
        causal=False,
        scale=1.0 / math.sqrt(d),
        kernel_version="v6nax_sparse",
    )
    if arm == "onthefly":
        kwargs.update(structured_window_probe=True,
                      structured_window_size=window_size)
    return _ext.sparse_attention_forward(q, k, v, mask, **kwargs)


def _time_call(arm: str, q, k, v, mask, d: int, window_size: int,
               dispatches: int):
    # The first call also proves the generated source compiles before timing.
    out = _call(arm, q, k, v, mask, d, window_size)
    mx.eval(out)
    mx.synchronize()
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(dispatches):
            out = _call(arm, q, k, v, mask, d, window_size)
            mx.eval(out)
            mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0 / dispatches)
    return out, samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("mask", "onthefly"), required=True)
    parser.add_argument("--order", choices=("mask-first", "onthefly-first"), required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--dispatches", type=int, default=DISPATCHES)
    parser.add_argument(
        "--window-size", type=int, default=None,
        help="Override the default window for a targeted real-density probe.",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Restrict to one cell as N,D,dtype, for example 4096,128,bf16.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.session < 1:
        raise SystemExit("--session must be >= 1")

    if args.only is None:
        cells = GRID
    else:
        try:
            n, d, dtype_name = args.only.split(",")
            cells = [(int(n), int(d), dtype_name)]
        except ValueError as exc:
            raise SystemExit("--only must be N,D,dtype") from exc
        if cells[0] not in GRID:
            raise SystemExit("--only must select a cell from the benchmark grid")

    rows = []
    for n, d, dtype_name in cells:
        window_size = args.window_size or WINDOWS[n]
        q, k, v, mask, ref = _make_case(n, d, dtype_name, window_size)
        out, samples = _time_call(
            args.arm, q, k, v, mask, d, window_size, args.dispatches
        )
        mx.eval(out, ref)
        density = float(mx.mean(mask.astype(mx.float32)))
        rows.append({
            "arm": args.arm,
            "order": args.order,
            "session": args.session,
            "N": n,
            "D": d,
            "dtype": dtype_name,
            "window_size": window_size,
            "density": density,
            "cosine_vs_fp32_oracle": _cosine(out, ref),
            "max_abs_vs_fp32_oracle": float(mx.max(mx.abs(
                out.astype(mx.float32) - ref))),
            "ms_per_dispatch_samples": samples,
            "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
            "which_binary": (
                "v6nax_sparse_mask_load"
                if args.arm == "mask" else
                "v6nax_sparse_structured_window_probe"
            ),
        })

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "device": str(mx.default_device()),
        "arm": args.arm,
        "order": args.order,
        "session": args.session,
        "dispatches_per_sample": args.dispatches,
        "rows": rows,
    }
    output = args.output or (DEFAULT_OUT /
                             f"{args.order}_{args.arm}_s{args.session}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "arm": args.arm,
        "order": args.order,
        "session": args.session,
        "rows": len(rows),
        "min_cosine": min(r["cosine_vs_fp32_oracle"] for r in rows),
        "median_ms": {
            f"N{r['N']}_D{r['D']}_{r['dtype']}": r["median_ms"]
            for r in rows
        },
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
