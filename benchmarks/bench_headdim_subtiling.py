#!/usr/bin/env python3
"""Focused evidence harness for the head-dimension sub-tiling investigation.

Stage 0 intentionally measures the historical BT=64 audit geometry through the
public direct sparse helper.  It keeps the original 64-wide mask as input, so
the ``expanded`` arm proves the bounded BT64->BT32 gate rather than merely
timing a separately-authored BT32 mask.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa.lcsa_nax import _bool_mask_to_float_bias, sparse_attention_nax


REPO = Path(__file__).resolve().parents[1]
N = 4096
D = 64
H = 1
BLOCK_TILE = 64
SESSIONS = 5
SAMPLES_PER_SESSION = 8
WARMUP = 2
DENSITIES = (0.10, 0.30)
DTYPES = (mx.float16, mx.bfloat16)


@contextmanager
def _kernel_version(value: str | None):
    previous = os.environ.get("MFA_LCSA_KERNEL_VERSION")
    try:
        if value is None:
            os.environ.pop("MFA_LCSA_KERNEL_VERSION", None)
        else:
            os.environ["MFA_LCSA_KERNEL_VERSION"] = value
        yield
    finally:
        if previous is None:
            os.environ.pop("MFA_LCSA_KERNEL_VERSION", None)
        else:
            os.environ["MFA_LCSA_KERNEL_VERSION"] = previous


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _inputs(dtype: mx.Dtype, requested_density: float):
    # Match the original BT64 audit exactly, so Stage 0 reports a genuine
    # before/after rather than a nearby random-mask sample.
    mx.random.seed(2501 + D + int(requested_density * 100))
    q = (mx.random.normal((1, H, N, D)) * 0.05).astype(dtype)
    k = (mx.random.normal((1, H, N, D)) * 0.05).astype(dtype)
    v = (mx.random.normal((1, H, N, D)) * 0.05).astype(dtype)
    rng = np.random.default_rng(3101 + int(requested_density * 100))
    mask64_np = rng.random((N // BLOCK_TILE, N // BLOCK_TILE)) < requested_density
    # Every Q block must have one live K block; this makes the cosine meaningful.
    mask64_np[:, 0] = True
    mask64 = mx.array(mask64_np)
    mask32 = mx.repeat(mx.repeat(mask64, 2, axis=-2), 2, axis=-1)
    mx.eval(q, k, v, mask64, mask32)
    return q, k, v, mask64, mask32


def _call(arm: str, q, k, v, mask64, mask32):
    if arm == "scalar":
        with _kernel_version("v1"):
            return sparse_attention_nax(q, k, v, mask64, block_tile=BLOCK_TILE)
    if arm == "expanded":
        with _kernel_version(None):
            return sparse_attention_nax(q, k, v, mask64, block_tile=BLOCK_TILE)
    if arm == "native32":
        with _kernel_version(None):
            return sparse_attention_nax(q, k, v, mask32, block_tile=32)
    raise ValueError(f"unknown arm {arm}")


def _trace(call):
    with dtrace.capture() as trace:
        out = call()
        mx.eval(out)
    if not trace:
        raise RuntimeError("missing sparse dispatch trace")
    return out, trace


def _cosine(a, b) -> float:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    value = mx.sum(af * bf) / mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    mx.eval(value)
    return float(value.item())


def _time(call) -> dict[str, object]:
    samples: list[float] = []
    for _ in range(SESSIONS):
        for _ in range(WARMUP):
            mx.eval(call())
        for _ in range(SAMPLES_PER_SESSION):
            mx.synchronize()
            start = time.perf_counter()
            mx.eval(call())
            mx.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": samples,
    }


def _fp32_oracle(q, k, v, mask64):
    bias = _bool_mask_to_float_bias(mask64, BLOCK_TILE, N, N, mx.float32)
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=1.0 / math.sqrt(D), mask=bias,
    )


def _stage0(args):
    payload: dict[str, object] = {
        "stage": 0,
        "purpose": "BT64 historical-audit recovery at exact geometry",
        "commit": _git_head(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mlx_version": mx.__version__,
        "platform": platform.platform(),
        "geometry": {"B": 1, "H": H, "N": N, "D": D, "block_tile": BLOCK_TILE},
        "method": {
            "sessions": SESSIONS,
            "samples_per_session": SAMPLES_PER_SESSION,
            "warmup": WARMUP,
            "orders": ["scalar,expanded,native32", "native32,expanded,scalar"],
            "sampling_asymmetry": "none; both arms use five sessions and eight dispatches per session",
        },
        "cells": [],
    }
    orders = ("scalar,expanded,native32", "native32,expanded,scalar")
    for dtype in DTYPES:
        for density in DENSITIES:
            q, k, v, mask64, mask32 = _inputs(dtype, density)
            calls = {
                arm: lambda arm=arm: _call(arm, q, k, v, mask64, mask32)
                for arm in ("scalar", "expanded", "native32")
            }
            outputs, traces = {}, {}
            for arm, call in calls.items():
                outputs[arm], traces[arm] = _trace(call)
            oracle = _fp32_oracle(q, k, v, mask64)
            mx.eval(oracle, *outputs.values())
            expanded_vs_native32 = float(mx.max(mx.abs(
                outputs["expanded"].astype(mx.float32) - outputs["native32"].astype(mx.float32)
            )).item())
            expanded_cos = _cosine(outputs["expanded"], oracle)
            if traces["scalar"][-1][0] != "scalar_fallback":
                raise RuntimeError(f"scalar arm did not engage scalar fallback: {traces['scalar']}")
            if traces["expanded"][-1][0] != "v6nax_sparse":
                raise RuntimeError(f"expanded arm did not engage V6NAX: {traces['expanded']}")
            if traces["native32"][-1][0] != "v6nax_sparse":
                raise RuntimeError(f"BT32 control did not engage V6NAX: {traces['native32']}")
            if expanded_vs_native32 != 0.0:
                raise RuntimeError(f"BT64 expansion drifted from native BT32: {expanded_vs_native32}")
            if expanded_cos < 0.999:
                raise RuntimeError(f"expanded BT64 cosine below gate: {expanded_cos}")
            timings = {}
            for order in orders:
                timing = {arm: _time(calls[arm]) for arm in order.split(",")}
                timings[order] = timing
            scalar_over_expanded = {
                order: timings[order]["scalar"]["median_ms"] / timings[order]["expanded"]["median_ms"]
                for order in orders
            }
            row = {
                "dtype": str(dtype),
                "requested_density": density,
                "actual_density": float(mx.mean(mask64.astype(mx.float32)).item()),
                "traces": traces,
                "expanded_vs_native32_max_abs": expanded_vs_native32,
                "expanded_vs_fp32_cos": expanded_cos,
                "timing": timings,
                "scalar_over_expanded": scalar_over_expanded,
            }
            payload["cells"].append(row)
            print(
                f"{dtype} density={row['actual_density']:.3f}: "
                f"scalar/expanded="
                f"{scalar_over_expanded[orders[0]]:.2f}x,"
                f"{scalar_over_expanded[orders[1]]:.2f}x "
                f"cos={expanded_cos:.9f} trace={traces['expanded'][-1][0]}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("0",), default="0")
    parser.add_argument(
        "--out", type=Path,
        default=REPO / "benchmarks/results/headdim_step0_bt64_exact.json",
    )
    args = parser.parse_args()
    _stage0(args)


if __name__ == "__main__":
    main()
