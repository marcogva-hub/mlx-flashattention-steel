#!/usr/bin/env python3
"""D=256 V6NAX head-subtile gate benchmark.

The NAX arm calls the expert binding directly.  That binding is the engagement
proof: D=256 rejects unless ``force_v6nax=True`` and its generated MSL has a
distinct D-subtile cache key.  SDPA and legacy STEEL are separately traced.
"""
from __future__ import annotations

import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import _ext, flash_attention


REPO = Path(__file__).resolve().parents[1]
SESSIONS = 5
SAMPLES = 5
WARMUP = 2
CELLS = (
    # N, B*H, dtype, causal.  The set spans the prior NO-GO's length,
    # dtype, causal and resource-pressure axes without inventing a route.
    (1024, 1, mx.float16, False), (1024, 1, mx.bfloat16, True),
    (4096, 1, mx.float16, False), (4096, 1, mx.float16, True),
    (4096, 1, mx.bfloat16, False), (4096, 1, mx.bfloat16, True),
    (4096, 4, mx.float16, False), (4096, 4, mx.bfloat16, False),
    (8192, 1, mx.float16, False), (8192, 1, mx.float16, True),
    (8192, 1, mx.bfloat16, False), (8192, 1, mx.bfloat16, True),
)


def _cosine(a: mx.array, b: mx.array) -> float:
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    value = mx.sum(a * b) / mx.sqrt(mx.sum(a * a) * mx.sum(b * b))
    mx.eval(value)
    return float(value.item())


def _timed(call):
    samples: list[float] = []
    for _ in range(SESSIONS):
        for _ in range(WARMUP):
            mx.eval(call())
        for _ in range(SAMPLES):
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
        "samples_ms": samples,
    }


def _trace(call):
    with dtrace.capture() as trace:
        out = call()
        mx.eval(out)
    return out, trace


def _dtype_name(dtype: mx.Dtype) -> str:
    return "bf16" if dtype == mx.bfloat16 else "fp16"


def _cell(N: int, H: int, dtype: mx.Dtype, causal: bool):
    D = 256
    seed = 256700 + N + H * 17 + int(causal) + (1 if dtype == mx.bfloat16 else 0)
    mx.random.seed(seed)
    q = (mx.random.normal((1, H, N, D)) * 0.05).astype(dtype)
    k = (mx.random.normal((1, H, N, D)) * 0.05).astype(dtype)
    v = (mx.random.normal((1, H, N, D)) * 0.05).astype(dtype)
    mx.eval(q, k, v)
    scale = 1.0 / math.sqrt(D)

    arms = {
        "nax_subtile": lambda: _ext.v6_nax_forward(q, k, v, causal, True)[0],
        "sdpa": lambda: flash_attention(q, k, v, causal=causal, scale=scale, backend="sdpa"),
        "steel": lambda: flash_attention(q, k, v, causal=causal, scale=scale, backend="mfa"),
    }
    nax = arms["nax_subtile"]()
    sdpa, sdpa_trace = _trace(arms["sdpa"])
    steel, steel_trace = _trace(arms["steel"])
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=scale, mask="causal" if causal else None,
    )
    mx.eval(nax, sdpa, steel, ref)
    cos = {"nax_subtile_fp32": _cosine(nax, ref), "steel_fp32": _cosine(steel, ref)}
    if min(cos.values()) < 0.999:
        raise RuntimeError(f"D256 correctness gate failed: {cos}")
    if not sdpa_trace or sdpa_trace[-1][0] != "sdpa":
        raise RuntimeError(f"SDPA engagement failed: {sdpa_trace}")
    if not steel_trace or steel_trace[-1][0] != "mfa_primitive":
        raise RuntimeError(f"STEEL engagement failed: {steel_trace}")
    # The NAX arm is a direct force_v6nax=True expert binding.  A fallback is
    # impossible by contract; retain a nonzero delta to SDPA as corroboration.
    delta_nax_sdpa = float(mx.max(mx.abs(nax.astype(mx.float32) - sdpa.astype(mx.float32))).item())
    if delta_nax_sdpa == 0.0:
        raise RuntimeError("NAX and SDPA fingerprint unexpectedly collapsed")

    timings = {}
    for order in ("sdpa,nax_subtile,steel", "steel,nax_subtile,sdpa"):
        timings[order] = {arm: _timed(arms[arm]) for arm in order.split(",")}
    ratios = {
        order: {
            "sdpa_over_nax": timing["sdpa"]["median_ms"] / timing["nax_subtile"]["median_ms"],
            "steel_over_nax": timing["steel"]["median_ms"] / timing["nax_subtile"]["median_ms"],
        }
        for order, timing in timings.items()
    }
    return {
        "N": N,
        "B": 1,
        "H": H,
        "D": D,
        "dtype": _dtype_name(dtype),
        "causal": causal,
        "correction": cos,
        "which_binary": {
            "nax_subtile": "direct _ext.v6_nax_forward(force_v6nax=True), D=256 expert source",
            "nax_vs_sdpa_max_abs": delta_nax_sdpa,
            "sdpa_trace": sdpa_trace,
            "steel_trace": steel_trace,
        },
        "timing": timings,
        "ratios": ratios,
    }


def main():
    output = REPO / "benchmarks/results/headdim_stage1_d256.json"
    payload = {
        "stage": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mlx_version": mx.__version__,
        "platform": platform.platform(),
        "method": {"sessions": SESSIONS, "samples": SAMPLES, "warmup": WARMUP,
                   "orders": ["sdpa,nax_subtile,steel", "steel,nax_subtile,sdpa"]},
        "cells": [],
    }
    for params in CELLS:
        row = _cell(*params)
        payload["cells"].append(row)
        print(
            f"N={row['N']} H={row['H']} {row['dtype']} causal={row['causal']}: "
            f"SDPA/NAX=" + ", ".join(
                f"{v['sdpa_over_nax']:.3f}x" for v in row["ratios"].values()
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
