#!/usr/bin/env python3
"""Recheck the dense D=128 spot against the real fp16 MLX SDPA path.

Each arm is run in a fresh process.  The direct MLX arm has no mlx-mfa
telemetry by construction; the wrapper arm proves the public ``backend=sdpa``
route and is compared elementwise with the direct arm.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import flash_attention


SESSIONS = 5
WARMUPS = 2
DISPATCHES_PER_SAMPLE = 20
D, N = 128, 4096
SCALE = 1.0 / math.sqrt(D)


def evaluate(value):
    mx.eval(*value) if isinstance(value, (tuple, list)) else mx.eval(value)
    mx.synchronize()


def cosine(a, b):
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    value = mx.sum(af * bf) / mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    mx.eval(value)
    return float(value.item())


def make_inputs():
    mx.random.seed(20260713)
    q = mx.random.normal((1, 4, N, D)).astype(mx.float16)
    k = mx.random.normal((1, 4, N, D)).astype(mx.float16)
    v = mx.random.normal((1, 4, N, D)).astype(mx.float16)
    return q, k, v


def arm_fn(arm, q, k, v):
    if arm == "public_nax":
        return lambda: flash_attention(q, k, v, scale=SCALE, causal=False)
    if arm == "mlx_sdpa":
        return lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=SCALE,
        )
    if arm == "public_sdpa":
        return lambda: flash_attention(
            q, k, v, scale=SCALE, causal=False, backend="sdpa",
        )
    raise ValueError(arm)


def time_arm(fn):
    for _ in range(WARMUPS):
        for _ in range(DISPATCHES_PER_SAMPLE):
            evaluate(fn())
    samples = []
    for _ in range(SESSIONS):
        started = time.perf_counter()
        for _ in range(DISPATCHES_PER_SAMPLE):
            evaluate(fn())
        samples.append((time.perf_counter() - started) * 1000.0 / DISPATCHES_PER_SAMPLE)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
        "n": len(samples),
    }


def run(arm):
    q, k, v = make_inputs()
    fn = arm_fn(arm, q, k, v)
    with dtrace.capture() as trace:
        probe = fn()
        evaluate(probe)
    terminal_trace = [item for item in trace if not item[1].startswith("[reentrant]")]

    # Independent fp32 oracle.  It is evaluated for correction only, never timed.
    oracle = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=SCALE,
    )
    evaluate((probe, oracle))
    correction = {
        "cos": cosine(probe, oracle),
        "finite": bool(mx.all(mx.isfinite(probe)).item()),
        "max_abs_vs_fp32_oracle": float(mx.max(mx.abs(
            probe.astype(mx.float32) - oracle
        )).item()),
    }
    if correction["cos"] < 0.999 or not correction["finite"]:
        raise RuntimeError(f"correction failed for {arm}: {correction}")

    # The public forced-SDPA wrapper must be the same plain MLX operation as
    # the direct baseline.  This is a path check, not a timed arm.
    direct = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
    with dtrace.capture() as wrapper_trace:
        wrapped = flash_attention(q, k, v, scale=SCALE, causal=False, backend="sdpa")
        evaluate((direct, wrapped))
    wrapper_terminal = [item for item in wrapper_trace if not item[1].startswith("[reentrant]")]
    wrapper_delta = float(mx.max(mx.abs(
        direct.astype(mx.float32) - wrapped.astype(mx.float32)
    )).item())
    wrapper_cos = cosine(direct, wrapped)
    if not wrapper_terminal or wrapper_terminal[-1][0] != "sdpa":
        raise RuntimeError(f"public SDPA engagement failed: {wrapper_trace}")
    if wrapper_delta != 0.0 or wrapper_cos != 1.0:
        raise RuntimeError(
            f"public SDPA wrapper diverged from direct MLX SDPA: "
            f"delta={wrapper_delta}, cos={wrapper_cos}"
        )

    timing = time_arm(fn)
    print(f"{arm}: median={timing['median_ms']:.6f}ms trace={terminal_trace}", flush=True)
    return {
        "arm": arm,
        "shape": {"B": 1, "Hq": 4, "Hkv": 4, "N": N, "D": D,
                  "dtype": "float16", "causal": False, "mask": None},
        "which_binary": {
            "mlx_mfa_trace": trace,
            "terminal_trace": terminal_trace,
            "expected": {
                "public_nax": "nax_dense",
                "public_sdpa": "sdpa",
                "mlx_sdpa": "outside_mlx_mfa",
            }[arm],
            "public_sdpa_vs_direct_mlx": {
                "wrapper_trace": wrapper_trace,
                "wrapper_terminal_trace": wrapper_terminal,
                "max_abs": wrapper_delta,
                "cos": wrapper_cos,
            },
        },
        "correction": correction,
        "timing": timing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("public_nax", "mlx_sdpa", "public_sdpa"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = run(args.arm)
    payload = {
        "schema": "mlx-mfa.dense-reference-recheck.v1",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "mlx": importlib.metadata.version("mlx"),
        "platform": platform.platform(),
        "method": {
            "sessions": SESSIONS,
            "warmups": WARMUPS,
            "dispatches_per_sample": DISPATCHES_PER_SAMPLE,
            "process_isolated": True,
            "orders": "run externally in both arm orders",
        },
        "row": row,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
