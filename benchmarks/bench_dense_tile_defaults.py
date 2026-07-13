#!/usr/bin/env python3
"""Measure the D=128 dense NAX tile-default candidate.

One invocation measures one arm in one fresh process.  The shell/driver owns
the two arm orders and captures ``MFA_V6_DUMP_SOURCE`` stderr; this keeps the
compiled tile fingerprint separate from the requested environment.
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
from mlx_mfa import flash_attention


SESSIONS = 5
WARMUPS = 2
DISPATCHES_PER_SAMPLE = 20
D = 128
SCALE = 1.0 / math.sqrt(D)


def _bh_shape(bh: int) -> tuple[int, int]:
    if bh == 8:
        return 1, 8
    if bh == 32:
        return 2, 16
    if bh == 64:
        return 4, 16
    raise ValueError(f"unsupported B*H={bh}")


def _dtype(name: str):
    return {"fp16": mx.float16, "bf16": mx.bfloat16}[name]


def _eval(*values):
    mx.eval(*values)
    mx.synchronize()


def _cosine(a, b) -> float:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    value = mx.sum(af * bf) / mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    _eval(value)
    return float(value.item())


def _make_inputs(b: int, h: int, n: int, dtype_name: str, causal: bool):
    # Stable shape-specific data lets the two arms be compared without making
    # the timing path depend on a different random input realization.
    seed = 20260713 + n + b * 17 + h * 31 + (1 if dtype_name == "bf16" else 0)
    mx.random.seed(seed)
    dtype = _dtype(dtype_name)
    q = mx.random.normal((b, h, n, D)).astype(dtype)
    k = mx.random.normal((b, h, n, D)).astype(dtype)
    v = mx.random.normal((b, h, n, D)).astype(dtype)
    _eval(q, k, v)
    return q, k, v


def _arm_env(arm: str) -> dict[str, str]:
    if arm == "default":
        return {}
    if arm == "candidate":
        return {"MFA_V6_NAX_BQ": "64", "MFA_V6_NAX_BK": "32", "MFA_V6_NAX_WM": "2"}
    raise ValueError(arm)


def _time(fn):
    for _ in range(WARMUPS):
        for _ in range(DISPATCHES_PER_SAMPLE):
            _eval(fn())
    samples = []
    for _ in range(SESSIONS):
        started = time.perf_counter()
        for _ in range(DISPATCHES_PER_SAMPLE):
            _eval(fn())
        samples.append((time.perf_counter() - started) * 1000.0 / DISPATCHES_PER_SAMPLE)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "n": len(samples),
    }


def run(args):
    b, h = _bh_shape(args.bh)
    q, k, v = _make_inputs(b, h, args.n, args.dtype, args.causal)
    env = _arm_env(args.arm)
    old = {key: os.environ.get(key) for key in ("MFA_V6_NAX_BQ", "MFA_V6_NAX_BK", "MFA_V6_NAX_WM")}
    try:
        for key in old:
            os.environ.pop(key, None)
        os.environ.update(env)

        with dtrace.capture() as trace:
            probe = flash_attention(q, k, v, scale=SCALE, causal=args.causal)
            _eval(probe)
        terminal = [item for item in trace if not item[1].startswith("[reentrant]")]
        if terminal != [("nax_dense", "auto D128 N>=v6_min_n")]:
            raise RuntimeError(f"which-binary failed: expected terminal nax_dense, got {terminal}")

        oracle = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=SCALE, mask=("causal" if args.causal else None),
        )
        _eval(probe, oracle)
        correction = {
            "cos": _cosine(probe, oracle),
            "finite": bool(mx.all(mx.isfinite(probe)).item()),
            "max_abs_vs_fp32": float(mx.max(mx.abs(
                probe.astype(mx.float32) - oracle
            )).item()),
        }
        if correction["cos"] < 0.999 or not correction["finite"]:
            raise RuntimeError(f"correction failed: {correction}")

        timing = _time(lambda: flash_attention(q, k, v, scale=SCALE, causal=args.causal))
    finally:
        for key, value in old.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    return {
        "schema": "mlx-mfa.dense-tile-defaults.v1",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "mlx": importlib.metadata.version("mlx"),
        "platform": platform.platform(),
        "device": str(mx.default_device()),
        "method": {
            "sessions": SESSIONS,
            "warmups": WARMUPS,
            "dispatches_per_sample": DISPATCHES_PER_SAMPLE,
            "process_isolated": True,
            "same_build_required": True,
        },
        "arm": args.arm,
        "requested_env": env,
        "shape": {
            "B": b, "Hq": h, "Hkv": h, "BH": args.bh,
            "N": args.n, "D": D, "dtype": args.dtype, "causal": args.causal,
        },
        "which_binary": {"trace": trace, "terminal": terminal, "expected": "nax_dense"},
        "correction": correction,
        "timing": timing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("default", "candidate"), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--bh", type=int, choices=(8, 32, 64), required=True)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "arm": args.arm,
        "shape": payload["shape"],
        "tile_requested": payload["requested_env"],
        "tile_fingerprint": "see MFA_V6_DUMP_SOURCE stderr",
        "median_ms": payload["timing"]["median_ms"],
        "cos": payload["correction"]["cos"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
