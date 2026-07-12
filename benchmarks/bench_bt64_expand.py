#!/usr/bin/env python3
"""Measure BT64 sparse expansion against scalar fallback and native BT32."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import flash_attention_sparse


REPO = Path(__file__).resolve().parents[1]
SESSIONS = 5
SAMPLES = 5
WARMUP = 2


def _inputs(D: int):
    N, H = 4096, 1
    mx.random.seed(9400 + D)
    q = mx.random.normal((1, H, N, D)).astype(mx.float16)
    k = mx.random.normal((1, H, N, D)).astype(mx.float16)
    v = mx.random.normal((1, H, N, D)).astype(mx.float16)
    rng = np.random.default_rng(9400 + D)
    m64 = rng.random((N // 64, N // 64)) < 0.10
    m64[:, 0] = True
    m64 = mx.array(m64)
    m32 = mx.repeat(mx.repeat(m64, 2, axis=-2), 2, axis=-1)
    mx.eval(q, k, v, m64, m32)
    return q, k, v, m64, m32


def _call(arm, q, k, v, m64, m32, scale):
    prior = os.environ.get("MFA_LCSA_KERNEL_VERSION")
    try:
        if arm == "scalar":
            os.environ["MFA_LCSA_KERNEL_VERSION"] = "v1"
            return flash_attention_sparse(q, k, v, m64, scale=scale)
        os.environ.pop("MFA_LCSA_KERNEL_VERSION", None)
        return flash_attention_sparse(q, k, v, m64 if arm == "bt64" else m32, scale=scale)
    finally:
        if prior is None:
            os.environ.pop("MFA_LCSA_KERNEL_VERSION", None)
        else:
            os.environ["MFA_LCSA_KERNEL_VERSION"] = prior


def _bench(fn):
    samples = []
    for _ in range(SESSIONS):
        for _ in range(WARMUP):
            mx.eval(fn())
        current = []
        for _ in range(SAMPLES):
            mx.synchronize(); t0 = time.perf_counter()
            mx.eval(fn()); mx.synchronize()
            current.append((time.perf_counter() - t0) * 1000)
        samples.extend(current)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def _trace(fn):
    with dtrace.capture() as trace:
        out = fn(); mx.eval(out)
    return out, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--order", choices=("scalar,bt64,bt32", "bt32,bt64,scalar"), required=True)
    p.add_argument("--out", type=Path, default=REPO / "benchmarks/results/bt64_expand.json")
    args = p.parse_args()
    payload = {"order": args.order, "sessions": SESSIONS, "samples_per_arm": SAMPLES, "cells": []}
    for D in (64, 128):
        q, k, v, m64, m32 = _inputs(D); scale = 1 / math.sqrt(D)
        fns = {arm: lambda arm=arm: _call(arm, q, k, v, m64, m32, scale) for arm in ("scalar", "bt64", "bt32")}
        outputs = {}; traces = {}
        for arm in fns:
            outputs[arm], traces[arm] = _trace(fns[arm])
        mx.eval(*outputs.values())
        delta = float(mx.max(mx.abs(outputs["bt64"].astype(mx.float32) - outputs["bt32"].astype(mx.float32))).item())
        if delta != 0.0 or traces["bt64"][-1][0] != "v6nax_sparse" or traces["scalar"][-1][0] != "scalar_fallback":
            raise RuntimeError(f"D{D}: engagement/equivalence failed {traces}, delta={delta}")
        timing = {arm: _bench(fns[arm]) for arm in args.order.split(",")}
        row = {"D": D, "traces": traces, "bt64_vs_bt32_delta": delta, "timing": timing,
               "scalar_over_bt64": timing["scalar"]["median_ms"] / timing["bt64"]["median_ms"]}
        payload["cells"].append(row)
        print(f"D{D}: scalar/bt64={row['scalar_over_bt64']:.2f}x scalar={timing['scalar']['median_ms']:.3f}ms bt64={timing['bt64']['median_ms']:.3f}ms")
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
