#!/usr/bin/env python3
"""Public routed-path spot checks for the final consolidation table."""

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
from mlx_mfa import flash_attention, flash_attention_gna, flash_attention_sparse
from mlx_mfa.attention import make_sliding_window_mask
from benchmarks.bench_gna_nax import make_gna_mask


SESSIONS = 5
WARMUPS = 2
DISPATCHES_PER_SAMPLE = 20


def evaluate(value):
    mx.eval(*value) if isinstance(value, (tuple, list)) else mx.eval(value)
    mx.synchronize()


def stats(samples):
    return {"median_ms": statistics.median(samples), "p95_ms": float(np.percentile(samples, 95)),
            "samples_ms": samples, "n": len(samples)}


def cosine(a, b):
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    x = mx.sum(af * bf) / mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    mx.eval(x)
    return float(x.item())


def time_arm(fn):
    for _ in range(WARMUPS):
        for _ in range(DISPATCHES_PER_SAMPLE):
            evaluate(fn())
    values = []
    for _ in range(SESSIONS):
        started = time.perf_counter()
        for _ in range(DISPATCHES_PER_SAMPLE):
            evaluate(fn())
        values.append((time.perf_counter() - started) * 1000.0 / DISPATCHES_PER_SAMPLE)
    return stats(values)


def run(kind, arm):
    d, n, dtype = 128, 4096, mx.float16
    scale = 1.0 / math.sqrt(d)
    mx.random.seed(20260713 + len(kind))
    if kind == "dense":
        q = mx.random.normal((1, 4, n, d)).astype(dtype)
        k = mx.random.normal((1, 4, n, d)).astype(dtype)
        v = mx.random.normal((1, 4, n, d)).astype(dtype)
        def public():
            return flash_attention(q, k, v, scale=scale, causal=False)
        expected = "nax_dense"
        reference = lambda: mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=scale
        )
        label = "dense_d128_n4096"
    elif kind == "sparse":
        q = mx.random.normal((1, 1, n, d)).astype(dtype)
        k = mx.random.normal((1, 1, n, d)).astype(dtype)
        v = mx.random.normal((1, 1, n, d)).astype(dtype)
        block_mask = make_sliding_window_mask(n, 128, head_dim=d)
        block_mask = block_mask.astype(mx.bool_)
        expanded = mx.repeat(mx.repeat(block_mask.astype(mx.float32), 32, -2), 32, -1)
        bias = mx.where(expanded, mx.array(0.0, mx.float32), mx.array(-1e30, mx.float32))
        def public():
            return flash_attention_sparse(q, k, v, block_mask, scale=scale, causal=False)
        expected = "v6nax_sparse"
        reference = lambda: mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=scale, mask=bias
        )
        label = "sparse_bt32_d128_n4096"
    elif kind == "gna":
        seq_shape = (4, 32, 32)
        q = mx.random.normal((1, 1, n, d)).astype(dtype)
        k = mx.random.normal((1, 1, n, d)).astype(dtype)
        v = mx.random.normal((1, 1, n, d)).astype(dtype)
        window, stride = (1, 7, 7), (1, 1, 1)
        mask = make_gna_mask(seq_shape, window, stride).astype(mx.float32)
        def public():
            return flash_attention_gna(q, k, v, seq_shape, window, stride, scale=scale)
        expected = "gna_v6nax"
        reference = lambda: mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=scale, mask=mask
        )
        label = "gna_d128_3d_n4096"
    else:
        raise ValueError(kind)

    def baseline():
        return reference()

    if arm == "public":
        with dtrace.capture() as trace:
            probe = public()
            evaluate(probe)
        terminal_trace = [item for item in trace if not item[1].startswith("[reentrant]")]
    else:
        probe = baseline()
        evaluate(probe)
        trace = []
        terminal_trace = []
    ref = reference()
    evaluate((probe, ref))
    correction = {"cos": cosine(probe, ref),
                  "finite": bool(mx.all(mx.isfinite(probe)).item())}
    delta = float(mx.max(mx.abs(probe.astype(mx.float32) - ref.astype(mx.float32))).item())
    if arm == "public" and (not terminal_trace or terminal_trace[-1][0] != expected):
        raise RuntimeError(f"which-binary failed for {label}: {trace}")
    if correction["cos"] < 0.999 or not correction["finite"] or (arm == "public" and delta == 0.0):
        raise RuntimeError(f"correction/engagement failed for {label}: {correction}, delta={delta}")

    timing = time_arm(public if arm == "public" else baseline)
    print(f"{label} arm={arm}: median={timing['median_ms']:.3f}ms trace={terminal_trace}", flush=True)
    return {"label": label, "kind": kind, "shape": {"N": n, "D": d},
            "arm": arm, "which_binary": {"public_trace": trace, "terminal_trace": terminal_trace, "expected": expected,
                             "public_vs_reference_max_abs": delta},
            "correction": correction, "timing": timing}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("dense", "sparse", "gna"), required=True)
    parser.add_argument("--arm", choices=("public", "sdpa"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = run(args.kind, args.arm)
    payload = {"schema": "mlx-mfa.final-routed-spots.v1",
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "mlx": importlib.metadata.version("mlx"), "platform": platform.platform(),
               "arm": args.arm,
               "method": {"sessions": SESSIONS, "warmups": WARMUPS,
                          "samples_per_session": 1,
                          "dispatches_per_sample": DISPATCHES_PER_SAMPLE},
               "row": row}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
