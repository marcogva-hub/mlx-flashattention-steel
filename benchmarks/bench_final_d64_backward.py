#!/usr/bin/env python3
"""Fresh-process D64 backward engagement and timing lock."""

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

import mlx_mfa
from mlx_mfa import _dispatch_trace as dtrace


N, D, H = 4096, 64, 4
SESSIONS, WARMUPS = 5, 2


def evaluate(value):
    mx.eval(*value) if isinstance(value, (tuple, list)) else mx.eval(value)
    mx.synchronize()


def manual_fp32_grads(q, k, v, causal):
    scale = 1.0 / math.sqrt(D)

    def forward(q_, k_, v_):
        qf, kf, vf = q_.astype(mx.float32), k_.astype(mx.float32), v_.astype(mx.float32)
        scores = (qf @ kf.swapaxes(-1, -2)) * scale
        if causal:
            scores = scores + mx.triu(mx.full((N, N), -1e30, dtype=mx.float32), k=1)
        return (mx.softmax(scores, axis=-1) @ vf).sum()

    return mx.grad(forward, argnums=(0, 1, 2))(q, k, v)


def cosine(a, b):
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    x = mx.sum(af * bf) / mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    mx.eval(x)
    return float(x.item())


def stats(values):
    return {"median_ms": statistics.median(values), "p95_ms": float(np.percentile(values, 95)),
            "samples_ms": values, "n": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("v6", "sdpa"), required=True)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--order-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.pop("MFA_ENABLE_V6_BACKWARD", None)
    if args.arm == "sdpa":
        os.environ["MFA_DISABLE_V6_BACKWARD"] = "1"
    else:
        os.environ.pop("MFA_DISABLE_V6_BACKWARD", None)

    mx.random.seed(20260713 + int(args.causal))
    q = (mx.random.normal((1, H, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.normal((1, H, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.normal((1, H, N, D)) * 0.1).astype(mx.float16)
    evaluate((q, k, v))
    scale = 1.0 / math.sqrt(D)

    def grad_call():
        fn = lambda q_, k_, v_: mlx_mfa.flash_attention(
            q_, k_, v_, scale=scale, causal=args.causal
        ).sum()
        return mx.grad(fn, argnums=(0, 1, 2))(q, k, v)

    with dtrace.capture() as trace:
        probe = grad_call()
        evaluate(probe)
    terminal = [item for item in trace if not item[1].startswith("[reentrant]")]
    oracle = manual_fp32_grads(q, k, v, args.causal)
    evaluate((probe, oracle))
    metrics = {
        "cos_dq": cosine(probe[0], oracle[0]),
        "cos_dk": cosine(probe[1], oracle[1]),
        "cos_dv": cosine(probe[2], oracle[2]),
        "finite": all(bool(mx.all(mx.isfinite(item)).item()) for item in probe),
    }
    if min(metrics["cos_dq"], metrics["cos_dk"], metrics["cos_dv"]) < 0.999 or not metrics["finite"]:
        raise RuntimeError(f"gradient correction failed: {metrics}")
    if args.arm == "v6" and (not terminal or terminal[-1][0] != "v6_split_backward"):
        raise RuntimeError(f"V6 backward engagement failed: {trace}")
    if args.arm == "sdpa" and any(item[0] == "v6_split_backward" for item in terminal):
        raise RuntimeError(f"SDPA arm engaged V6 backward: {trace}")

    for _ in range(WARMUPS):
        evaluate(grad_call())
    samples = []
    for _ in range(SESSIONS):
        started = time.perf_counter()
        evaluate(grad_call())
        samples.append((time.perf_counter() - started) * 1000.0)
    timing = stats(samples)
    print(f"D64 N{N} causal={int(args.causal)} arm={args.arm}: "
          f"median={timing['median_ms']:.3f}ms terminal={terminal}", flush=True)
    payload = {
        "schema": "mlx-mfa.final-d64-backward.v1", "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "mlx": importlib.metadata.version("mlx"), "platform": platform.platform(),
        "shape": {"B": 1, "H": H, "N": N, "D": D, "causal": args.causal},
        "arm": args.arm, "order_label": args.order_label,
        "which_binary": {"trace": trace, "terminal": terminal,
                          "v6": "v6_split_backward", "sdpa": "mx.grad(SDPA fallback)"},
        "correction": metrics, "timing": timing,
        "method": {"sessions": SESSIONS, "warmups": WARMUPS, "samples_per_session": 1},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
