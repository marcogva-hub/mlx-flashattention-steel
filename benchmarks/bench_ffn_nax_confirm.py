#!/usr/bin/env python3
"""Order-controlled confirmation for the two exact FlashVSR FFN tiles."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_mfa
from mlx_mfa import _ext


CASES = [
    ("up_gelu", 2048, 1536, 8960, True, (64, 64, 256, 2, 2)),
    ("down", 2048, 8960, 1536, False, (64, 256, 256, 2, 4)),
]
KEYS = ("MFA_FFN_NAX_BM", "MFA_FFN_NAX_BN", "MFA_FFN_NAX_BK", "MFA_FFN_NAX_WM", "MFA_FFN_NAX_WN")


def cosine(a, b):
    af = np.asarray(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.asarray(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))


def timing(fn, sessions, iters):
    for _ in range(10):
        mx.eval(fn())
    values = []
    for _ in range(sessions):
        start = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        values.append((time.perf_counter() - start) * 1000 / iters)
    return {"median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values), "samples_ms": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("mlx-first", "mfa-first"), required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, M, K, N, gelu, tile in CASES:
        mx.random.seed(11001 + K + N)
        x = (mx.random.normal((M, K)) * 0.02).astype(mx.float16)
        w = (mx.random.normal((N, K)) * 0.01).astype(mx.float16)
        b = (mx.random.normal((N,)) * 0.005).astype(mx.float16)
        mx.eval(x, w, b)
        for key, value in zip(KEYS, tile):
            os.environ[key] = str(value)

        def mlx_arm():
            y = x @ w.T + b
            return nn.gelu_approx(y) if gelu else y

        def mfa_arm():
            return _ext.v6_nax_linear(x, w, b, gelu)

        mfa_out, mlx_out = mfa_arm(), mlx_arm()
        ref = x.astype(mx.float32) @ w.astype(mx.float32).T + b.astype(mx.float32)
        if gelu:
            ref = nn.gelu_approx(ref)
        mx.eval(mfa_out, mlx_out, ref)
        cos = cosine(mfa_out, ref)
        if cos < 0.999:
            raise RuntimeError(f"{name} correction failed: {cos}")
        if args.order == "mlx-first":
            mlx_time = timing(mlx_arm, args.sessions, args.iters)
            mfa_time = timing(mfa_arm, args.sessions, args.iters)
        else:
            mfa_time = timing(mfa_arm, args.sessions, args.iters)
            mlx_time = timing(mlx_arm, args.sessions, args.iters)
        rows.append({
            "shape": name,
            "tile": dict(zip(("BM", "BN", "BK", "WM", "WN"), tile)),
            "cos_mfa_fp32": cos,
            "max_abs_mfa_mlx": float(mx.max(mx.abs(mfa_out.astype(mx.float32) - mlx_out.astype(mx.float32))).item()),
            "which_binary": "direct _ext.v6_nax_linear",
            "mfa": mfa_time,
            "mlx": mlx_time,
            "mlx_over_mfa": mlx_time["median_ms"] / mfa_time["median_ms"],
        })
        print(json.dumps(rows[-1]), flush=True)
    for key in KEYS:
        os.environ.pop(key, None)
    payload = {"mlx": mx.__version__, "mlx_mfa": mlx_mfa.__version__, "device": mlx_mfa.get_device_info(), "order": args.order, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
