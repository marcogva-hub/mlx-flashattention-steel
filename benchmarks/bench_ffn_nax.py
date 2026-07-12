#!/usr/bin/env python3
"""Exact-shape V6 NAX FFN Linear/GELU benchmark and tile sweep."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx_mfa
from mlx_mfa import _ext


TILE_KEYS = ("MFA_FFN_NAX_BM", "MFA_FFN_NAX_BN", "MFA_FFN_NAX_BK", "MFA_FFN_NAX_WM", "MFA_FFN_NAX_WN")
TILES = [
    (64, 128, 256, 2, 4),  # MLX g17s default
    (64, 64, 256, 2, 2),
    (64, 256, 256, 2, 4),
    (128, 128, 256, 4, 4),
    (128, 256, 256, 4, 4),
    (64, 128, 128, 2, 4),
    (64, 256, 128, 2, 4),
    (128, 128, 128, 4, 4),
]


def cosine(a, b) -> float:
    af = np.asarray(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.asarray(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    den = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / den) if den else 1.0


def time_arm(fn, sessions: int, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(sessions):
        start = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0 / iters)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def set_tile(tile):
    for key, value in zip(TILE_KEYS, tile):
        os.environ[key] = str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not _ext.device_has_neural_accelerators():
        raise SystemExit("V6 NAX hardware required")

    rows = []
    shapes = [
        ("up_gelu", 2048, 1536, 8960, True),
        ("down", 2048, 8960, 1536, False),
    ]
    for shape_name, M, K, N, gelu in shapes:
        mx.random.seed(10001 + K + N)
        x = (mx.random.normal((M, K)) * 0.02).astype(mx.float16)
        w = (mx.random.normal((N, K)) * 0.01).astype(mx.float16)
        b = (mx.random.normal((N,)) * 0.005).astype(mx.float16)
        mx.eval(x, w, b)

        def mlx_arm():
            y = x @ w.T + b
            return nn.gelu_approx(y) if gelu else y

        ref = x.astype(mx.float32) @ w.astype(mx.float32).T + b.astype(mx.float32)
        if gelu:
            ref = nn.gelu_approx(ref)
        mlx_out = mlx_arm()
        mx.eval(ref, mlx_out)
        mlx_cos = cosine(mlx_out, ref)
        if mlx_cos < 0.999:
            raise RuntimeError(f"MLX reference cosine failed for {shape_name}: {mlx_cos}")
        mlx_timing = time_arm(mlx_arm, args.sessions, args.iters, args.warmup)

        for tile in TILES:
            if K % tile[2] != 0:
                continue
            set_tile(tile)

            def mfa_arm():
                return _ext.v6_nax_linear(x, w, b, gelu)

            out = mfa_arm()
            mx.eval(out)
            cos = cosine(out, ref)
            delta_mlx = float(mx.max(mx.abs(out.astype(mx.float32) - mlx_out.astype(mx.float32))).item())
            if cos < 0.999 or not bool(mx.all(mx.isfinite(out)).item()):
                raise RuntimeError(f"MFA correction failed for {shape_name}/{tile}: cos={cos}")
            timing = time_arm(mfa_arm, args.sessions, args.iters, args.warmup)
            rows.append({
                "shape": shape_name,
                "M": M,
                "N": N,
                "K": K,
                "gelu": gelu,
                "tile": dict(zip(("BM", "BN", "BK", "WM", "WN"), tile)),
                "correctness": {
                    "cos_mfa_fp32": cos,
                    "cos_mlx_fp32": mlx_cos,
                    "max_abs_mfa_mlx": delta_mlx,
                    "which_binary": "direct _ext.v6_nax_linear",
                },
                "mfa": timing,
                "mlx": mlx_timing,
                "mlx_over_mfa": mlx_timing["median_ms"] / timing["median_ms"],
            })
            print(json.dumps(rows[-1]), flush=True)

    for key in TILE_KEYS:
        os.environ.pop(key, None)
    payload = {
        "stamp": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "python": platform.python_version(),
            "mlx": mx.__version__,
            "mlx_mfa": mlx_mfa.__version__,
            "device": mlx_mfa.get_device_info(),
            "sessions": args.sessions,
            "iters": args.iters,
        },
        "mlx_fingerprint": {
            "kernel": "steel_gemm_fused_nax_nn_float16_float16",
            "tile_g17s": {"BM": 64, "BN": 128, "BK": 256, "WM": 2, "WN": 4},
            "swizzle_log": 2,
            "activation_epilogue": False,
            "source": "MLX 0.31.2 metal/matmul.cpp + steel_gemm_fused_nax.h",
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
