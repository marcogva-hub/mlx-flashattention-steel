"""Benchmark V6 NAX quantized matmul expert path vs MLX quantized_matmul.

Runs per-arm sustained sessions: all sessions for mlx-mfa, then all sessions
for MLX, never interleaved inside a shape.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _ext


def cos(a, b) -> float:
    af = np.array(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.array(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    den = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / den) if den else 1.0


def make_case(M: int, N: int, K: int, dtype, bits: int, group_size: int):
    seed = 17 + M * 3 + N * 5 + K * 7 + bits * 11 + group_size * 13
    key = mx.random.key(seed)
    kx, kw = mx.random.split(key)
    x = (mx.random.normal((M, K), key=kx) * 0.25).astype(dtype)
    w = (mx.random.normal((N, K), key=kw) * 0.25).astype(dtype)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(x, w_q, scales, biases)
    mx.synchronize()
    return x, w_q, scales, biases


def correctness(x, w_q, scales, biases, group_size: int, bits: int):
    y_mfa = _ext.v6_nax_quantized_matmul(x, w_q, scales, biases, group_size, bits)
    y_mlx = mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases,
        group_size=group_size, bits=bits, transpose=True)
    w_deq = mx.dequantize(w_q, scales, biases, group_size=group_size, bits=bits).astype(mx.float32)
    ref = mx.matmul(x.astype(mx.float32), mx.transpose(w_deq))
    mx.eval(y_mfa, y_mlx, ref)
    mx.synchronize()
    return {
        "cos_mfa_fp32": cos(y_mfa, ref),
        "cos_mlx_fp32": cos(y_mlx, ref),
        "cos_mfa_mlx": cos(y_mfa, y_mlx),
        "finite_mfa": bool(np.isfinite(np.array(y_mfa.astype(mx.float32))).all()),
    }


def run_arm(fn, *, warmup: int, iters: int, sessions: int):
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
    mx.synchronize()

    session_ms = []
    for _ in range(sessions):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            y = fn()
            mx.eval(y)
        mx.synchronize()
        session_ms.append((time.perf_counter() - t0) * 1000.0 / iters)
    return {
        "sessions_ms": session_ms,
        "median_ms": statistics.median(session_ms),
        "min_ms": min(session_ms),
        "max_ms": max(session_ms),
    }


def shape_grid(mode: str):
    if mode == "smoke":
        return [(256, 1024, 1024)]
    if mode == "autoresearch":
        return [
            (128, 512, 512),
            (256, 1024, 1024),
            (1024, 1024, 1024),
            (2048, 1024, 1024),
            (4096, 1024, 1024),
            (1024, 2048, 1024),
            (1024, 1024, 2048),
        ]
    return [
        (256, 1024, 1024),
        (1024, 1024, 1024),
        (4096, 1024, 1024),
        (1024, 2048, 2048),
    ]


def dtype_grid(mode: str):
    return [mx.float16] if mode == "smoke" else [mx.float16, mx.bfloat16]


def dtype_name(dtype) -> str:
    return "bf16" if dtype == mx.bfloat16 else "fp16"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "autoresearch", "final"], default="smoke")
    parser.add_argument("--bits", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--groups", type=int, nargs="*", default=[32, 64, 128])
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlx": mx.__version__,
        "has_nax": bool(_ext.device_has_neural_accelerators()),
        "device_info": dict(_ext.get_device_info()),
        "env_tile": {k: os.environ.get(k) for k in [
            "MFA_QMM_NAX_BM", "MFA_QMM_NAX_BN", "MFA_QMM_NAX_BK",
            "MFA_QMM_NAX_WM", "MFA_QMM_NAX_WN"]},
    }
    rows = []
    for dtype in dtype_grid(args.mode):
        for bits in args.bits:
            for group_size in args.groups:
                for M, N, K in shape_grid(args.mode):
                    if K % group_size != 0:
                        continue
                    x, w_q, scales, biases = make_case(M, N, K, dtype, bits, group_size)
                    corr = correctness(x, w_q, scales, biases, group_size, bits)
                    if corr["cos_mfa_fp32"] < 0.999 or not corr["finite_mfa"]:
                        raise RuntimeError(f"correction failed for {(M, N, K, dtype_name(dtype), bits, group_size)}: {corr}")

                    mfa = run_arm(
                        lambda: _ext.v6_nax_quantized_matmul(x, w_q, scales, biases, group_size, bits),
                        warmup=args.warmup, iters=args.iters, sessions=args.sessions)
                    mlx_arm = run_arm(
                        lambda: mx.quantized_matmul(
                            x, w_q, scales=scales, biases=biases,
                            group_size=group_size, bits=bits, transpose=True),
                        warmup=args.warmup, iters=args.iters, sessions=args.sessions)
                    row = {
                        "M": M, "N": N, "K": K,
                        "dtype": dtype_name(dtype),
                        "bits": bits,
                        "group_size": group_size,
                        "correctness": corr,
                        "mfa": mfa,
                        "mlx": mlx_arm,
                        "mfa_over_mlx": mfa["median_ms"] / mlx_arm["median_ms"],
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)

    result = {"stamp": stamp, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
