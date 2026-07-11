#!/usr/bin/env python3
"""Per-arm benchmark for expert GNA V6 NAX forward.

Default grid targets the VSR-shaped GNA regime:
N in {2048,4096,8192} as T x 32 x 32, D in {64,128},
fp16/bf16, and small/large 3D windows.
"""

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

from mlx_mfa import _ext


DTYPES = {"fp16": mx.float16, "bf16": mx.bfloat16}
WINDOWS = {
    "small": ((1, 7, 7), (1, 1, 1)),
    "large": ((3, 11, 11), (1, 1, 1)),
    "stride2": ((3, 11, 11), (1, 2, 2)),
}


def parse_list(raw: str, typ=int):
    return [typ(x.strip()) for x in raw.split(",") if x.strip()]


def seq_shape_for_n(n: int) -> tuple[int, int, int]:
    if n % 1024 != 0:
        raise ValueError("N must be a multiple of 1024 for T x 32 x 32 benchmark shapes")
    return (n // 1024, 32, 32)


def make_gna_mask(seq_shape, window, stride):
    dim0, dim1, dim2 = seq_shape
    win0, win1, win2 = window
    str0, str1, str2 = stride
    n = dim0 * dim1 * dim2
    dim12 = dim1 * dim2
    idx = np.arange(n, dtype=np.int32)
    q0 = idx // dim12
    q1 = (idx // dim2) % dim1
    q2 = idx % dim2
    g0, g1, g2 = q0 // str0, q1 // str1, q2 // str2
    lo0 = np.maximum(0, g0 * str0 - (win0 - str0) // 2)
    hi0 = np.minimum(dim0 - 1, (g0 + 1) * str0 + (win0 - str0 + 1) // 2 - 1)
    lo1 = np.maximum(0, g1 * str1 - (win1 - str1) // 2)
    hi1 = np.minimum(dim1 - 1, (g1 + 1) * str1 + (win1 - str1 + 1) // 2 - 1)
    lo2 = np.maximum(0, g2 * str2 - (win2 - str2) // 2)
    hi2 = np.minimum(dim2 - 1, (g2 + 1) * str2 + (win2 - str2 + 1) // 2 - 1)
    k0, k1, k2 = q0, q1, q2
    mask = (
        (k0[None, :] >= lo0[:, None]) & (k0[None, :] <= hi0[:, None]) &
        (k1[None, :] >= lo1[:, None]) & (k1[None, :] <= hi1[:, None]) &
        (k2[None, :] >= lo2[:, None]) & (k2[None, :] <= hi2[:, None])
    )
    return mx.array(np.where(mask, 0.0, -1e9).astype(np.float32))


def cosine(a, b) -> float:
    af = np.array(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.array(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    den = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / den) if den else 1.0


def time_arm(fn, warmup: int, sessions: int, iters: int) -> dict:
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
    mx.synchronize()
    samples = []
    for _ in range(sessions):
        t0 = time.perf_counter()
        for _ in range(iters):
            y = fn()
            mx.eval(y)
        mx.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0 / iters)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="2048,4096,8192")
    ap.add_argument("--Ds", default="64,128")
    ap.add_argument("--dtypes", default="fp16,bf16")
    ap.add_argument("--windows", default="small,large")
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="benchmarks/results/gna_nax_latest.json")
    args = ap.parse_args()

    if not bool(_ext.device_has_neural_accelerators()):
        raise SystemExit("GNA NAX benchmark requires V6 NAX hardware")

    results = []
    tile_env = {
        "MFA_GNA_NAX_BQ": os.environ.get("MFA_GNA_NAX_BQ"),
        "MFA_GNA_NAX_BK": os.environ.get("MFA_GNA_NAX_BK"),
        "MFA_GNA_NAX_WM": os.environ.get("MFA_GNA_NAX_WM"),
    }
    for dtype_name in parse_list(args.dtypes, str):
        dtype = DTYPES[dtype_name]
        for D in parse_list(args.Ds):
            for N in parse_list(args.Ns):
                seq_shape = seq_shape_for_n(N)
                for win_name in parse_list(args.windows, str):
                    window, stride = WINDOWS[win_name]
                    scale = 1.0 / math.sqrt(D)
                    key = mx.random.key(6000 + N + D + len(win_name))
                    q = mx.random.normal((1, 1, N, D), key=key).astype(dtype)
                    k = mx.random.normal((1, 1, N, D), key=mx.random.split(key)[0]).astype(dtype)
                    v = mx.random.normal((1, 1, N, D), key=mx.random.split(key)[1]).astype(dtype)
                    mask = make_gna_mask(seq_shape, window, stride).astype(dtype)
                    mx.eval(q, k, v, mask)

                    def gna_nax():
                        return _ext.mfa_gna_nax_forward(q, k, v, *seq_shape, *window, *stride, scale)

                    def sdpa_masked():
                        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

                    arms = {
                        "gna_nax": gna_nax,
                        "sdpa_masked": sdpa_masked,
                    }
                    if D == 128:
                        def gna_steel():
                            return _ext.mfa_gna_forward(q, k, v, scale, *seq_shape, *window, *stride)
                        arms["gna_steel"] = gna_steel

                    outputs = {name: fn() for name, fn in arms.items()}
                    mx.eval(*outputs.values())
                    cos_vs_sdpa = {
                        name: cosine(out, outputs["sdpa_masked"])
                        for name, out in outputs.items()
                    }
                    if min(cos_vs_sdpa.values()) < 0.999:
                        raise RuntimeError(
                            f"correctness failed for N={N} D={D} {dtype_name} {win_name}: {cos_vs_sdpa}"
                        )

                    for name, fn in arms.items():
                        mx.clear_cache()
                        stat = time_arm(fn, args.warmup, args.sessions, args.iters)
                        row = {
                            "N": N,
                            "D": D,
                            "dtype": dtype_name,
                            "window": win_name,
                            "seq_shape": seq_shape,
                            "window_size": window,
                            "stride": stride,
                            "arm": name,
                            "cos_vs_sdpa": cos_vs_sdpa[name],
                            "which_binary": "direct _ext.mfa_gna_nax_forward" if name == "gna_nax" else name,
                            "tile_env": tile_env,
                            **stat,
                        }
                        results.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mlx_version": mx.__version__,
        "sessions": args.sessions,
        "iters": args.iters,
        "warmup": args.warmup,
        "tile_env": tile_env,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = out_path.with_suffix(".md")
    lines = [
        "| dtype | D | N | window | arm | median ms | cos vs SDPA |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['dtype']} | {r['D']} | {r['N']} | {r['window']} | {r['arm']} | "
            f"{r['median_ms']:.4f} | {r['cos_vs_sdpa']:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
