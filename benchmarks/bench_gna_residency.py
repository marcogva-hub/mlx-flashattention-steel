#!/usr/bin/env python3
"""Measure the MLX-style GNA grid swizzle as an isolated residency probe.

The candidate changes only the threadgroup walk order.  Both arms use the
public ``flash_attention_gna`` route, and the dispatch trace is captured before
timing so an SDPA/STEEL fallback cannot masquerade as a GNA result.
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

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa.attention import flash_attention_gna
from benchmarks.bench_gna_nax import WINDOWS, make_gna_mask


DTYPES = {"fp16": mx.float16, "bf16": mx.bfloat16}


def cosine(a, b) -> float:
    af = np.asarray(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.asarray(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    den = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / den) if den else 1.0


def configure_arm(D: int, arm: str, mode: str) -> None:
    swizzle_log = 0
    if mode.startswith("swizzle") and arm == "candidate":
        swizzle_log = int(mode.removeprefix("swizzle"))
    if swizzle_log == 0:
        os.environ.pop("MFA_GNA_NAX_SWIZZLE_LOG", None)
    else:
        os.environ["MFA_GNA_NAX_SWIZZLE_LOG"] = str(swizzle_log)
    for name in (
        "MFA_GNA_NAX_BQ",
        "MFA_GNA_NAX_BK",
        "MFA_GNA_NAX_WM",
        "MFA_GNA_NAX_PRECOMPUTE_RANGE",
    ):
        os.environ.pop(name, None)
    if mode == "bq-resident" and arm == "candidate":
        os.environ["MFA_GNA_NAX_BQ"] = "64" if D == 64 else "128"
        os.environ["MFA_GNA_NAX_BK"] = "32"
        os.environ["MFA_GNA_NAX_WM"] = "4"


def timed(fn, warmup: int, sessions: int, dispatches: int) -> dict:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(sessions):
        start = time.perf_counter()
        for _ in range(dispatches):
            mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0 / dispatches)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("baseline-first", "candidate-first"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--dispatches", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--candidate-mode", default="swizzle1",
        choices=("swizzle1", "swizzle2", "bq-resident"),
    )
    args = parser.parse_args()

    rows = []
    arm_order = (("baseline", "baseline"), ("candidate", args.candidate_mode))
    if args.order == "candidate-first":
        arm_order = tuple(reversed(arm_order))

    for dtype_name, dtype in DTYPES.items():
        for D in (64, 128):
            N = 4096
            seq_shape = (4, 32, 32)
            scale = 1.0 / math.sqrt(D)
            key = mx.random.key(9400 + D + (0 if dtype_name == "fp16" else 100))
            q = mx.random.normal((1, 1, N, D), key=key).astype(dtype)
            k = mx.random.normal((1, 1, N, D), key=mx.random.split(key)[0]).astype(dtype)
            v = mx.random.normal((1, 1, N, D), key=mx.random.split(key)[1]).astype(dtype)
            mx.eval(q, k, v)
            for window_name in ("small", "large"):
                window, stride = WINDOWS[window_name]
                mask = make_gna_mask(seq_shape, window, stride).astype(dtype)
                oracle = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask
                )
                mx.eval(oracle)
                cell = {
                    "dtype": dtype_name,
                    "D": D,
                    "N": N,
                    "seq_shape": seq_shape,
                    "window": window_name,
                    "window_size": window,
                    "stride": stride,
                    "order": args.order,
                    "sessions": args.sessions,
                    "dispatches_per_sample": args.dispatches,
                    "which_binary_expected": "public flash_attention_gna -> gna_v6nax",
                }

                def public_gna():
                    return flash_attention_gna(
                        q, k, v, seq_shape, window, stride, scale=scale
                    )

                for arm_name, mode in arm_order:
                    configure_arm(D, arm_name, mode)
                    with dtrace.capture() as trace:
                        probe = public_gna()
                        mx.eval(probe)
                    labels = [event[0] for event in trace]
                    if labels[-1:] != ["gna_v6nax"]:
                        raise RuntimeError(
                            f"which-binary failure for {arm_name}: trace={trace}"
                        )
                    cos = cosine(probe, oracle)
                    if cos < 0.999:
                        raise RuntimeError(
                            f"oracle failure for {arm_name}: cos={cos} trace={trace}"
                        )
                    mx.clear_cache()
                    stat = timed(public_gna, args.warmup, args.sessions, args.dispatches)
                    rows.append({
                        **cell,
                        "arm": arm_name,
                        "candidate_mode": args.candidate_mode,
                        "swizzle_log": int(mode.removeprefix("swizzle"))
                        if mode.startswith("swizzle") else 0,
                        "trace": trace,
                        "cos_vs_masked_sdpa": cos,
                        "max_abs_vs_masked_sdpa": float(
                            np.max(
                                np.abs(
                                    np.asarray(probe.astype(mx.float32))
                                    - np.asarray(oracle.astype(mx.float32))
                                )
                            )
                        ),
                        **stat,
                    })
                    print(json.dumps(rows[-1], sort_keys=True), flush=True)

    payload = {
        "schema": "gna-residency-bricks-v1",
        "mlx_version": mx.__version__,
        "order": args.order,
        "candidate_mode": args.candidate_mode,
        "rows": rows,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
