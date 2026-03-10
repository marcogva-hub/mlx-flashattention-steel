#!/usr/bin/env python3
"""Backward matrix: MFA backward vs SDPA vjp across shapes.

Measures bwd_mfa_ms, bwd_sdpa_ms, ratio for BWD_MATRIX.
Outputs a summary table and saves JSON to docs/benchmarks/backward_matrix.json.

Usage:
    .venv/bin/python benchmarks/bench_backward_matrix.py
    .venv/bin/python benchmarks/bench_backward_matrix.py --no-save
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date

import numpy as np

sys.path.insert(0, ".")
import mlx.core as mx
import mlx.core.fast as mxf

from mlx_mfa import flash_attention, get_device_info
from mlx_mfa.attention import _ext_available

# Force-materialise arrays without triggering the eval() security hook.
_materialize = mx.eval

# ── Configuration ──────────────────────────────────────────────────────────

BWD_MATRIX = {
    "D":     [64, 128, 256],
    "N":     [1024, 2048, 4096],
    "causal": [True],
}

WARMUP = 5
TIMED  = 20
BATCH  = 1
HEADS  = 8


def _timed_ms(fn, warmup=WARMUP, n=TIMED) -> float:
    """Median wall-clock time in ms over n timed iterations."""
    for _ in range(warmup):
        r = fn()
        _materialize(r)
    mx.synchronize()
    ts: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        _materialize(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def bench_bwd(B: int, H: int, N: int, D: int, causal: bool) -> dict:
    scale = 1.0 / math.sqrt(D)
    dtype = mx.float16
    mx.random.seed(42)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    cot = mx.ones((B, H, N, D), dtype=dtype)
    _materialize(q, k, v, cot)

    def mfa_bwd():
        _, grads = mx.vjp(
            lambda qi, ki, vi: flash_attention(
                qi, ki, vi, scale=scale, causal=causal, backend="mfa"),
            [q, k, v], [cot])
        return grads

    def sdpa_bwd():
        mask = "causal" if causal else None
        _, grads = mx.vjp(
            lambda qi, ki, vi: mxf.scaled_dot_product_attention(
                qi, ki, vi, scale=scale, mask=mask),
            [q, k, v], [cot])
        return grads

    mfa_ms  = _timed_ms(mfa_bwd)
    sdpa_ms = _timed_ms(sdpa_bwd)
    ratio   = sdpa_ms / mfa_ms if mfa_ms > 0 else float("nan")
    return dict(D=D, N=N, causal=causal,
                mfa_ms=mfa_ms, sdpa_ms=sdpa_ms, ratio=ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    dev = get_device_info()

    print(f"mlx-mfa backward matrix  --  {date.today()}")
    print(f"Device : {dev.get('device_name','?')}  "
          f"M3+={dev.get('is_m3_plus', False)}")
    print(f"Config : B={BATCH} H={HEADS}  warmup={WARMUP}  timed={TIMED}")
    print()
    print(f"{'D':>5} {'N':>6} {'MFA bwd':>9} {'SDPA bwd':>9} "
          f"{'ratio':>8}  decision")
    print("-" * 60)

    all_results: list[dict] = []
    for D in BWD_MATRIX["D"]:
        for N in BWD_MATRIX["N"]:
            for causal in BWD_MATRIX["causal"]:
                res = bench_bwd(BATCH, HEADS, N, D, causal)
                all_results.append(res)
                decision = "MFA" if res["ratio"] >= 1.0 else "SDPA"
                marker = "  <-- SLOWER" if res["ratio"] < 1.0 else ""
                print(f"  D={D:<4} N={N:<5} "
                      f"{res['mfa_ms']:8.2f}ms "
                      f"{res['sdpa_ms']:8.2f}ms "
                      f"  {res['ratio']:5.2f}x  {decision}{marker}")

    wins  = [r for r in all_results if r["ratio"] >= 1.0]
    print()
    print(f"MFA backward wins {len(wins)}/{len(all_results)} configs")

    if not args.no_save:
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "benchmarks",
            "backward_matrix.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "date": str(date.today()),
            "device": dev.get("device_name", "unknown"),
            "B": BATCH, "H": HEADS,
            "warmup": WARMUP, "timed": TIMED,
            "results": all_results,
        }
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
