#!/usr/bin/env python3
"""Dispatch matrix: MFA vs SDPA across all shapes.

Measures mfa_ms, sdpa_ms, ratio (sdpa/mfa) for every combination in MATRIX.
Outputs a summary table and saves JSON to docs/benchmarks/dispatch_matrix.json.

Usage:
    .venv/bin/python benchmarks/bench_dispatch_matrix.py
    .venv/bin/python benchmarks/bench_dispatch_matrix.py --no-save
    .venv/bin/python benchmarks/bench_dispatch_matrix.py --json-only
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

from mlx_mfa import flash_attention, get_device_info
from mlx_mfa.attention import _fallback_sdpa, _ext_available

# ── Configuration ──────────────────────────────────────────────────────────

MATRIX = {
    "D":     [64, 128, 256, 512],
    "N":     [512, 1024, 2048, 4096, 8192],
    "dtype": [mx.float16],
    "causal": [True, False],
}

WARMUP = 5
TIMED  = 20
BATCH  = 1
HEADS  = 8


def _timed_ms(fn, warmup=WARMUP, n=TIMED) -> float:
    """Median wall-clock time in ms over n timed iterations."""
    for _ in range(warmup):
        r = fn()
        mx.eval(r)
    mx.synchronize()
    ts: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def bench_one(B: int, H: int, N: int, D: int, causal: bool,
              dtype) -> dict:
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(42)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    mx.eval(q, k, v)

    mfa_ms  = _timed_ms(lambda: flash_attention(q, k, v, scale=scale,
                                                 causal=causal,
                                                 backend="mfa"))
    sdpa_ms = _timed_ms(lambda: _fallback_sdpa(q, k, v, scale, causal))
    ratio   = sdpa_ms / mfa_ms if mfa_ms > 0 else float("nan")
    return dict(D=D, N=N, causal=causal, mfa_ms=mfa_ms,
                sdpa_ms=sdpa_ms, ratio=ratio)


def _ratio_str(r: float) -> str:
    if math.isnan(r):
        return "  n/a  "
    marker = " ss" if r >= 1.0 else " xx"
    return f"{r:5.2f}x{marker}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true",
                    help="Skip writing JSON")
    ap.add_argument("--json-only", action="store_true",
                    help="Skip human-readable output")
    args = ap.parse_args()

    dev = get_device_info()
    ext = _ext_available()

    if not args.json_only:
        print(f"mlx-mfa dispatch matrix  --  {date.today()}")
        print(f"Device : {dev.get('device_name','?')}  "
              f"M3+={dev.get('is_m3_plus', False)}  "
              f"ext={ext}")
        print(f"Config : B={BATCH} H={HEADS}  warmup={WARMUP}  timed={TIMED}")

    all_results: list[dict] = []

    for causal in MATRIX["causal"]:
        causal_str = "causal" if causal else "non-causal"
        if not args.json_only:
            print()
            print(f"=== {causal_str} " + "=" * 60)
            print(f"{'D':>5} {'N':>6} {'MFA ms':>8} {'SDPA ms':>9} "
                  f"{'MFA/SDPA':>10}  decision")
            print("-" * 60)

        for D in MATRIX["D"]:
            for N in MATRIX["N"]:
                res = bench_one(BATCH, HEADS, N, D, causal, mx.float16)
                res["causal"] = causal
                all_results.append(res)

                if not args.json_only:
                    decision = "MFA" if res["ratio"] >= 1.0 else "SDPA"
                    print(f"  D={D:<4} N={N:<5} "
                          f"{res['mfa_ms']:7.2f}ms "
                          f"{res['sdpa_ms']:8.2f}ms "
                          f"  {_ratio_str(res['ratio'])}  {decision}")

    # Summary: where does MFA win?
    if not args.json_only:
        wins  = [r for r in all_results if r["ratio"] >= 1.0]
        total = len(all_results)
        print()
        print(f"MFA wins {len(wins)}/{total} configs "
              f"({100*len(wins)/total:.0f}%)")
        print()
        print("Winners (MFA >= SDPA):")
        for r in sorted(wins, key=lambda x: -x["ratio"]):
            c = "causal" if r["causal"] else "non-causal"
            print(f"  D={r['D']:>3} N={r['N']:>5} {c:<11}  "
                  f"{r['ratio']:.2f}x")
        losers = [r for r in all_results if r["ratio"] < 0.90]
        if losers:
            print()
            print("Worst losers (MFA < 0.90x SDPA):")
            for r in sorted(losers, key=lambda x: x["ratio"]):
                c = "causal" if r["causal"] else "non-causal"
                print(f"  D={r['D']:>3} N={r['N']:>5} {c:<11}  "
                      f"{r['ratio']:.2f}x")

    if not args.no_save:
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "benchmarks",
            "dispatch_matrix.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "date": str(date.today()),
            "device": dev.get("device_name", "unknown"),
            "is_m3_plus": dev.get("is_m3_plus", False),
            "B": BATCH, "H": HEADS,
            "warmup": WARMUP, "timed": TIMED,
            "results": all_results,
        }
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nSaved -> {os.path.abspath(out_path)}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
