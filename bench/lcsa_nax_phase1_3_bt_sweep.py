"""Sprint B Phase 1.3 - BT autoresearch sweep + SDPA-bias baseline.

Per design S8 row Phase 1.3, sweep block_tile across {16, 32, 64} for each
of the 6 LCSA production shape clusters and record:
  - sparse_attention_nax median latency at each BT
  - mx.fast.scaled_dot_product_attention + float bias median latency (SDPA
    bias-fallback baseline, equivalent to v2.33.1's _sparse_fallback_sdpa)
  - SDPA dense baseline (no bias)

Output: docs/lcsa-nax/lcsa-nax-phase1_3-bt-sweep.json with per-shape per-BT
data + verdict "best BT per cluster".

Three-axis validation S7 smoke gate first (sentinel-fill RMSE at smallest
shape) - exit non-zero before timing if correctness regressed.

Single session, 5 runs/cell, warm-up 2x before timing.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mlx.core as mx

from mlx_mfa.lcsa_nax import sparse_attention_nax

# Production LCSA shape clusters per design S6 table.
SHAPE_CLUSTERS = [
    # name, qL, kL, density, B, Hq, Hk, D, seed
    ("lcsa_small_seq4k",          4096,  4096, 0.24, 1, 12, 12, 128, 100),
    ("lcsa_small_seq4k_sparse",   4096,  4096, 0.07, 1, 12, 12, 128, 101),
    ("lcsa_mid_seq8k",            8192,  8192, 0.12, 1,  8,  8, 128, 102),
    ("lcsa_mid_seq8k_sparse",     8192,  8192, 0.03, 1,  8,  8, 128, 103),
    ("lcsa_large_seq16k",        16384, 16384, 0.12, 1,  4,  4, 128, 104),
    ("lcsa_large_seq16k_sparse", 16384, 16384, 0.03, 1,  4,  4, 128, 105),
]
BT_GRID = [16, 32, 64]
N_RUNS = 5
N_WARMUP = 2

SMOKE_RMSE_BAR = 5e-3


def _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed):
    mx.random.seed(seed)
    Q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(mx.float16)
    K = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    V = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(seed + 1)
    bm = (rng.random((NQ, NK)) < density).astype(np.bool_)
    for q in range(NQ):
        if not bm[q].any():
            bm[q, min(q, NK - 1)] = True
    mask = mx.array(bm)
    # Float bias for SDPA-bias baseline
    full = np.repeat(np.repeat(bm, BT, axis=0), BT, axis=1)
    bias_np = np.where(full, 0.0, -np.inf).astype(np.float16)
    bias = mx.array(bias_np)
    mx.async_eval(Q, K, V, mask, bias); mx.synchronize()
    return Q, K, V, mask, bias


def smoke_gate():
    """Sentinel-fill RMSE at lcsa_small_seq4k - non-zero exit if regressed."""
    qL = kL = 4096
    B, Hq, Hk, D, BT = 1, 4, 4, 128, 32
    Q, K, V, mask, bias = _build_inputs(B, Hq, Hk, qL, kL, D, 0.24, BT, 999)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    O_ref = mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0/math.sqrt(D), mask=bias)
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) - np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    return rmse < SMOKE_RMSE_BAR, {"rmse": rmse, "bar": SMOKE_RMSE_BAR}


def _time_callable(fn, n_runs, n_warmup):
    for _ in range(n_warmup):
        out = fn()
        mx.async_eval(out); mx.synchronize()
    times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.async_eval(out); mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "times_ms": times,
    }


def time_cell(name, qL, kL, density, B, Hq, Hk, D, seed):
    print(f"\n=== {name} (qL=kL={qL}, density={density}, H={Hq}) ===", flush=True)
    cell = {"name": name, "qL": qL, "kL": kL, "density": density,
            "B": B, "Hq": Hq, "Hk": Hk, "D": D,
            "BT_results": {}, "sdpa_bias_ms": None, "sdpa_dense_ms": None}

    # Sparse with BT=32 baseline (also build bias for SDPA-bias)
    BT_ref = 32
    Q, K, V, mask_ref, bias_ref = _build_inputs(B, Hq, Hk, qL, kL, D, density, BT_ref, seed)
    scale = 1.0 / math.sqrt(D)

    # SDPA + float bias (v2.33.1 fast-fallback equivalent)
    def _sdpa_bias():
        return mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=bias_ref)
    t = _time_callable(_sdpa_bias, N_RUNS, N_WARMUP)
    cell["sdpa_bias_ms"] = t["median_ms"]
    print(f"  SDPA+bias median: {t['median_ms']:.3f} ms", flush=True)

    # SDPA dense (no mask)
    def _sdpa_dense():
        return mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale)
    t_d = _time_callable(_sdpa_dense, N_RUNS, N_WARMUP)
    cell["sdpa_dense_ms"] = t_d["median_ms"]
    print(f"  SDPA dense median: {t_d['median_ms']:.3f} ms", flush=True)

    # Sparse NAX across BT grid
    for BT in BT_GRID:
        Q, K, V, mask, _ = _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed)
        def _sparse(BT=BT, mask=mask):
            return sparse_attention_nax(Q, K, V, mask, block_tile=BT)
        try:
            ts = _time_callable(_sparse, N_RUNS, N_WARMUP)
            cell["BT_results"][str(BT)] = ts
            ratio_bias = cell["sdpa_bias_ms"] / ts["median_ms"] if ts["median_ms"] > 0 else 0
            ratio_dense = cell["sdpa_dense_ms"] / ts["median_ms"] if ts["median_ms"] > 0 else 0
            print(f"  BT={BT:2d}: med={ts['median_ms']:.3f} ms  "
                  f"vs SDPA+bias {ratio_bias:.2f}x  "
                  f"vs SDPA dense {ratio_dense:.2f}x", flush=True)
        except Exception as e:
            cell["BT_results"][str(BT)] = {"error": str(e)[:200]}
            print(f"  BT={BT:2d}: ERROR {str(e)[:120]}", flush=True)
    return cell


def capture_conditions():
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "platform": platform.platform()}
    for n, c in [("sw_vers", ["sw_vers"]), ("uname", ["uname", "-a"])]:
        try:
            out[n] = subprocess.run(c, check=True, capture_output=True,
                                    text=True, timeout=5).stdout.strip()
        except Exception as e:
            out[f"{n}_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",
                    default="docs/lcsa-nax/lcsa-nax-phase1_3-bt-sweep.json")
    args = ap.parse_args()

    print("=== Smoke gate ===", flush=True)
    ok, diag = smoke_gate()
    print(f"smoke: {diag}", flush=True)
    if not ok:
        print("SMOKE FAIL - abort sweep", flush=True)
        sys.exit(2)

    cells = []
    for shape in SHAPE_CLUSTERS:
        try:
            cells.append(time_cell(*shape))
        except Exception as e:
            cells.append({"name": shape[0], "error": str(e)[:300]})
            print(f"  CELL ERROR: {e}", flush=True)

    # Verdict: pick best BT per cluster vs SDPA+bias (production baseline).
    verdict = []
    for c in cells:
        if "error" in c:
            verdict.append({"cluster": c["name"], "best_BT": None,
                             "reason": c.get("error")})
            continue
        valid = [(int(bt), r["median_ms"]) for bt, r in c["BT_results"].items()
                 if "error" not in r]
        if not valid:
            verdict.append({"cluster": c["name"], "best_BT": None,
                             "reason": "all BTs errored"})
            continue
        best_BT, best_ms = min(valid, key=lambda x: x[1])
        ratio_bias = c["sdpa_bias_ms"] / best_ms if best_ms > 0 else 0
        ratio_dense = c["sdpa_dense_ms"] / best_ms if best_ms > 0 else 0
        verdict.append({
            "cluster": c["name"], "best_BT": best_BT,
            "best_BT_ms": best_ms,
            "sdpa_bias_ms": c["sdpa_bias_ms"],
            "sdpa_dense_ms": c["sdpa_dense_ms"],
            "vs_sdpa_bias": ratio_bias,
            "vs_sdpa_dense": ratio_dense,
        })

    out_data = {
        "phase": "Sprint B Phase 1.3 BT autoresearch sweep",
        "design_doc": "docs/lcsa-nax/lcsa-nax-design.md S8 row Phase 1.3",
        "conditions": capture_conditions(),
        "smoke_gate": diag,
        "cells": cells,
        "verdict": verdict,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nVerdict (best BT per cluster):", flush=True)
    for v in verdict:
        if v["best_BT"] is None:
            print(f"  {v['cluster']}: ERROR ({v.get('reason')})", flush=True)
        else:
            print(f"  {v['cluster']}: BT={v['best_BT']}  "
                  f"{v['best_BT_ms']:.3f} ms  "
                  f"vs SDPA+bias {v['vs_sdpa_bias']:.2f}x  "
                  f"vs SDPA dense {v['vs_sdpa_dense']:.2f}x", flush=True)
    print(f"\nWritten: {out_path}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
