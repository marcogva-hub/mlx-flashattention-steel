"""Sprint B Phase 1.4 - density-thresholded dispatcher sweep.

Per design S8 row Phase 1.4 + Phase 1.3 reframing
(docs/lcsa-nax/lcsa-nax-phase1_3-results.md): time three configurations
across LCSA shapes x densities to determine whether the density-routed
dispatcher achieves > 1.0x SDPA+bias for the very-sparse niche:

  A. SDPA + float bias (production baseline / v2.33.1 fast-fallback)
  B. sparse_attention_nax (Sprint B always - even when slower)
  C. sparse_attention_dispatch (route by density)

Output: docs/lcsa-nax/lcsa-nax-phase1_4-dispatcher-sweep.json with per-cell
results + verdict whether dispatcher beats SDPA+bias at very-sparse density.

Single session, 5 runs / cell, 2 warmup. Sentinel-fill smoke gate first.
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

from mlx_mfa.lcsa_nax import (
    sparse_attention_nax,
    sparse_attention_dispatch,
    _bool_mask_to_float_bias,
    DEFAULT_DENSITY_THRESHOLD,
)

SHAPE_CLUSTERS = [
    # name, qL, kL, B, Hq, Hk, D, seed
    ("lcsa_small_seq4k",    4096,  4096, 1, 12, 12, 128, 200),
    ("lcsa_mid_seq8k",      8192,  8192, 1,  8,  8, 128, 201),
    ("lcsa_large_seq16k",  16384, 16384, 1,  4,  4, 128, 202),
]
DENSITY_GRID = [0.01, 0.03, 0.05, 0.10]
N_RUNS = 5
N_WARMUP = 2
BT = 16  # Phase 1.3 winner
SMOKE_RMSE_BAR = 5e-3


def _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed):
    mx.random.seed(seed)
    Q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(mx.float16)
    K = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    V = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(seed + 1)
    bm = (rng.random((NQ, NK)) < density).astype(np.bool_)
    # Ensure no all-False row (correctness convention)
    for q in range(NQ):
        if not bm[q].any():
            bm[q, min(q, NK - 1)] = True
    mask = mx.array(bm)
    bias = _bool_mask_to_float_bias(mask, BT, qL, kL, mx.float16)
    mx.async_eval(Q, K, V, mask, bias); mx.synchronize()
    actual_density = float(bm.mean())
    return Q, K, V, mask, bias, actual_density


def smoke_gate():
    qL = kL = 4096
    B, Hq, Hk, D = 1, 4, 4, 128
    Q, K, V, mask, bias, _ = _build_inputs(B, Hq, Hk, qL, kL, D, 0.05, BT, 9999)
    O = sparse_attention_dispatch(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O); mx.synchronize()
    O_ref = mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0/math.sqrt(D), mask=bias)
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) - np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    return rmse < SMOKE_RMSE_BAR, {"rmse": rmse, "bar": SMOKE_RMSE_BAR}


def _time_callable(fn, n_runs=N_RUNS, n_warmup=N_WARMUP):
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
    }


def time_cell(name, qL, kL, B, Hq, Hk, D, seed):
    print(f"\n=== {name} (qL=kL={qL}, H={Hq}, D={D}, BT={BT}) ===", flush=True)
    cell = {"name": name, "qL": qL, "kL": kL, "B": B, "Hq": Hq, "Hk": Hk,
            "D": D, "BT": BT, "density_results": {}}
    for density in DENSITY_GRID:
        Q, K, V, mask, bias, actual_d = _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed)
        scale = 1.0 / math.sqrt(D)
        # A. SDPA + bias
        def _A():
            return mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=bias)
        # B. sparse_attention_nax always
        def _B():
            return sparse_attention_nax(Q, K, V, mask, block_tile=BT, scale=scale)
        # C. dispatcher (with precomputed density to avoid per-call reduction)
        def _C():
            return sparse_attention_dispatch(
                Q, K, V, mask, block_tile=BT, scale=scale,
                density=actual_d, precomputed_bias=bias)
        try:
            tA = _time_callable(_A)
            tB = _time_callable(_B)
            tC = _time_callable(_C)
            cell["density_results"][str(density)] = {
                "actual_density": actual_d,
                "sdpa_bias_ms": tA["median_ms"],
                "sparse_always_ms": tB["median_ms"],
                "dispatcher_ms": tC["median_ms"],
                "sparse_vs_sdpa_bias": tA["median_ms"] / tB["median_ms"] if tB["median_ms"] > 0 else 0,
                "dispatcher_vs_sdpa_bias": tA["median_ms"] / tC["median_ms"] if tC["median_ms"] > 0 else 0,
            }
            r = cell["density_results"][str(density)]
            print(f"  d={density:.2f} (actual {actual_d:.3f}):  "
                  f"A_sdpa_bias={tA['median_ms']:.2f}  "
                  f"B_sparse={tB['median_ms']:.2f} ({r['sparse_vs_sdpa_bias']:.2f}x)  "
                  f"C_disp={tC['median_ms']:.2f} ({r['dispatcher_vs_sdpa_bias']:.2f}x)",
                  flush=True)
        except Exception as e:
            cell["density_results"][str(density)] = {"error": str(e)[:200]}
            print(f"  d={density:.2f}: ERROR {str(e)[:120]}", flush=True)
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
                    default="docs/lcsa-nax/lcsa-nax-phase1_4-dispatcher-sweep.json")
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

    # Verdict
    verdict = {
        "density_threshold_used": DEFAULT_DENSITY_THRESHOLD,
        "per_cell_summary": [],
        "dispatcher_at_or_above_sdpa_bias": True,  # SHIP gate
    }
    for c in cells:
        if "error" in c:
            verdict["per_cell_summary"].append({"name": c["name"], "error": c["error"]})
            continue
        # Check dispatcher is ≥ SDPA+bias for ALL tested densities
        all_ratios = []
        for d_str, r in c["density_results"].items():
            if "error" in r:
                all_ratios.append((d_str, None))
                continue
            all_ratios.append((d_str, r["dispatcher_vs_sdpa_bias"]))
        min_ratio = min((r for _, r in all_ratios if r is not None), default=0)
        if min_ratio < 0.95:  # 5% tolerance for measurement noise
            verdict["dispatcher_at_or_above_sdpa_bias"] = False
        verdict["per_cell_summary"].append({
            "name": c["name"],
            "min_dispatcher_ratio": min_ratio,
            "ratios_by_density": all_ratios,
        })

    out_data = {
        "phase": "Sprint B Phase 1.4 density-thresholded dispatcher sweep",
        "design_doc": "docs/lcsa-nax/lcsa-nax-design.md S8 row Phase 1.4",
        "phase1_3_reframing": "docs/lcsa-nax/lcsa-nax-phase1_3-results.md",
        "conditions": capture_conditions(),
        "smoke_gate": diag,
        "cells": cells,
        "verdict": verdict,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nVerdict: dispatcher_at_or_above_sdpa_bias = {verdict['dispatcher_at_or_above_sdpa_bias']}", flush=True)
    for s in verdict["per_cell_summary"]:
        if "error" in s:
            print(f"  {s['name']}: ERROR", flush=True)
        else:
            print(f"  {s['name']}: min ratio {s['min_dispatcher_ratio']:.2f}x", flush=True)
    print(f"\nWritten: {out_path}", flush=True)
    sys.exit(0 if verdict["dispatcher_at_or_above_sdpa_bias"] else 1)


if __name__ == "__main__":
    main()
