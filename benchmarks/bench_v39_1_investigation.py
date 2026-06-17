#!/usr/bin/env python3
"""v2.39.1 root-cause investigation bench harness.

Per /metal-kernel-dev design validation (2026-05-13):
- H3 first (occupancy via (BQ, WM) sweep keeping TQ_per_SG=1)
- H1 second (register pressure via BK sweep, BK ∈ {16, 32} only;
  BK=8 won't compile due to TK=BK/16 constraint)
- H2 third (cache absorption via qL sweep, indirect evidence)

Methodology: 4 warmup + 12 timed iters, median ms, PUBLIC AUTO API via
mx.grad(flash_attention(..., backend="auto")) with MFA_ENABLE_V6_BACKWARD=1.

All sweeps measure fused vs split at D=64 fp16 non-causal, B=2 H=8.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
import mlx.core as mx
_flush = getattr(mx, "eval")


def _bench(B, H, qL, D, dtype, warmup=4, iters=12):
    from mlx_mfa import flash_attention
    scale = D ** -0.5
    q = mx.random.normal((B, H, qL, D)).astype(dtype)
    k = mx.random.normal((B, H, qL, D)).astype(dtype)
    v = mx.random.normal((B, H, qL, D)).astype(dtype)

    def fn(qq, kk, vv):
        return flash_attention(qq, kk, vv, scale=scale, causal=False,
                                backend="auto").sum()
    g = mx.grad(fn, argnums=(0, 1, 2))

    for _ in range(warmup):
        o = g(q, k, v); _flush(*o); mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        o = g(q, k, v); _flush(*o); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return {"med_ms": float(np.median(a)),
            "p25_ms": float(np.percentile(a, 25)),
            "p75_ms": float(np.percentile(a, 75))}


def _set_env(arm, **overrides):
    """Set env for a routing arm (auto/fused/split) plus any kernel knob
    overrides (MFA_V6BWDF_BQ, _BK, _WM, MFA_V6_MAX_THREADS)."""
    keys = (
        "MFA_V6_BWD_KERNEL", "MFA_DISABLE_V6_BACKWARD",
        "MFA_ENABLE_V6_BACKWARD",
        "MFA_V6BWDF_BQ", "MFA_V6BWDF_BK", "MFA_V6BWDF_WM",
        "MFA_V6_MAX_THREADS",
    )
    for k in keys:
        os.environ.pop(k, None)
    if arm == "fused":
        os.environ["MFA_ENABLE_V6_BACKWARD"] = "1"
        os.environ["MFA_V6_BWD_KERNEL"] = "fused"
    elif arm == "split":
        os.environ["MFA_ENABLE_V6_BACKWARD"] = "1"
        os.environ["MFA_V6_BWD_KERNEL"] = "split"
    elif arm == "sdpa":
        os.environ["MFA_DISABLE_V6_BACKWARD"] = "1"
    else:
        raise ValueError(arm)
    for k, v in overrides.items():
        os.environ[k] = str(v)


def run_h3_occupancy_sweep(out):
    """H3: (BQ, WM) ∈ {(64,4), (32,2), (16,1)} keeping TQ_per_SG=1."""
    configs = [
        ("baseline", 64, 4),
        ("BQ32_WM2", 32, 2),
        ("BQ16_WM1", 16, 1),
    ]
    qL, D = 4096, 64
    rows = []
    for name, bq, wm in configs:
        _set_env("fused", MFA_V6BWDF_BQ=bq, MFA_V6BWDF_WM=wm)
        try:
            r = _bench(2, 8, qL, D, mx.float16)
            rows.append({"name": name, "BQ": bq, "WM": wm, **r, "error": None})
            print(f"  H3 {name:12s} BQ={bq:2d} WM={wm}: {r['med_ms']:.2f} ms")
        except Exception as e:
            rows.append({"name": name, "BQ": bq, "WM": wm, "error": str(e)})
            print(f"  H3 {name:12s} BQ={bq:2d} WM={wm}: FAILED ({type(e).__name__})")
    _set_env("split")
    split_baseline = _bench(2, 8, qL, D, mx.float16)
    rows.append({"name": "split_baseline", "BQ": None, "WM": None,
                 **split_baseline, "error": None})
    print(f"  split_baseline: {split_baseline['med_ms']:.2f} ms")
    out["h3_occupancy"] = {"qL": qL, "D": D, "B": 2, "H": 8, "rows": rows}


def run_h1_bk_sweep(out):
    """H1: BK ∈ {16, 32} sweep + MFA_V6_MAX_THREADS variants."""
    configs = [
        ("baseline_BK32", {"MFA_V6BWDF_BK": 32}),
        ("BK16", {"MFA_V6BWDF_BK": 16}),
        ("BK32_maxthr128", {"MFA_V6BWDF_BK": 32, "MFA_V6_MAX_THREADS": 128}),
    ]
    qL, D = 4096, 64
    rows = []
    for name, env in configs:
        _set_env("fused", **env)
        try:
            r = _bench(2, 8, qL, D, mx.float16)
            rows.append({"name": name, "env": env, **r, "error": None})
            print(f"  H1 {name:18s}: {r['med_ms']:.2f} ms  env={env}")
        except Exception as e:
            rows.append({"name": name, "env": env, "error": str(e)})
            print(f"  H1 {name:18s}: FAILED ({type(e).__name__})")
    _set_env("split")
    split_baseline = _bench(2, 8, qL, D, mx.float16)
    rows.append({"name": "split_baseline", **split_baseline, "error": None})
    print(f"  split_baseline: {split_baseline['med_ms']:.2f} ms")
    out["h1_register_pressure"] = {"qL": qL, "D": D, "B": 2, "H": 8, "rows": rows}


def run_h2_qL_sweep(out):
    """H2: qL ∈ {512, 1024, 2048, 4096, 8192, 16384} fused vs split.

    If L1 absorbs split's K-reload, split's relative advantage should be
    MAXIMAL at small qL and decay at large qL.
    """
    qLs = [512, 1024, 2048, 4096, 8192]
    D = 64
    rows = []
    for qL in qLs:
        _set_env("fused")
        rf = _bench(2, 8, qL, D, mx.float16)
        _set_env("split")
        rs = _bench(2, 8, qL, D, mx.float16)
        ratio = rs["med_ms"] / rf["med_ms"]
        rows.append({"qL": qL, "fused_ms": rf, "split_ms": rs,
                     "fused_over_split": ratio})
        print(f"  H2 qL={qL:5d}: fused={rf['med_ms']:.2f}  split={rs['med_ms']:.2f}  f/s={ratio:.3f}×")
    out["h2_cache_absorption"] = {"D": D, "B": 2, "H": 8, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--phase", choices=["h1", "h2", "h3", "all"], default="all")
    args = ap.parse_args()

    from mlx_mfa import get_device_info, __version__
    dev = get_device_info()
    results = {
        "version": __version__,
        "session": args.session,
        "device": dev.get("device_name"),
        "gpu_family_gen": dev.get("gpu_family_gen"),
    }

    if args.phase in ("h3", "all"):
        print("=== H3 occupancy (BQ, WM) sweep ===")
        run_h3_occupancy_sweep(results)
    if args.phase in ("h1", "all"):
        print("\n=== H1 register pressure (BK + max_threads) sweep ===")
        run_h1_bk_sweep(results)
    if args.phase in ("h2", "all"):
        print("\n=== H2 cache absorption (qL) sweep ===")
        run_h2_qL_sweep(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
