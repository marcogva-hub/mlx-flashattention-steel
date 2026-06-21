#!/usr/bin/env python3
"""v2.39.0 Option γ fused dK+dV perf measurement (D=64 only, Phase C.1.a).

Methodology: same as v2.38.1 — 4 warmup + 12 timed iters, median ms,
PUBLIC API via mx.grad(flash_attention(..., backend="auto")).

Routing arms (all with MFA_ENABLE_V6_BACKWARD=1):
- "fused": MFA_V6_BWD_KERNEL=fused → new fused kernel
- "split": MFA_V6_BWD_KERNEL=split → v2.38.1 split path
- "sdpa":  MFA_DISABLE_V6_BACKWARD=1 → SDPA-vjp baseline

Per /metal-kernel-dev audit: structural perf win is K-bandwidth amortization
(fused loads K/V once per K-tile vs twice in split).  Expected: smaller qL
sees bigger win (K-reload share larger relative to K-loop work).
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


SHAPES = [
    # (id, B, H, qL, D, dtype_name)
    ("S1", 2, 8, 2048, 64, "f16"),
    ("S2", 2, 8, 4096, 64, "f16"),
    ("S3", 2, 8, 8192, 64, "f16"),
    ("S4", 2, 8, 16384, 64, "f16"),
]


def _dtype(name):
    return {"f16": mx.float16, "bf16": mx.bfloat16}[name]


def _bench_shape(B, H, qL, D, dtype, warmup=4, iters=12):
    from mlx_mfa import flash_attention
    scale = D ** -0.5

    q = mx.random.normal((B, H, qL, D)).astype(dtype)
    k = mx.random.normal((B, H, qL, D)).astype(dtype)
    v = mx.random.normal((B, H, qL, D)).astype(dtype)

    def fn(qq, kk, vv):
        return flash_attention(qq, kk, vv, scale=scale, causal=False,
                                backend="auto").sum()

    grad_fn = mx.grad(fn, argnums=(0, 1, 2))

    for _ in range(warmup):
        out = grad_fn(q, k, v)
        _flush(*out)
        mx.synchronize()

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = grad_fn(q, k, v)
        _flush(*out)
        mx.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times_ms)
    return {
        "med_ms": float(np.median(arr)),
        "p25_ms": float(np.percentile(arr, 25)),
        "p75_ms": float(np.percentile(arr, 75)),
    }


def _set_env(arm):
    for k in ("MFA_V6_BWD_KERNEL", "MFA_DISABLE_V6_BACKWARD",
              "MFA_ENABLE_V6_BACKWARD"):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    from mlx_mfa import get_device_info, __version__
    dev = get_device_info()

    results = {
        "version": __version__,
        "session": args.session,
        "device": dev.get("device_name"),
        "gpu_family_gen": dev.get("gpu_family_gen"),
        "shapes": [],
    }

    for sid, B, H, qL, D, dt in SHAPES:
        dtype = _dtype(dt)
        row = {"id": sid, "B": B, "H": H, "qL": qL, "D": D, "dtype": dt}
        for arm in ("fused", "split", "sdpa"):
            _set_env(arm)
            row[f"{arm}_ms"] = _bench_shape(B, H, qL, D, dtype)

        fused_med = row["fused_ms"]["med_ms"]
        split_med = row["split_ms"]["med_ms"]
        sdpa_med = row["sdpa_ms"]["med_ms"]
        row["fused_vs_split"] = float(split_med / fused_med)
        row["fused_vs_sdpa"] = float(sdpa_med / fused_med)
        row["split_vs_sdpa"] = float(sdpa_med / split_med)
        results["shapes"].append(row)
        print(f"  {sid} D={D} qL={qL}: "
              f"fused={fused_med:.2f}  split={split_med:.2f}  sdpa={sdpa_med:.2f}  "
              f"f/s={row['fused_vs_split']:.3f}x  f/sdpa={row['fused_vs_sdpa']:.2f}x  "
              f"s/sdpa={row['split_vs_sdpa']:.2f}x")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
