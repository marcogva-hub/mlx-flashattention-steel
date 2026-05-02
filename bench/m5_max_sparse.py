#!/usr/bin/env python3
"""M5 Max sparse-path benchmarks.

Currently uses the SDPA fallback (the V1 STEEL sparse Metal kernel has a
miscompile bug on M5/MSL-4 — see docs/v6-nax/sparse-bug-investigation.md).

Once the kernel is fixed (or replaced with V6 NAX sparse), update this
script to compare native vs SDPA fallback.
"""

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx_mfa
from mlx_mfa import flash_attention_sparse


def make_local_window_mask(NQ: int, NK: int, window_radius: int) -> mx.array:
    """Block mask: each Q-tile attends to K-tiles within window_radius."""
    mask_np = np.zeros((NQ, NK), dtype=bool)
    for q in range(NQ):
        # Approximate: Q-tile q maps to a center K-tile
        center = int(q * NK / NQ)
        lo = max(0, center - window_radius)
        hi = min(NK, center + window_radius + 1)
        mask_np[q, lo:hi] = True
    return mx.array(mask_np)


def make_random_block_mask(NQ: int, NK: int, density: float) -> mx.array:
    """Block mask: random pattern with given density (fraction of True)."""
    rng = np.random.default_rng(42)
    mask_np = rng.random((NQ, NK)) < density
    return mx.array(mask_np)


SPARSE_SHAPES = [
    # FlashVSR LCSA-class shapes
    {"name": "FlashVSR-LCSA-small", "B": 1, "H": 10, "N_q": 4096, "N_kv": 4096,
     "D": 64, "dtype": "float16", "mask_type": "window", "param": 4},
    {"name": "FlashVSR-LCSA-medium", "B": 1, "H": 10, "N_q": 16384, "N_kv": 16384,
     "D": 64, "dtype": "float16", "mask_type": "window", "param": 8},
    # Generic sparse shapes
    {"name": "sparse-small-D64-50pct", "B": 1, "H": 8, "N_q": 2048, "N_kv": 2048,
     "D": 64, "dtype": "float16", "mask_type": "random", "param": 0.5},
    {"name": "sparse-small-D128-50pct", "B": 1, "H": 8, "N_q": 2048, "N_kv": 2048,
     "D": 128, "dtype": "float16", "mask_type": "random", "param": 0.5},
    {"name": "sparse-D128-25pct", "B": 1, "H": 8, "N_q": 4096, "N_kv": 4096,
     "D": 128, "dtype": "float16", "mask_type": "random", "param": 0.25},
]


def _dtype(s):
    return {"float16": mx.float16, "bfloat16": mx.bfloat16}[s]


def benchmark_sparse(shape, warmup=3, iterations=10):
    name = shape["name"]
    dtype = _dtype(shape["dtype"])
    B, H = shape["B"], shape["H"]
    N_q, N_kv, D = shape["N_q"], shape["N_kv"], shape["D"]
    scale = 1.0 / math.sqrt(D)

    from mlx_mfa.attention import _steel_block_config
    BQ, BK = _steel_block_config(D)
    NQ = (N_q + BQ - 1) // BQ
    NK = (N_kv + BK - 1) // BK

    record = {
        "name": name,
        "shape": {"B": B, "H": H, "N_q": N_q, "N_kv": N_kv, "D": D},
        "dtype": shape["dtype"],
        "mask": {"type": shape["mask_type"], "param": shape["param"], "NQ": NQ, "NK": NK},
    }

    # Allocate
    try:
        mx.random.seed(42)
        q = mx.random.normal((B, H, N_q, D)).astype(dtype)
        k = mx.random.normal((B, H, N_kv, D)).astype(dtype)
        v = mx.random.normal((B, H, N_kv, D)).astype(dtype)
        mx.eval(q, k, v)
    except RuntimeError as e:
        record["status"] = "OOM"
        record["error"] = str(e)
        return record

    # Build mask
    if shape["mask_type"] == "window":
        mask = make_local_window_mask(NQ, NK, shape["param"])
    elif shape["mask_type"] == "random":
        mask = make_random_block_mask(NQ, NK, shape["param"])
    else:
        mask = mx.ones((NQ, NK), dtype=mx.bool_)
    density = float(mask.astype(mx.float32).mean().item())
    record["mask"]["density"] = round(density, 3)

    # Warmup + measure
    try:
        for _ in range(warmup):
            out = flash_attention_sparse(q, k, v, mask, scale=scale)
            mx.eval(out)

        times_ms = []
        for _ in range(iterations):
            mx.synchronize()
            t0 = time.perf_counter()
            out = flash_attention_sparse(q, k, v, mask, scale=scale)
            mx.eval(out)
            mx.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000)
    except Exception as e:
        record["status"] = "ERROR"
        record["error"] = f"{type(e).__name__}: {e}"
        return record

    times_sorted = sorted(times_ms)
    record["times_ms"] = [round(t, 3) for t in times_ms]
    record["p50_ms"] = round(times_sorted[len(times_sorted) // 2], 3)
    record["mean_ms"] = round(sum(times_ms) / len(times_ms), 3)
    record["min_ms"] = round(min(times_ms), 3)
    record["max_ms"] = round(max(times_ms), 3)
    record["status"] = "OK"
    record["note"] = "M5 SDPA fallback (V1 STEEL sparse kernel disabled due to MSL-4 miscompile)"
    return record


def main():
    parser = argparse.ArgumentParser(description="M5 Max sparse benchmarks")
    parser.add_argument("--output", default="docs/v6-nax/m5-max-sparse-baseline.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    info = mlx_mfa.get_device_info()
    print(f"Device: {info.get('device_name', '?')} ({info.get('gpu_cores', '?')} cores)")
    print(f"M5+: {info.get('is_m5_plus', '?')} | NAX: {info.get('has_nax', '?')}")
    print(f"MLX {mx.__version__} | mlx-mfa {mlx_mfa.__version__}")
    print()
    print(f"Note: sparse path uses Python SDPA fallback on M5+ due to Metal-compiler miscompile.")
    print(f"      See docs/v6-nax/sparse-bug-investigation.md")
    print()

    results = []
    for i, shape in enumerate(SPARSE_SHAPES, 1):
        print(f"[{i}/{len(SPARSE_SHAPES)}] {shape['name']:<28} ", end="", flush=True)
        rec = benchmark_sparse(shape, args.warmup, args.iterations)
        results.append(rec)
        if rec.get("p50_ms") is not None:
            print(f"{rec['status']} | density={rec['mask']['density']:.2f} | p50={rec['p50_ms']:.2f} ms")
        else:
            print(f"{rec['status']} | {rec.get('error', '')[:60]}")

    # Save JSON
    machine_info = {
        "machine": platform.machine(),
        "mac_model": subprocess.run(["sysctl", "-n", "hw.model"],
                                     capture_output=True, text=True).stdout.strip(),
        "memory_gb": int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                          capture_output=True, text=True).stdout.strip()) // (1024**3),
        "os_version": platform.mac_ver()[0],
        "python_version": platform.python_version(),
        "mlx_version": mx.__version__,
        "mlx_mfa_version": mlx_mfa.__version__,
        "mfa_device_info": info,
    }
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_info": machine_info,
        "benchmark_config": {
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "kernel_path": "Python SDPA fallback (M5+ workaround — see investigation report)",
        },
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}")

    # Summary
    print()
    print(f"{'Shape':<30} {'Density':>8} {'p50 (ms)':>10}")
    print("-" * 52)
    for r in results:
        d = r['mask'].get('density', 0) if 'mask' in r else 0
        p50 = r.get('p50_ms')
        p50_s = f"{p50:.2f}" if p50 is not None else "  -  "
        print(f"{r['name']:<30} {d:>8.3f} {p50_s:>10}")


if __name__ == "__main__":
    main()
