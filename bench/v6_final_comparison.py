#!/usr/bin/env python3
"""V6 final comparison — V6 tuned (v3 dispatch table) vs SDPA vs V2 STEEL.

Uses the rigorous protocol from the campaign brief:
- warmup=5, iterations=15, runs=3 (median of medians)
- All same-process for SDPA/V2/V6-default; subprocess for V6 tuned (env var)
"""

import json
import math
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import mlx_mfa
from mlx_mfa import flash_attention
from mlx_mfa._ext import v6_nax_forward

WARMUP = 5
ITERATIONS = 15
RUNS = 3

SHAPES = [
    ("FlashVSR-dense", 1, 10, 10, 4096, 4096, 64),
    ("SeedVR2-small",  1, 20, 20, 26730, 26730, 128),
    ("CogVideoX",      1, 30, 30, 70200, 70200, 128),
    ("SeedVR2-large",  1, 20, 20, 111375, 111375, 128),
]


def benchmark_in_proc(fn, q, k, v, scale, warmup=WARMUP, iters=ITERATIONS):
    for _ in range(warmup):
        out = fn(q, k, v, scale)
        mx.eval(out)
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn(q, k, v, scale)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[len(times) // 2]


def benchmark_in_proc_stable(fn, q, k, v, scale, runs=RUNS):
    medians = [benchmark_in_proc(fn, q, k, v, scale) for _ in range(runs)]
    return sorted(medians)[len(medians) // 2]


def benchmark_v6_tuned_subproc(shape, block_r, block_c, exec_sg,
                                warmup=WARMUP, iters=ITERATIONS, runs=RUNS):
    """Run V6 with specific tile params via subprocess (env var). 3 runs."""
    src = f"""
import sys, time, math
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward

B, H, N_q, N_kv, D = {shape['B']}, {shape['H']}, {shape['N_q']}, {shape['N_kv']}, {shape['D']}
mx.random.seed(42)
q = mx.random.normal((B, H, N_q, D)).astype(mx.float16)
k = mx.random.normal((B, H, N_kv, D)).astype(mx.float16)
v = mx.random.normal((B, H, N_kv, D)).astype(mx.float16)
mx.eval(q, k, v)

medians = []
for run in range({runs}):
    for _ in range({warmup}):
        out, _ = v6_nax_forward(q, k, v, False)
        mx.eval(out)
    times = []
    for _ in range({iters}):
        mx.synchronize()
        t0 = time.perf_counter()
        out, _ = v6_nax_forward(q, k, v, False)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    medians.append(sorted(times)[len(times) // 2])
medians.sort()
median_of_medians = medians[len(medians) // 2]
print("OK," + ("%.4f" % median_of_medians))
"""
    env = os.environ.copy()
    env["MFA_V6_BLOCK_R"] = str(block_r)
    env["MFA_V6_BLOCK_C"] = str(block_c)
    env["MFA_V6_EXEC_SG"] = str(exec_sg)
    result = subprocess.run([".venv/bin/python", "-c", src], env=env,
                             capture_output=True, text=True, timeout=1800)
    out = result.stdout.strip()
    if out.startswith("OK,"):
        return float(out[3:])
    print(f"  V6 tuned subproc failed: {out[:200]}")
    return float('nan')


def main():
    # Load dispatch table (v3 / phase 3B for now; will be updated post-Axe 1)
    dispatch_table_paths = [
        "docs/v6-nax/v6-dispatch-table-v3.json",
        "docs/v6-nax/v6-dispatch-table.json",
    ]
    table = None
    for p in dispatch_table_paths:
        if Path(p).exists():
            with open(p) as f:
                table = json.load(f)
            print(f"Using dispatch table: {p}")
            break
    if not table:
        print("No dispatch table found")
        return

    # Roofline (FLOPS / NAX peak 70e12)
    NAX_TFLOPS = 70e12

    print(f"\n{'Shape':<18} {'V6 tuned':>10} {'V6 def':>10} {'SDPA':>10} {'V2':>10}  {'Vt/SDPA':>8} {'Vt/V2':>7} {'V6 eff':>7}")
    print("-" * 100)
    final = []

    for name, B, Hq, Hk, N_q, N_kv, D in SHAPES:
        shape_dict = {"name": name, "B": B, "H": Hq, "N_q": N_q, "N_kv": N_kv, "D": D}

        mx.random.seed(42)
        q = mx.random.normal((B, Hq, N_q, D)).astype(mx.float16)
        k = mx.random.normal((B, Hk, N_kv, D)).astype(mx.float16)
        v = mx.random.normal((B, Hk, N_kv, D)).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        # V6 default (in-process, 3 runs × 15 iters)
        def v6_def(q, k, v, scale):
            out, _ = v6_nax_forward(q, k, v, False)
            return out
        v6_def_p50 = benchmark_in_proc_stable(v6_def, q, k, v, scale)

        # V6 tuned (subprocess with best tile config from dispatch table)
        if name in table:
            cfg = table[name]
            v6_tuned_p50 = benchmark_v6_tuned_subproc(
                shape_dict, cfg["BLOCK_R"], cfg["BLOCK_C"], cfg["EXEC_SG"])
        else:
            v6_tuned_p50 = float('nan')

        # SDPA (in-process)
        sdpa_p50 = benchmark_in_proc_stable(
            lambda q, k, v, scale: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale),
            q, k, v, scale)

        # V2 STEEL (in-process)
        v2_p50 = benchmark_in_proc_stable(
            lambda q, k, v, scale: flash_attention(q, k, v, scale=scale, causal=False, backend="mfa"),
            q, k, v, scale)

        # Compute roofline efficiency
        flops = 4 * B * Hq * N_q * N_kv * D
        theoretical_ms = flops / NAX_TFLOPS * 1000
        v6_efficiency = (theoretical_ms / v6_tuned_p50) * 100 if v6_tuned_p50 == v6_tuned_p50 else float('nan')

        vt_sdpa = sdpa_p50 / v6_tuned_p50 if v6_tuned_p50 == v6_tuned_p50 else float('nan')
        vt_v2 = v2_p50 / v6_tuned_p50 if v6_tuned_p50 == v6_tuned_p50 else float('nan')

        if v6_tuned_p50 == v6_tuned_p50:
            print(f"{name:<18} {v6_tuned_p50:>9.2f}ms {v6_def_p50:>9.2f}ms {sdpa_p50:>9.2f}ms {v2_p50:>9.2f}ms {vt_sdpa:>7.2f}x {vt_v2:>6.2f}x {v6_efficiency:>5.1f}%")
        else:
            print(f"{name:<18} {'N/A':>10} {v6_def_p50:>9.2f}ms {sdpa_p50:>9.2f}ms {v2_p50:>9.2f}ms")

        final.append({
            "name": name,
            "v6_tuned_ms": v6_tuned_p50, "v6_default_ms": v6_def_p50,
            "sdpa_ms": sdpa_p50, "v2_steel_ms": v2_p50,
            "v6_tuned_vs_sdpa": vt_sdpa, "v6_tuned_vs_v2": vt_v2,
            "theoretical_ms_at_70TFLOPS": theoretical_ms,
            "v6_efficiency_pct": v6_efficiency,
            "tile_config": table.get(name),
        })

    output = {
        "timestamp": "2026-05-03",
        "machine": "Apple M5 Max",
        "protocol": f"warmup={WARMUP}, iters={ITERATIONS}, runs={RUNS}, metric=median-of-medians",
        "results": final,
    }
    out_path = "docs/v6-nax/m5-max-v6-final-comparison.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFinal comparison saved to {out_path}")


if __name__ == "__main__":
    main()
