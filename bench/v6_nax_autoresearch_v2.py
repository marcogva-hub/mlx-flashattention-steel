#!/usr/bin/env python3
"""V6 NAX tile-dimension autoresearch on M5 Max (Phase 3B).

Sweeps (BLOCK_R, BLOCK_C, executionSIMDGroups) for each production VSR shape,
finds the optimum, and writes a dispatch table.

Usage:
    .venv/bin/python bench/v6_nax_autoresearch.py [--shapes ...] [--quick]

Each (shape, config) is benchmarked in a fresh subprocess so the env var
override (MFA_V6_BLOCK_R=...) is respected at compile time. Invalid configs
(threadgroup-memory overflow, GPU panics, etc.) are caught and recorded.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Search space for tile dimensions (V5 doc §6.6 + a tighter focus).
SEARCH_SPACE = {
    "BLOCK_R": [4, 8, 16, 32, 64],
    "BLOCK_C": [16, 32, 48, 64, 80, 96, 128],
    "EXEC_SG": [2, 4, 8, 12, 16, 24, 32],
}

# Production VSR shapes (matches phase 1).
TARGET_SHAPES = [
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "N_q": 4096,   "N_kv": 4096,   "D": 64,  "dtype": "float16"},
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "N_q": 26730,  "N_kv": 26730,  "D": 128, "dtype": "float16"},
    {"name": "CogVideoX",      "B": 1, "H": 30, "N_q": 70200,  "N_kv": 70200,  "D": 128, "dtype": "float16"},
    {"name": "SeedVR2-large",  "B": 1, "H": 20, "N_q": 111375, "N_kv": 111375, "D": 128, "dtype": "float16"},
]

# Subprocess template: imports mlx_mfa, allocates inputs, times warmup+iters
SUBPROC_TEMPLATE = """
import sys, time, math
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward

B, H, N_q, N_kv, D = __B__, __H__, __NQ__, __NKV__, __D__
dtype = mx.__DTYPE__
warmup = __WARMUP__
iters = __ITERS__

mx.random.seed(42)
q = mx.random.normal((B, H, N_q, D)).astype(dtype)
k = mx.random.normal((B, H, N_kv, D)).astype(dtype)
v = mx.random.normal((B, H, N_kv, D)).astype(dtype)
mx.eval(q, k, v)

try:
    for _ in range(warmup):
        out, _ = v6_nax_forward(q, k, v, False)
        mx.eval(out)
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out, _ = v6_nax_forward(q, k, v, False)
        mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = times[len(times) // 2]
    print("OK," + ("%.4f" % p50))
except Exception as e:
    err = str(e).replace(chr(10), ' ')[:200]
    print("ERR," + type(e).__name__ + ": " + err)
"""


def is_valid_config(block_r, block_c, exec_sg, head_dim):
    """Reject configs that exceed threadgroup memory (32 KB)."""
    elem_size = 2  # FP16
    tgmem = block_r * block_c * exec_sg * elem_size
    if tgmem > 32768:
        return False, f"tgmem {tgmem}B > 32KB"
    return True, "ok"


def benchmark_config(shape, block_r, block_c, exec_sg, warmup=2, iters=5,
                     timeout_s=180):
    """Run a single config in a subprocess. Returns p50 ms or None."""
    valid, reason = is_valid_config(block_r, block_c, exec_sg, shape["D"])
    if not valid:
        return None, reason

    src = (SUBPROC_TEMPLATE
        .replace("__B__", str(shape["B"]))
        .replace("__H__", str(shape["H"]))
        .replace("__NQ__", str(shape["N_q"]))
        .replace("__NKV__", str(shape["N_kv"]))
        .replace("__D__", str(shape["D"]))
        .replace("__DTYPE__", shape["dtype"])
        .replace("__WARMUP__", str(warmup))
        .replace("__ITERS__", str(iters)))

    env = os.environ.copy()
    env["MFA_V6_BLOCK_R"] = str(block_r)
    env["MFA_V6_BLOCK_C"] = str(block_c)
    env["MFA_V6_EXEC_SG"] = str(exec_sg)

    try:
        result = subprocess.run(
            [".venv/bin/python", "-c", src],
            env=env, capture_output=True, text=True, timeout=timeout_s,
        )
        out = result.stdout.strip()
        if not out:
            return None, f"no output (exit {result.returncode})"
        if out.startswith("OK,"):
            return float(out[3:]), "ok"
        return None, out[4:200] if out.startswith("ERR,") else out[:200]
    except subprocess.TimeoutExpired:
        return None, f"timeout {timeout_s}s"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run_sweep(shapes, warmup, iters, output_path):
    configs = list(itertools.product(
        SEARCH_SPACE["BLOCK_R"],
        SEARCH_SPACE["BLOCK_C"],
        SEARCH_SPACE["EXEC_SG"],
    ))
    print(f"Sweep: {len(configs)} configs × {len(shapes)} shapes")
    print(f"Warmup: {warmup} iters | Measured: {iters} iters")
    print()

    results = {}
    overall_t0 = time.perf_counter()
    for shape in shapes:
        print(f"=== {shape['name']} (D={shape['D']}, H={shape['H']}, "
              f"N={shape['N_q']}/{shape['N_kv']}) ===")
        shape_results = []
        for i, (block_r, block_c, exec_sg) in enumerate(configs, 1):
            t0 = time.perf_counter()
            p50, status = benchmark_config(shape, block_r, block_c, exec_sg,
                                            warmup=warmup, iters=iters)
            dt = time.perf_counter() - t0
            cfg_label = f"R={block_r:>2} C={block_c:>3} SG={exec_sg:>2}"
            if p50 is not None:
                shape_results.append({
                    "block_r": block_r, "block_c": block_c, "exec_sg": exec_sg,
                    "p50_ms": p50, "valid": True,
                })
                print(f"  [{i:>2}/{len(configs)}] {cfg_label} → {p50:>9.2f} ms ({dt:.1f}s)")
            else:
                shape_results.append({
                    "block_r": block_r, "block_c": block_c, "exec_sg": exec_sg,
                    "p50_ms": None, "valid": False, "error": status,
                })
                print(f"  [{i:>2}/{len(configs)}] {cfg_label} → INVALID ({status[:60]})")

        valid_results = [r for r in shape_results if r["valid"]]
        if valid_results:
            best = min(valid_results, key=lambda r: r["p50_ms"])
            print(f"  BEST: R={best['block_r']} C={best['block_c']} "
                  f"SG={best['exec_sg']} → {best['p50_ms']:.2f} ms")
            print(f"  (vs default R=32 C=32 SG=4: ", end="")
            default = next((r for r in valid_results
                           if r["block_r"] == 32 and r["block_c"] == 32
                           and r["exec_sg"] == 4), None)
            if default:
                speedup = default["p50_ms"] / best["p50_ms"]
                print(f"{default['p50_ms']:.2f} ms, speedup {speedup:.2f}x)")
            else:
                print("default not measured)")
        else:
            best = None
            print(f"  ALL CONFIGS INVALID")

        results[shape["name"]] = {
            "shape": shape,
            "configs": shape_results,
            "best": best,
        }
        print()

    total_dt = time.perf_counter() - overall_t0
    print(f"Total sweep time: {total_dt/60:.1f} minutes")

    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine": "Apple M5 Max",
        "search_space": SEARCH_SPACE,
        "warmup": warmup,
        "iters": iters,
        "total_seconds": total_dt,
        "results": results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {output_path}")
    return results


def write_dispatch_table(results, output_path):
    """Write the best config per shape as a dispatch table."""
    table = {}
    for name, data in results.items():
        if data["best"] is None:
            continue
        table[name] = {
            "BLOCK_R": data["best"]["block_r"],
            "BLOCK_C": data["best"]["block_c"],
            "EXEC_SG": data["best"]["exec_sg"],
            "p50_ms": data["best"]["p50_ms"],
            "shape": {
                k: data["shape"][k] for k in ["B", "H", "N_q", "N_kv", "D", "dtype"]
            },
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(table, indent=2))
    print(f"Dispatch table → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/v6-nax/autoresearch-tile-results.json")
    parser.add_argument("--dispatch-table", default="docs/v6-nax/v6-dispatch-table.json")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--shapes", nargs="*", default=None,
                        help="Subset of shape names")
    parser.add_argument("--quick", action="store_true",
                        help="Reduce search space + skip large shapes")
    args = parser.parse_args()

    shapes = TARGET_SHAPES
    if args.shapes:
        wanted = set(args.shapes)
        shapes = [s for s in TARGET_SHAPES if s["name"] in wanted]

    if args.quick:
        # Tighter search space + skip the slowest shape
        global SEARCH_SPACE
        SEARCH_SPACE = {
            "BLOCK_R": [16, 32],
            "BLOCK_C": [32, 64],
            "EXEC_SG": [4, 8],
        }
        shapes = [s for s in shapes if s["name"] != "SeedVR2-large"]

    results = run_sweep(shapes, args.warmup, args.iters, args.output)
    write_dispatch_table(results, args.dispatch_table)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
