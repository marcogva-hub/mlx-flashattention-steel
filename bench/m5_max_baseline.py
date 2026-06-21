#!/usr/bin/env python3
"""M5 Max V2 STEEL baseline benchmarks for mlx-mfa.

Records V2 STEEL kernel performance on production VSR shapes (SeedVR2,
FlashVSR, CogVideoX, LTX-2). These numbers serve as the V2 baseline
for V6 NAX comparison in Phase 1.

Usage:
    .venv/bin/python bench/m5_max_baseline.py
    .venv/bin/python bench/m5_max_baseline.py --output docs/v6-nax/m5-max-baseline-v2-steel.json
    .venv/bin/python bench/m5_max_baseline.py --iterations 20 --warmup 5
"""

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx_mfa


# Shape matrix - VSR production workloads + small validation shapes
SHAPES = [
    # SeedVR2 typical (720p 5-frame batch)
    {"name": "SeedVR2-small", "B": 1, "H": 20, "N_q": 26730, "N_kv": 26730, "D": 128, "dtype": "float16"},
    # SeedVR2 large (720p 13-frame batch - worst case production)
    {"name": "SeedVR2-large", "B": 1, "H": 20, "N_q": 111375, "N_kv": 111375, "D": 128, "dtype": "float16"},
    # FlashVSR dense (the path that doesn't use sparse)
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "N_q": 4096, "N_kv": 4096, "D": 64, "dtype": "float16"},
    # CogVideoX / SparkVSR-class long-sequence
    {"name": "CogVideoX", "B": 1, "H": 30, "N_q": 70200, "N_kv": 70200, "D": 128, "dtype": "float16"},
    # LTX-2 audio-to-video cross-attention (asymmetric N_q vs N_kv)
    {"name": "LTX2-cross", "B": 1, "H": 8, "N_q": 2048, "N_kv": 14000, "D": 64, "dtype": "float16"},
    # Smaller validation shapes
    {"name": "small-D64", "B": 1, "H": 8, "N_q": 1024, "N_kv": 1024, "D": 64, "dtype": "float16"},
    {"name": "small-D128", "B": 1, "H": 8, "N_q": 1024, "N_kv": 1024, "D": 128, "dtype": "float16"},
    {"name": "small-D64-bf16", "B": 1, "H": 8, "N_q": 1024, "N_kv": 1024, "D": 64, "dtype": "bfloat16"},
    {"name": "small-D128-bf16", "B": 1, "H": 8, "N_q": 1024, "N_kv": 1024, "D": 128, "dtype": "bfloat16"},
]


def _sysctl(key):
    try:
        result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "?"


def collect_machine_info():
    memory_str = _sysctl("hw.memsize")
    try:
        memory_gb = int(memory_str) // (1024 ** 3)
    except ValueError:
        memory_gb = 0

    try:
        mlx_dev = mx.device_info()
    except (AttributeError, RuntimeError):
        mlx_dev = {}

    try:
        mfa_info = mlx_mfa.get_device_info()
    except Exception:
        mfa_info = {}

    return {
        "machine": platform.machine(),
        "mac_model": _sysctl("hw.model"),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_gb": memory_gb,
        "os_version": platform.mac_ver()[0],
        "python_version": platform.python_version(),
        "mlx_version": mx.__version__,
        "mlx_mfa_version": mlx_mfa.__version__,
        "mlx_device_info": mlx_dev,
        "mfa_device_info": mfa_info,
    }


def _dtype_from_str(s):
    return {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}[s]


def benchmark_shape(shape, warmup, iterations):
    name = shape["name"]
    dtype = _dtype_from_str(shape["dtype"])
    B, H = shape["B"], shape["H"]
    N_q, N_kv, D = shape["N_q"], shape["N_kv"], shape["D"]
    scale = 1.0 / math.sqrt(D)

    record = {
        "name": name,
        "shape": {"B": B, "H": H, "N_q": N_q, "N_kv": N_kv, "D": D},
        "dtype": shape["dtype"],
        "scale": scale,
    }

    # Allocate inputs
    try:
        mx.random.seed(42)
        q = mx.random.normal((B, H, N_q, D)).astype(dtype)
        k = mx.random.normal((B, H, N_kv, D)).astype(dtype)
        v = mx.random.normal((B, H, N_kv, D)).astype(dtype)
        mx.eval(q, k, v)  # materialize before timing
    except RuntimeError as e:
        record["status"] = "OOM"
        record["error"] = f"alloc failed: {e}"
        return record

    bytes_per_q = B * H * N_q * D * (2 if shape["dtype"] != "float32" else 4)
    bytes_per_kv = B * H * N_kv * D * (2 if shape["dtype"] != "float32" else 4)
    total_input_mb = (bytes_per_q + 2 * bytes_per_kv) / (1024 ** 2)
    record["input_size_mb"] = round(total_input_mb, 1)

    # Warmup
    try:
        for _ in range(warmup):
            out = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=False)
            mx.eval(out)
    except Exception as e:
        record["status"] = "ERROR"
        record["error"] = f"warmup failed: {type(e).__name__}: {e}"
        return record

    # Correctness check vs SDPA
    try:
        mx.synchronize()
        out_mfa = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=False)
        out_sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
        mx.eval(out_mfa, out_sdpa)
        diff = mx.subtract(out_mfa.astype(mx.float32), out_sdpa.astype(mx.float32))
        rmse = float(mx.sqrt(mx.mean(mx.square(diff))).item())
        record["correctness_rmse"] = rmse
    except Exception as e:
        record["correctness_rmse"] = None
        record["correctness_error"] = f"{type(e).__name__}: {e}"

    # Measured iterations
    times_ms = []
    try:
        for _ in range(iterations):
            mx.synchronize()
            t0 = time.perf_counter()
            out = mlx_mfa.flash_attention(q, k, v, scale=scale, causal=False)
            mx.eval(out)
            mx.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)
    except Exception as e:
        record["status"] = "ERROR"
        record["error"] = f"measure failed: {type(e).__name__}: {e}"
        record["times_ms"] = times_ms
        return record

    times_ms_sorted = sorted(times_ms)
    record["times_ms"] = [round(t, 3) for t in times_ms]
    record["p50_ms"] = round(times_ms_sorted[len(times_ms_sorted) // 2], 3)
    record["mean_ms"] = round(sum(times_ms) / len(times_ms), 3)
    record["min_ms"] = round(min(times_ms), 3)
    record["max_ms"] = round(max(times_ms), 3)
    record["status"] = "OK"

    return record


def _fmt_ms(value):
    if value is None:
        return "  -  "
    if value >= 1000:
        return f"{value:>8.1f}"
    return f"{value:>8.2f}"


def print_summary(machine_info, results):
    chip = machine_info["mfa_device_info"].get("device_name", machine_info.get("chip", "?"))
    cores = machine_info["mfa_device_info"].get("gpu_cores", "?")
    mlx_v = machine_info["mlx_version"]
    mfa_v = machine_info["mlx_mfa_version"]

    print()
    print(f"{chip} ({cores} cores) - V2 STEEL baseline - MLX {mlx_v} / mlx-mfa {mfa_v}")
    print("=" * 79)
    print(f"{'Shape':<22} {'Dtype':<6} {'N_q':>7} {'D':>4}  {'p50 (ms)':>10}  {'RMSE':>10}  Status")
    print("-" * 79)
    for r in results:
        nq = r["shape"]["N_q"]
        d = r["shape"]["D"]
        dtype = r["dtype"][:6]
        p50 = _fmt_ms(r.get("p50_ms"))
        rmse = r.get("correctness_rmse")
        rmse_s = f"{rmse:.2e}" if rmse is not None else "   -    "
        status = r.get("status", "?")
        print(f"{r['name']:<22} {dtype:<6} {nq:>7} {d:>4}  {p50}  {rmse_s:>10}  {status}")
    print("=" * 79)


def main():
    parser = argparse.ArgumentParser(description="M5 Max V2 STEEL baseline benchmarks")
    parser.add_argument("--output", default="docs/v6-nax/m5-max-baseline-v2-steel.json")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--shapes", nargs="*", default=None)
    args = parser.parse_args()

    machine_info = collect_machine_info()
    print(f"Device: {machine_info['mac_model']} ({machine_info['chip']})")
    print(f"Memory: {machine_info['memory_gb']} GB | macOS {machine_info['os_version']}")
    print(f"MLX {machine_info['mlx_version']} | mlx-mfa {machine_info['mlx_mfa_version']}")
    print(f"Python {machine_info['python_version']}")
    print()
    print(f"Warmup: {args.warmup} iterations | Measured: {args.iterations} iterations")
    print()

    shapes_to_run = SHAPES
    if args.shapes:
        wanted = set(args.shapes)
        shapes_to_run = [s for s in SHAPES if s["name"] in wanted]
        if not shapes_to_run:
            print(f"ERROR: no matching shapes for {args.shapes}", file=sys.stderr)
            sys.exit(1)

    results = []
    for i, shape in enumerate(shapes_to_run, 1):
        print(f"[{i}/{len(shapes_to_run)}] {shape['name']:<22} ", end="", flush=True)
        try:
            record = benchmark_shape(shape, args.warmup, args.iterations)
        except Exception as e:
            record = {
                "name": shape["name"],
                "shape": {k: shape[k] for k in ["B", "H", "N_q", "N_kv", "D"]},
                "dtype": shape["dtype"],
                "status": "ERROR",
                "error": f"{type(e).__name__}: {e}",
            }
        results.append(record)
        status = record.get("status", "?")
        p50 = record.get("p50_ms")
        if p50 is not None:
            print(f"{status} | p50 = {p50:.2f} ms")
        else:
            print(f"{status} | {record.get('error', '')[:60]}")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_info": machine_info,
        "benchmark_config": {
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "mlx_mfa_kernel": "V2 STEEL (automatic dispatch via flash_attention)",
            "comparison_baseline": "mx.fast.scaled_dot_product_attention (SDPA)",
        },
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}")

    print_summary(machine_info, results)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
