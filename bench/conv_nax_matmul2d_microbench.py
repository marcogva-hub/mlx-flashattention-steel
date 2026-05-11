#!/usr/bin/env python3
"""Sub-phase 0 microbench — matmul2d sustained FP16 TFLOPS.

!!! METHODOLOGY BLOCKER — DO NOT USE FOR GATE DECISION !!!
==========================================================
This harness was authored on 2026-05-11 and a smoke test produced
physically-impossible readings (101 TFLOPS on mid_resnet shape,
NAX FP16 peak is ~38 TF). Investigation traced the issue to
incorrect matmul2d_descriptor usage:

  - Apple's `matmul2d_descriptor(M, N, K)` parameters are PER-TILE
    dimensions (small, e.g. 32/64/128). Each threadgroup computes
    one M_tile × N_tile output tile.
  - Full-matrix coverage requires dispatching a 2D grid of
    threadgroups: MTLSize((M_full + M_tile - 1) / M_tile,
                          (N_full + N_tile - 1) / N_tile, 1)
  - Reference implementation: csrc/mfa/v6_nax/NAAttentionKernel.cpp:775
    uses BLOCK_DIMENSIONS_* (≤128) for descriptor + grid dispatch
    for full-matrix coverage.

This harness passes the FULL (M=20480, K=13824, N=512) as descriptor
params and dispatches only ONE threadgroup → it measures something
unphysical (likely a partial / garbage compute by a single TG).

See docs/conv-nax/conv-nax-phase1_1-microbench-blocker.md for the
diagnostic + paths forward. Phase 1.1 sub-phase 0 is BLOCKED pending
Marco's direction on methodology revision.
=========================================================="""
import argparse, json, math, os, platform, statistics, subprocess
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

# Grid per design §3.1 + small probes
SHAPES = [
    # Production-relevant chunked-matmul shapes
    ("mid_resnet",     20480,    13824, 512),   # smallest, single-chunk
    ("up1_resnet",     147456,   13824, 512),   # single-chunk
    ("up2_resnet0_chunk_cap", 297000, 13824, 256),  # up2_resnet0 chunk cap
    ("up3_resnet_chunk_cap",  594000,  3456, 128),  # up3_resnet chunk cap
    ("up2_resnet_full",      1114112,  6912, 256),  # peak shape
    ("up2_resnet0_peakflops", 1114112, 13824, 256), # peak FLOPs shape
    # Small probes for fitting TFLOPS-vs-M curve
    ("probe_floor", 4096,  13824, 512),
    ("probe_ramp",  8192,  13824, 512),
    # Sanity reference (may OOM at >4GB working set; conditional)
    ("up3_resnet0_full", 4456448, 6912, 128),
]

# MPP matmul2d kernel source (per Sprint 3 microbench learnings).
HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

def kernel_source(M, K, N):
    """Generate matmul2d wrapping kernel for a specific (M, K, N) shape."""
    return f"""
    // matmul2d shape M={M}, K={K}, N={N} (FP16, relaxed_precision, multiply mode)
    auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)A, dextents<int32_t, 2>({K}, {M}));
    auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)B, dextents<int32_t, 2>({N}, {K}));

    constexpr auto desc = matmul2d_descriptor(
        {M}, {N}, {K},
        /*leftT=*/false, /*rightT=*/false, /*relaxed=*/true,
        matmul2d_descriptor::mode::multiply);

    matmul2d<desc, execution_simdgroups<4>> op;
    auto cC = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
    op.run(tA, tB, cC);

    // Store cooperative tensor → device memory (column-major: idx[0]=N, idx[1]=M).
    for (ushort k = 0; k < cC.get_capacity(); ++k) {{
        if (cC.is_valid_element(k)) {{
            auto idx = cC.get_multidimensional_index(k);
            C[idx[1] * {N} + idx[0]] = (half)cC[k];
        }}
    }}
"""


def make_kernel(M, K, N):
    return mx.fast.metal_kernel(
        name=f"matmul2d_{M}_{K}_{N}",
        input_names=["A", "B"],
        output_names=["C"],
        source=kernel_source(M, K, N),
        header=HEADER,
    )


def time_matmul2d(kernel, A, B, M, K, N, *, n_runs):
    """Time a single (M, K, N) matmul2d via the JIT kernel."""
    # Warmup compile + cache
    out = kernel(
        inputs=[A, B],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
        grid=(128, 1, 1),
        threadgroup=(128, 1, 1),
    )[0]
    mx.async_eval(out); mx.synchronize()

    times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        out = kernel(
            inputs=[A, B],
            output_shapes=[(M, N)],
            output_dtypes=[mx.float16],
            grid=(128, 1, 1),
            threadgroup=(128, 1, 1),
        )[0]
        mx.async_eval(out); mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "times_ms": times,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def capture_conditions():
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
    }
    for cmd_name, cmd in [("sw_vers", ["sw_vers"]),
                          ("uptime", ["uptime"]),
                          ("uname", ["uname", "-a"])]:
        try:
            out[cmd_name] = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception as e:
            out[f"{cmd_name}_error"] = str(e)
    try:
        bt = subprocess.run(["sysctl", "-n", "kern.boottime"],
                            check=True, capture_output=True, text=True, timeout=5)
        out["kern_boottime_raw"] = bt.stdout.strip()
    except Exception as e:
        out["boottime_error"] = str(e)
    return out


def run_shape(label, M, K, N, *, n_runs, seed):
    """Allocate, time, and report a single (M, K, N) microbench cell."""
    mx.random.seed(seed)
    # Check memory budget — skip if expected working set > 32 GB.
    bytes_estimate = (M * K + K * N + M * N) * 2  # FP16
    if bytes_estimate > 32 * 1024**3:
        return {
            "shape": label, "M": M, "K": K, "N": N,
            "skipped": True,
            "reason": f"working set ~{bytes_estimate/1e9:.1f} GB exceeds 32 GB budget",
        }
    # Use small magnitude inputs to avoid FP16 overflow on K=13824 reductions
    A = (mx.random.uniform(0, 1, shape=(M, K)) * 0.1 + 0.05).astype(mx.float16)
    B = (mx.random.uniform(0, 1, shape=(K, N)) * 0.1 + 0.05).astype(mx.float16)
    mx.async_eval(A, B); mx.synchronize()

    try:
        kernel = make_kernel(M, K, N)
    except Exception as e:
        return {"shape": label, "M": M, "K": K, "N": N, "error": str(e)[:300]}

    try:
        timing = time_matmul2d(kernel, A, B, M, K, N, n_runs=n_runs)
    except Exception as e:
        return {"shape": label, "M": M, "K": K, "N": N, "error": str(e)[:300]}

    flops = 2.0 * M * K * N
    achieved_tflops = flops / (timing["median_ms"] * 1e-3) / 1e12

    return {
        "shape": label,
        "M": M, "K": K, "N": N,
        "flops": flops,
        "gflops": flops / 1e9,
        "achieved_tflops": achieved_tflops,
        "vs_peak_38tf_pct": 100.0 * achieved_tflops / 38.0,
        **timing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_label", help="e.g. v1_S1")
    ap.add_argument("--data_path",
                    default="docs/conv-nax/conv-nax-phase1_1-matmul2d-microbench.json")
    ap.add_argument("--cooldown_shape", type=float, default=60.0)
    ap.add_argument("--cooldown_initial", type=float, default=180.0)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[harness] session={args.session_label}")
    print(f"[harness] cooldowns: shape={args.cooldown_shape}s initial={args.cooldown_initial}s")
    print(f"[harness] initial cooldown {args.cooldown_initial}s")
    time.sleep(args.cooldown_initial)

    record = {
        "session_label": args.session_label,
        "phase": "C-1.1 sub-phase 0 (matmul2d sustained TFLOPS microbench)",
        "cooldowns": {
            "shape_s": args.cooldown_shape,
            "initial_s": args.cooldown_initial,
            "deviation_from_§4": args.cooldown_shape != 60.0 or args.cooldown_initial != 180.0,
        },
        "n_runs": args.n_runs,
        "conditions": capture_conditions(),
        "results": [],
    }

    for label, M, K, N in SHAPES:
        res = run_shape(label, M, K, N, n_runs=args.n_runs, seed=args.seed)
        record["results"].append(res)
        if res.get("skipped"):
            print(f"  {label:<28} SKIP  ({res['reason']})")
        elif "error" in res:
            print(f"  {label:<28} ERROR: {res['error'][:80]}")
        else:
            print(f"  {label:<28} M={M:>8} K={K:>6} N={N:>4}  "
                  f"median={res['median_ms']:>8.2f}ms  "
                  f"TF={res['achieved_tflops']:>5.2f}  "
                  f"({res['vs_peak_38tf_pct']:>5.1f}% peak)")
        time.sleep(args.cooldown_shape)

    data_path = Path(args.data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        existing = json.loads(data_path.read_text())
    else:
        existing = []
    existing.append(record)
    data_path.write_text(json.dumps(existing, indent=2))
    print(f"\n[harness] session '{args.session_label}' appended to {data_path}")


if __name__ == "__main__":
    main()
