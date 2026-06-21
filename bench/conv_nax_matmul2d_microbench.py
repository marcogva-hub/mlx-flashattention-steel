#!/usr/bin/env python3
"""Sub-phase 0 microbench v2 — matmul2d sustained FP16 TFLOPS.

v1 (commit 5e57430) was DEFECTIVE — passed full (M, K, N) as descriptor
params + dispatched one threadgroup → measured non-physical per-TG
throughput. See docs/conv-nax/conv-nax-phase1_1-microbench-blocker.md.

v2 follows the canonical pattern from
csrc/mfa/v6_nax/NAAttentionKernel.cpp:775 and csrc/v6_nax_toolchain_probe.cpp:

  - Descriptor takes PER-TILE dims (M_tile, N_tile, K_tile)
  - Grid dispatches one threadgroup per output tile
  - K-loop inside kernel, accumulating into cooperative_tensor
    with matmul2d_descriptor::mode::multiply_accumulate

Plus a sentinel-fill + RMSE smoke gate on a tiny shape, run BEFORE
any production-shape timing (Phase 1.1 v1 lesson learned).
"""
import argparse, json, math, os, platform, statistics, subprocess
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

SHAPES = [
    ("mid_resnet",                20480,    13824, 512),
    ("up1_resnet",               147456,    13824, 512),
    ("up2_resnet0_chunk_cap",    297000,    13824, 256),
    ("up3_resnet_chunk_cap",     594000,     3456, 128),
    ("up2_resnet_full",         1114112,     6912, 256),
    ("up2_resnet0_peakflops",   1114112,    13824, 256),
    ("probe_floor",                4096,    13824, 512),
    ("probe_ramp",                 8192,    13824, 512),
    ("up3_resnet0_full",        4456448,     6912, 128),
]

DOMINANT_SHAPES = {
    "mid_resnet", "up1_resnet", "up2_resnet0_chunk_cap",
    "up2_resnet_full", "up2_resnet0_peakflops",
}

# Tile config: (32, 32, 32, sg=1) wins on M-skewed shapes (mid_resnet:
# 43 TF vs 19 TF for (64,32,32,sg=4)). Matches V6 NAX choice exactly
# (csrc/mfa/v6_nax/NAAttentionKernel.cpp:775 uses BQ=BK=BD=32, sg=1).
# Verified correct via /tmp/verify_tile_correctness.py: RMSE within
# FP16 noise floor (rel_err = 2.5e-5 at K=13824, sqrt(K)*eps bound).
M_TILE = 32
N_TILE = 32
K_TILE = 32
EXEC_SIMDGROUPS = 1
TG_THREADS = 32 * EXEC_SIMDGROUPS

SMOKE_M, SMOKE_K, SMOKE_N = 128, 64, 64
SMOKE_RMSE_BAR = 1e-2

MAX_WORKING_SET_BYTES = 32 * 1024**3

HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""


def kernel_source(M, K, N, *, m_tile=M_TILE, n_tile=N_TILE, k_tile=K_TILE,
                  exec_sg=EXEC_SIMDGROUPS):
    """Per-shape MPP matmul2d wrapper.

    Layout (matches v6_nax_toolchain_probe.cpp):
      A wrapped as tensor(dextents(K, M))  → A[k, m]
      B wrapped as tensor(dextents(N, K))  → B[n, k]
      C row-major (M, N)                   → C[m, n]
    """
    return f"""
    constexpr uint M_FULL = {M};
    constexpr uint K_FULL = {K};
    constexpr uint N_FULL = {N};

    auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)A, dextents<int32_t, 2>(K_FULL, M_FULL));
    auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)B, dextents<int32_t, 2>(N_FULL, K_FULL));

    const uint m_origin = threadgroup_position_in_grid.y * {m_tile};
    const uint n_origin = threadgroup_position_in_grid.x * {n_tile};

    constexpr auto desc = matmul2d_descriptor(
        {m_tile}, {n_tile}, {k_tile},
        false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<{exec_sg}>> op;

    auto mA_init = tA.slice<{k_tile}, {m_tile}>(0, m_origin);
    auto mB_init = tB.slice<{n_tile}, {k_tile}>(n_origin, 0);

    auto cC = op.get_destination_cooperative_tensor<
        decltype(mA_init), decltype(mB_init), float>();
    #pragma clang loop unroll(full)
    for (ushort k = 0; k < cC.get_capacity(); ++k) {{
        if (cC.is_valid_element(k)) cC[k] = 0.0f;
    }}

    for (uint k_start = 0; k_start < K_FULL; k_start += {k_tile}) {{
        auto mA_k = tA.slice<{k_tile}, {m_tile}>(k_start, m_origin);
        auto mB_k = tB.slice<{n_tile}, {k_tile}>(n_origin, k_start);
        op.run(mA_k, mB_k, cC);
    }}

    #pragma clang loop unroll(full)
    for (ushort k = 0; k < cC.get_capacity(); ++k) {{
        if (cC.is_valid_element(k)) {{
            auto idx = cC.get_multidimensional_index(k);
            uint m_global = m_origin + idx[1];
            uint n_global = n_origin + idx[0];
            if (m_global < M_FULL && n_global < N_FULL) {{
                C[m_global * N_FULL + n_global] = (half)cC[k];
            }}
        }}
    }}
"""


def make_kernel(M, K, N):
    return mx.fast.metal_kernel(
        name=f"matmul2d_v2_{M}_{K}_{N}",
        input_names=["A", "B"],
        output_names=["C"],
        source=kernel_source(M, K, N),
        header=HEADER,
        ensure_row_contiguous=True,
    )


def dispatch_kernel(kernel, A, B, M, K, N):
    n_tg_x = (N + N_TILE - 1) // N_TILE
    n_tg_y = (M + M_TILE - 1) // M_TILE
    return kernel(
        inputs=[A, B],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
        grid=(n_tg_x * TG_THREADS, n_tg_y, 1),
        threadgroup=(TG_THREADS, 1, 1),
    )[0]


def run_smoke_gate():
    """Sentinel-fill + RMSE oracle check on a tiny shape."""
    M, K, N = SMOKE_M, SMOKE_K, SMOKE_N
    mx.random.seed(0)
    A = (mx.random.uniform(0, 1, shape=(M, K)) * 0.1 + 0.05).astype(mx.float16)
    B = (mx.random.uniform(0, 1, shape=(K, N)) * 0.1 + 0.05).astype(mx.float16)
    mx.async_eval(A, B); mx.synchronize()

    kernel = make_kernel(M, K, N)
    try:
        C = dispatch_kernel(kernel, A, B, M, K, N)
        mx.async_eval(C); mx.synchronize()
    except Exception as e:
        return False, {"error": f"dispatch failed: {e}"}

    oracle = mx.matmul(A.astype(mx.float32), B.astype(mx.float32)).astype(mx.float16)
    mx.async_eval(oracle); mx.synchronize()

    C_f32 = C.astype(mx.float32)
    O_f32 = oracle.astype(mx.float32)
    err = mx.abs(C_f32 - O_f32)
    rmse = mx.sqrt(mx.mean(err * err))
    maxerr = mx.max(err)
    n_inf = int(mx.sum(mx.isinf(C_f32)))
    n_nan = int(mx.sum(mx.isnan(C_f32)))
    out_mag = float(mx.max(mx.abs(O_f32)))

    rmse_v = float(rmse)
    maxerr_v = float(maxerr)
    passed = (
        rmse_v < SMOKE_RMSE_BAR
        and maxerr_v < SMOKE_RMSE_BAR * 10
        and n_inf == 0
        and n_nan == 0
    )
    diag = {
        "M": M, "K": K, "N": N,
        "rmse_fp16": rmse_v, "maxerr_fp16": maxerr_v,
        "out_max_abs": out_mag,
        "n_inf": n_inf, "n_nan": n_nan,
        "rmse_bar": SMOKE_RMSE_BAR,
        "passed": passed,
    }
    return passed, diag


def time_matmul2d(kernel, A, B, M, K, N, *, n_runs):
    out = dispatch_kernel(kernel, A, B, M, K, N)
    mx.async_eval(out); mx.synchronize()

    times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        out = dispatch_kernel(kernel, A, B, M, K, N)
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
    for n, c in [("sw_vers", ["sw_vers"]), ("uptime", ["uptime"]),
                 ("uname", ["uname", "-a"])]:
        try:
            out[n] = subprocess.run(c, check=True, capture_output=True,
                                    text=True, timeout=5).stdout.strip()
        except Exception as e:
            out[f"{n}_error"] = str(e)
    try:
        bt = subprocess.run(["sysctl", "-n", "kern.boottime"], check=True,
                            capture_output=True, text=True, timeout=5)
        out["kern_boottime_raw"] = bt.stdout.strip()
    except Exception as e:
        out["boottime_error"] = str(e)
    return out


def run_shape(label, M, K, N, *, n_runs, seed):
    mx.random.seed(seed)
    b = (M * K + K * N + M * N) * 2
    if b > MAX_WORKING_SET_BYTES:
        return {"shape": label, "M": M, "K": K, "N": N,
                "skipped": True,
                "reason": f"working set ~{b/1e9:.1f} GB > "
                          f"{MAX_WORKING_SET_BYTES/1e9:.0f} GB"}
    A = (mx.random.uniform(0, 1, shape=(M, K)) * 0.1 + 0.05).astype(mx.float16)
    B = (mx.random.uniform(0, 1, shape=(K, N)) * 0.1 + 0.05).astype(mx.float16)
    mx.async_eval(A, B); mx.synchronize()

    try:
        kernel = make_kernel(M, K, N)
    except Exception as e:
        return {"shape": label, "M": M, "K": K, "N": N,
                "error": f"compile: {str(e)[:300]}"}
    try:
        t = time_matmul2d(kernel, A, B, M, K, N, n_runs=n_runs)
    except Exception as e:
        return {"shape": label, "M": M, "K": K, "N": N,
                "error": f"dispatch: {str(e)[:300]}"}

    flops = 2.0 * M * K * N
    tf = flops / (t["median_ms"] * 1e-3) / 1e12
    return {
        "shape": label, "M": M, "K": K, "N": N,
        "flops": flops, "gflops": flops / 1e9,
        "achieved_tflops": tf,
        "vs_peak_38tf_pct": 100.0 * tf / 38.0,
        "tile_dims": [M_TILE, N_TILE, K_TILE],
        "exec_simdgroups": EXEC_SIMDGROUPS,
        **t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_label", help="e.g. v2_S1")
    ap.add_argument("--data_path",
                    default="docs/conv-nax/conv-nax-phase1_1-matmul2d-microbench-v2.json")
    ap.add_argument("--cooldown_shape", type=float, default=60.0)
    ap.add_argument("--cooldown_initial", type=float, default=180.0)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_initial_cooldown", action="store_true")
    ap.add_argument("--smoke_only", action="store_true")
    args = ap.parse_args()

    print(f"[harness v2] session={args.session_label}")
    print(f"[harness v2] tile=({M_TILE}x{N_TILE}x{K_TILE}) "
          f"exec_sg={EXEC_SIMDGROUPS} tg_threads={TG_THREADS}")

    print(f"[harness v2] smoke gate (M={SMOKE_M}, K={SMOKE_K}, N={SMOKE_N})...")
    passed, diag = run_smoke_gate()
    print(f"[harness v2] smoke: rmse={diag.get('rmse_fp16', float('nan')):.6f} "
          f"max={diag.get('maxerr_fp16', float('nan')):.6f} "
          f"out_mag={diag.get('out_max_abs', float('nan')):.4f} "
          f"inf={diag.get('n_inf', '?')} nan={diag.get('n_nan', '?')} "
          f"-> {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("[harness v2] STATUS: HARNESS_SELF_TEST_FAILED", file=sys.stderr)
        print(f"[harness v2] diagnostic: {json.dumps(diag, indent=2)}",
              file=sys.stderr)
        sys.exit(2)

    if args.smoke_only:
        print("[harness v2] --smoke_only set; exiting without production runs.")
        return

    if not args.skip_initial_cooldown:
        print(f"[harness v2] initial cooldown {args.cooldown_initial}s")
        time.sleep(args.cooldown_initial)

    record = {
        "session_label": args.session_label,
        "phase": "C-1.1 sub-phase 0 v2 (corrected per-tile matmul2d)",
        "tile_config": {"M_TILE": M_TILE, "N_TILE": N_TILE, "K_TILE": K_TILE,
                        "EXEC_SIMDGROUPS": EXEC_SIMDGROUPS},
        "cooldowns": {"shape_s": args.cooldown_shape,
                      "initial_s": args.cooldown_initial,
                      "skip_initial": args.skip_initial_cooldown},
        "n_runs": args.n_runs,
        "smoke_gate": diag,
        "conditions": capture_conditions(),
        "results": [],
    }

    for label, M, K, N in SHAPES:
        res = run_shape(label, M, K, N, n_runs=args.n_runs, seed=args.seed)
        record["results"].append(res)
        if res.get("skipped"):
            print(f"  {label:<28} SKIP  ({res['reason']})")
        elif "error" in res:
            print(f"  {label:<28} ERROR: {res['error'][:100]}")
        else:
            print(f"  {label:<28} M={M:>8} K={K:>6} N={N:>4}  "
                  f"median={res['median_ms']:>9.2f}ms  "
                  f"TF={res['achieved_tflops']:>5.2f}  "
                  f"({res['vs_peak_38tf_pct']:>5.1f}% peak)")
        time.sleep(args.cooldown_shape)

    p = Path(args.data_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))
    print(f"\n[harness v2] session '{args.session_label}' -> {p}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
