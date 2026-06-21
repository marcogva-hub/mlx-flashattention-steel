"""Sprint B Phase 1.1 sub-phase 0 microbench - matmul2d FP16 TFLOPS at Sprint B
per-tile granularity (BT-sized internal cooperative-tensor tiles), amortized
across enough tile pairs that mx.fast.metal_kernel dispatch overhead is
negligible.

The Sprint B sparse kernel will issue ONE dispatch per attention call that
internally loops over (NQ x NK) Q@K^T tile pairs and (NQ x NK) P@V tile pairs.
The relevant per-tile question is "what TFLOPS does matmul2d sustain when the
internal m_tile=n_tile=BT=32 / k_tile=32?", not "what does it cost to dispatch
ONE 32x32 matmul as its own kernel" (which is ~250 us overhead-bound).

This microbench answers the former by measuring (M, N) >> BT with internal
BT-sized tiles. The dominant shape mirrors lcsa_small_seq4k dense-baseline
(M=4096, K=128, N=4096) with internal 32x32x32 cooperative tiles.

Hard gate (design S3): median sustained TFLOPS on dominant shape >= 5 TF.

Smoke gate first (sentinel-fill RMSE oracle at small shape).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

# Per-tile granularity (matmul2d internal cooperative-tensor tiles).
BT = 32  # Sprint B default block tile for lcsa_small_seq4k
INNER_M_TILE = BT
INNER_N_TILE = BT
INNER_K_TILE = 32
EXEC_SG = 1
TG_THREADS = 32 * EXEC_SG

# Sweep grid: amortize dispatch over MANY tile pairs.
# Per-tile granularity stays at 32x32 internal; aggregate (M, N) grows.
M_GRID = [256, 1024, 4096]
N_GRID = [256, 1024, 4096]
K_GRID = [64, 128]  # head_dim D
N_RUNS = 5

# Dominant: matches lcsa_small_seq4k full dense matmul, internal 32x32 tiles.
DOMINANT_M, DOMINANT_K, DOMINANT_N = 4096, 128, 4096
GATE_TFLOPS = 5.0

SMOKE_M, SMOKE_K, SMOKE_N = 256, 128, 256
SMOKE_RMSE_BAR = 5e-3

HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""


def kernel_source(M, K, N):
    return f"""
    constexpr uint M_FULL = {M};
    constexpr uint K_FULL = {K};
    constexpr uint N_FULL = {N};

    auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)A, dextents<int32_t, 2>(K_FULL, M_FULL));
    auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)B, dextents<int32_t, 2>(N_FULL, K_FULL));

    const uint m_origin = threadgroup_position_in_grid.y * {INNER_M_TILE};
    const uint n_origin = threadgroup_position_in_grid.x * {INNER_N_TILE};

    constexpr auto desc = matmul2d_descriptor(
        {INNER_M_TILE}, {INNER_N_TILE}, {INNER_K_TILE},
        false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<{EXEC_SG}>> op;

    auto mA_init = tA.slice<{INNER_K_TILE}, {INNER_M_TILE}>(0, m_origin);
    auto mB_init = tB.slice<{INNER_N_TILE}, {INNER_K_TILE}>(n_origin, 0);

    auto cC = op.get_destination_cooperative_tensor<
        decltype(mA_init), decltype(mB_init), float>();
    #pragma clang loop unroll(full)
    for (ushort k = 0; k < cC.get_capacity(); ++k) {{
        if (cC.is_valid_element(k)) cC[k] = 0.0f;
    }}

    for (uint k_start = 0; k_start < K_FULL; k_start += {INNER_K_TILE}) {{
        auto mA_k = tA.slice<{INNER_K_TILE}, {INNER_M_TILE}>(k_start, m_origin);
        auto mB_k = tB.slice<{INNER_N_TILE}, {INNER_K_TILE}>(n_origin, k_start);
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
        name=f"lcsa_amortized_{M}_{K}_{N}",
        input_names=["A", "B"],
        output_names=["C"],
        source=kernel_source(M, K, N),
        header=HEADER,
        ensure_row_contiguous=True,
    )


def dispatch(kernel, A, B, M, K, N):
    n_tg_x = (N + INNER_N_TILE - 1) // INNER_N_TILE
    n_tg_y = (M + INNER_M_TILE - 1) // INNER_M_TILE
    return kernel(
        inputs=[A, B],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float16],
        grid=(n_tg_x * TG_THREADS, n_tg_y, 1),
        threadgroup=(TG_THREADS, 1, 1),
    )[0]


def smoke_gate():
    mx.random.seed(0)
    A = (mx.random.uniform(0, 1, shape=(SMOKE_M, SMOKE_K)) * 0.1 + 0.05).astype(mx.float16)
    B = (mx.random.uniform(0, 1, shape=(SMOKE_K, SMOKE_N)) * 0.1 + 0.05).astype(mx.float16)
    mx.async_eval(A, B); mx.synchronize()
    try:
        ker = make_kernel(SMOKE_M, SMOKE_K, SMOKE_N)
        C = dispatch(ker, A, B, SMOKE_M, SMOKE_K, SMOKE_N)
        mx.async_eval(C); mx.synchronize()
    except Exception as e:
        return False, {"error": f"smoke dispatch: {e}"}
    oracle = mx.matmul(A.astype(mx.float32), B.astype(mx.float32)).astype(mx.float16)
    mx.async_eval(oracle); mx.synchronize()
    err = mx.abs(C.astype(mx.float32) - oracle.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(err * err)))
    maxerr = float(mx.max(err))
    n_inf = int(mx.sum(mx.isinf(C.astype(mx.float32))))
    n_nan = int(mx.sum(mx.isnan(C.astype(mx.float32))))
    passed = rmse < SMOKE_RMSE_BAR and n_inf == 0 and n_nan == 0
    return passed, {"rmse": rmse, "maxerr": maxerr, "n_inf": n_inf, "n_nan": n_nan,
                    "rmse_bar": SMOKE_RMSE_BAR, "passed": passed}


def time_shape(M, K, N, n_runs):
    mx.random.seed(M * 1000003 + K * 1009 + N)
    A = (mx.random.uniform(0, 1, shape=(M, K)) * 0.1 + 0.05).astype(mx.float16)
    B = (mx.random.uniform(0, 1, shape=(K, N)) * 0.1 + 0.05).astype(mx.float16)
    mx.async_eval(A, B); mx.synchronize()
    try:
        ker = make_kernel(M, K, N)
    except Exception as e:
        return {"M": M, "K": K, "N": N, "error": f"compile: {str(e)[:300]}"}
    try:
        out = dispatch(ker, A, B, M, K, N)
        mx.async_eval(out); mx.synchronize()
    except Exception as e:
        return {"M": M, "K": K, "N": N, "error": f"dispatch: {str(e)[:300]}"}
    times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        out = dispatch(ker, A, B, M, K, N)
        mx.async_eval(out); mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    median_ms = statistics.median(times)
    flops = 2.0 * M * K * N
    tflops = flops / (median_ms * 1e-3) / 1e12
    n_tile_pairs = ((M + INNER_M_TILE - 1) // INNER_M_TILE) * ((N + INNER_N_TILE - 1) // INNER_N_TILE)
    return {"M": M, "K": K, "N": N, "n_tile_pairs": n_tile_pairs,
            "times_ms": times, "median_ms": median_ms,
            "min_ms": min(times), "max_ms": max(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "median_tflops": tflops}


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
                    default="docs/lcsa-nax/lcsa-nax-phase1_1-pertile-microbench.json")
    ap.add_argument("--n-runs", type=int, default=N_RUNS)
    args = ap.parse_args()

    print("=== Smoke gate ===", flush=True)
    smoke_ok, smoke_diag = smoke_gate()
    print(f"smoke: {smoke_diag}", flush=True)
    if not smoke_ok:
        print("SMOKE FAIL - abort microbench", flush=True)
        sys.exit(2)

    print(f"=== Per-tile amortized sweep (BT={BT}, internal {INNER_M_TILE}x{INNER_N_TILE}x{INNER_K_TILE} tiles) ===", flush=True)
    results = []
    for K in K_GRID:
        for M in M_GRID:
            for N in N_GRID:
                r = time_shape(M, K, N, args.n_runs)
                results.append(r)
                if "error" in r:
                    print(f"  M={M:5d} K={K:3d} N={N:5d}: ERROR {r['error']}", flush=True)
                else:
                    print(f"  M={M:5d} K={K:3d} N={N:5d}  "
                          f"({r['n_tile_pairs']:5d} tile-pairs): "
                          f"med={r['median_ms']:.3f} ms  "
                          f"{r['median_tflops']:.2f} TF", flush=True)

    dominant = None
    for r in results:
        if r.get("M") == DOMINANT_M and r.get("K") == DOMINANT_K and r.get("N") == DOMINANT_N:
            dominant = r
            break
    if dominant is None or "error" in dominant:
        verdict = {"gate_passed": False,
                   "reason": "dominant shape missing or errored", "dominant": dominant}
    else:
        verdict = {"gate_passed": dominant["median_tflops"] >= GATE_TFLOPS,
                   "gate_tflops": GATE_TFLOPS,
                   "dominant_median_tflops": dominant["median_tflops"],
                   "dominant_shape": {"M": DOMINANT_M, "K": DOMINANT_K, "N": DOMINANT_N},
                   "interpretation": "amortized over many tile pairs - simulates Sprint B sparse kernel inner-loop throughput"}

    out_data = {
        "phase": "Sprint B Phase 1.1 sub-phase 0 (per-tile matmul2d microbench, amortized)",
        "design_doc": "docs/lcsa-nax/lcsa-nax-design.md S3",
        "internal_tile": {"M": INNER_M_TILE, "N": INNER_N_TILE, "K": INNER_K_TILE,
                          "exec_sg": EXEC_SG, "BT": BT},
        "conditions": capture_conditions(),
        "smoke_gate": smoke_diag,
        "results": results,
        "verdict": verdict,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nVerdict: {verdict}", flush=True)
    print(f"Written: {out_path}", flush=True)
    sys.exit(0 if verdict.get("gate_passed") else 1)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
