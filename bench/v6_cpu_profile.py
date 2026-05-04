#!/usr/bin/env python3
"""V6 NAX vs SDPA CPU-side profiling.

Since GPU counter data in .gputrace is in Apple's proprietary MTSP/xdic
binary format (not parseable without Xcode), this profiler measures what
we CAN observe from the host:

  1. End-to-end attention call wall time
  2. Kernel-only time (between async_eval and synchronize)
  3. Per-op breakdown for V6's pre-kernel transposes/contiguous calls
  4. Implied dispatch count (V6 = transpose×3 + contig×3 + kernel + transpose×1)
  5. Peak memory delta per call (proxy for memory traffic)

This directly tests:
  - Hypothesis 1 (dispatch fragmentation): kernel-only vs end-to-end ratio
  - Hypothesis 4 (memory patterns): peak memory delta

Hypotheses 2 (register spill) and 3 (occupancy) require Xcode GUI for
counter access — documented as out-of-scope for programmatic analysis.
"""
import gc, json, math, statistics, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward

ROOT = Path(__file__).resolve().parent.parent

SHAPES = [
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "Nq": 4096,  "Nkv": 4096,  "D": 64,  "R": 16, "C": 64, "SG": 16},
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "Nq": 26730, "Nkv": 26730, "D": 128, "R": 16, "C": 48, "SG": 16},
]

WARMUP = 3
ITERS = 20


def time_op(fn, warmup=WARMUP, iters=ITERS):
    """Time a callable that returns mlx arrays. Returns (median_ms, stdev_ms)."""
    for _ in range(warmup):
        out = fn()
        if not isinstance(out, (list, tuple)): out = [out]
        mx.async_eval(*out); mx.synchronize()
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if not isinstance(out, (list, tuple)): out = [out]
        mx.async_eval(*out); mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times)//2], statistics.stdev(times) if len(times) > 1 else 0.0


def measure_v6_breakdown(s):
    """Time each step of V6's pre-kernel pipeline separately."""
    import os
    os.environ["MFA_V6_BLOCK_R"] = str(s["R"])
    os.environ["MFA_V6_BLOCK_C"] = str(s["C"])
    os.environ["MFA_V6_EXEC_SG"] = str(s["SG"])

    mx.random.seed(0)
    B, H, Nq, Nkv, D = s["B"], s["H"], s["Nq"], s["Nkv"], s["D"]
    q = mx.random.normal((B,H,Nq,D)).astype(mx.float16)
    k = mx.random.normal((B,H,Nkv,D)).astype(mx.float16)
    v = mx.random.normal((B,H,Nkv,D)).astype(mx.float16)
    mx.async_eval(q,k,v); mx.synchronize()

    # 1. Full V6 forward (Python wrapper: 3 transposes + 3 contiguous + kernel + 1 transpose)
    full_med, full_std = time_op(lambda: v6_nax_forward(q, k, v, False))

    # 2. Transposes alone (3 transposes BHND→BNHD)
    def transposes_only():
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))
        return [q_t, k_t, v_t]
    tx_med, tx_std = time_op(transposes_only)

    # 3. Transposes + contiguous (the actual pre-kernel pipeline)
    def transposes_contig():
        q_t = mx.contiguous(mx.transpose(q, (0, 2, 1, 3)))
        k_t = mx.contiguous(mx.transpose(k, (0, 2, 1, 3)))
        v_t = mx.contiguous(mx.transpose(v, (0, 2, 1, 3)))
        return [q_t, k_t, v_t]
    txc_med, txc_std = time_op(transposes_contig)

    # 4. SDPA for comparison (no transposes — uses BHND directly)
    def sdpa():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(D))
    sdpa_med, sdpa_std = time_op(sdpa)

    return {
        "shape": s["name"],
        "full_v6_p50_ms": round(full_med, 4),
        "full_v6_stdev_ms": round(full_std, 4),
        "transposes_only_p50_ms": round(tx_med, 4),
        "transposes_only_stdev_ms": round(tx_std, 4),
        "transposes_contig_p50_ms": round(txc_med, 4),
        "transposes_contig_stdev_ms": round(txc_std, 4),
        "sdpa_p50_ms": round(sdpa_med, 4),
        "sdpa_stdev_ms": round(sdpa_std, 4),
        "v6_kernel_only_implied_ms": round(full_med - txc_med, 4),
        "transpose_overhead_pct": round(100.0 * txc_med / full_med, 2),
        "v6_vs_sdpa_ratio": round(full_med / sdpa_med, 4),
        "kernel_only_vs_sdpa_ratio": round((full_med - txc_med) / sdpa_med, 4),
    }


def measure_peak_memory(s):
    """Capture peak memory delta per call."""
    B, H, Nq, Nkv, D = s["B"], s["H"], s["Nq"], s["Nkv"], s["D"]
    mx.random.seed(0)
    q = mx.random.normal((B,H,Nq,D)).astype(mx.float16)
    k = mx.random.normal((B,H,Nkv,D)).astype(mx.float16)
    v = mx.random.normal((B,H,Nkv,D)).astype(mx.float16)
    mx.async_eval(q,k,v); mx.synchronize()

    # Q/K/V baseline allocation
    base = mx.metal.get_active_memory()

    # V6: peak memory during call
    mx.metal.reset_peak_memory()
    out, _ = v6_nax_forward(q, k, v, False)
    mx.async_eval(out); mx.synchronize()
    v6_peak = mx.metal.get_peak_memory()
    v6_delta = v6_peak - base
    del out; gc.collect(); mx.clear_cache()

    # SDPA: peak memory during call
    mx.metal.reset_peak_memory()
    out2 = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(D))
    mx.async_eval(out2); mx.synchronize()
    sdpa_peak = mx.metal.get_peak_memory()
    sdpa_delta = sdpa_peak - base

    return {
        "shape": s["name"],
        "qkv_base_bytes": int(base),
        "v6_peak_delta_bytes": int(v6_delta),
        "v6_peak_delta_mb": round(v6_delta / 1e6, 2),
        "sdpa_peak_delta_bytes": int(sdpa_delta),
        "sdpa_peak_delta_mb": round(sdpa_delta / 1e6, 2),
        "v6_extra_vs_sdpa_mb": round((v6_delta - sdpa_delta) / 1e6, 2),
    }


def main():
    print("V6 NAX vs SDPA — CPU-side profiling")
    print("=" * 90)
    results = {"timing": [], "memory": []}

    # 1. Timing breakdown
    print("\n--- TIMING BREAKDOWN ---")
    print(f"{'shape':<18} {'full_v6':>10} {'transp+ct':>10} {'kernel_only':>12} {'sdpa':>10} {'V6/SDPA':>9}")
    for s in SHAPES:
        r = measure_v6_breakdown(s)
        results["timing"].append(r)
        print(f"{r['shape']:<18} "
              f"{r['full_v6_p50_ms']:>10.3f} "
              f"{r['transposes_contig_p50_ms']:>10.3f} "
              f"{r['v6_kernel_only_implied_ms']:>12.3f} "
              f"{r['sdpa_p50_ms']:>10.3f} "
              f"{r['v6_vs_sdpa_ratio']:>9.3f}")

    # 2. Memory peak
    print("\n--- PEAK MEMORY DELTA ---")
    print(f"{'shape':<18} {'V6 peak Δ MB':>14} {'SDPA peak Δ MB':>16} {'V6 extra MB':>14}")
    for s in SHAPES:
        r = measure_peak_memory(s)
        results["memory"].append(r)
        print(f"{r['shape']:<18} {r['v6_peak_delta_mb']:>14.2f} "
              f"{r['sdpa_peak_delta_mb']:>16.2f} {r['v6_extra_vs_sdpa_mb']:>14.2f}")

    out_path = ROOT / "docs/v6-nax/profiling-counters.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
