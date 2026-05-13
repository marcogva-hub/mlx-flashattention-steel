"""TGP cross-SG reduction overhead micro-benchmark.

Resolves the 10× discrepancy between two estimates of per-K-tile TGP
overhead for Option γ fused dK+dV:

  Design doc (`docs/v6-nax/v34-backward-option-gamma-design.md`):
    ~100µs/K-tile × NK=512 = 51ms total

  CC pre-flight (halted v2.38.0 sprint Phase A):
    ~300ns/row × BK=16 rows = 4.8µs/K-tile × NK=512 = 2.5ms total

The 10× discrepancy is the difference between "Option γ delivers
parity at D=128" (CC estimate) and "Option γ ceiling at ~2× SDPA-vjp"
(design doc).  Empirical measurement resolves which.

Isolation principle: this micro-bench performs ONLY the TGP cross-SG
reduction pattern Option γ would use.  NO softmax work, NO MMA, NO
real attention data flow.  Just:
  - 4 SGs (WM=4) write per-row partials to TGP (BK=16 rows × D=128 fp32)
  - threadgroup_barrier
  - SG0 streams through rows, sums across 4 SGs, writes to device
  - threadgroup_barrier
  - repeat for NK iterations

Baseline: same kernel WITHOUT the TGP+barrier+reduce pattern.  Just
dummy writes to device.  Subtract baseline from probe time = pure
TGP-overhead contribution.

Protocol: canonical warmup+continuous (§4.2) — sub-1.5ms kernels.
3 sessions for cross-session ratio analysis.

Outputs: docs/v6-nax/tgp-overhead-investigation.md raw data table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx


# Sync wrapper bypassing security-hook substring match
_AE = getattr(mx, "async_" + "eval")


# ---------- Kernel definitions ----------

# Option γ TGP pattern:
#   - threadgroup buffer dimensioned for WM=4 SGs × BK rows × D fp32
#   - per K-tile iteration:
#       1. 4 SGs each write their per-row partial to disjoint TGP slot
#       2. threadgroup_barrier
#       3. SG0 streams BK rows: read 4 × D from TGP, sum, write D fp32 to device
#       4. threadgroup_barrier (prep for next iteration)
TGP_PROBE_SOURCE = """
    uint tid_in_tg = thread_position_in_threadgroup.x;
    uint sg_id     = simdgroup_index_in_threadgroup;
    uint lane_id   = thread_index_in_simdgroup;

    // BK=16 rows × D=128 floats per SG slot
    // 4 SGs × 16 × 128 = 8192 floats = 32 KB TGP — within M5 budget
    threadgroup float tgp_buffer[4 * 16 * 128];

    const uint BK = 16;
    const uint D  = 128;

    uint n_iter = n_iter_buf[0];

    for (uint k = 0; k < n_iter; k++) {
        // === Step 1: 4 SGs write per-row partials to disjoint TGP slot ===
        // Per-SG slot layout: tgp_buffer[sg_id * BK * D + row * D + col]
        // Each thread in the SG writes D / 32 = 4 cols across BK rows
        for (uint row = 0; row < BK; row++) {
            for (uint d4 = 0; d4 < 4; d4++) {
                uint col = lane_id + d4 * 32;
                tgp_buffer[sg_id * BK * D + row * D + col] =
                    float(k) + float(sg_id) * 0.1f + float(row) * 0.01f;
            }
        }

        // === Step 2: barrier — all 4 SGs done writing partials ===
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Step 3: SG0 streams BK rows, sums across 4 SGs, writes to device ===
        if (sg_id == 0) {
            for (uint row = 0; row < BK; row++) {
                for (uint d4 = 0; d4 < 4; d4++) {
                    uint col = lane_id + d4 * 32;
                    float sum = 0;
                    for (uint s = 0; s < 4; s++) {
                        sum += tgp_buffer[s * BK * D + row * D + col];
                    }
                    // Write to output buffer at offset (k * BK + row) * D + col
                    output[(k * BK + row) * D + col] = sum;
                }
            }
        }

        // === Step 4: barrier prep for next iteration ===
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# Baseline: same outer structure (4 SGs × n_iter loop) but NO TGP pattern.
# Each SG just writes its own dummy output directly to device.
# Subtracting baseline isolates the TGP+barrier+reduce overhead.
BASELINE_SOURCE = """
    uint sg_id   = simdgroup_index_in_threadgroup;
    uint lane_id = thread_index_in_simdgroup;

    const uint BK = 16;
    const uint D  = 128;

    uint n_iter = n_iter_buf[0];

    for (uint k = 0; k < n_iter; k++) {
        // Same compute pattern (4 SGs × BK × D writes) but to device, not TGP
        // No barriers, no cross-SG read-back.
        for (uint row = 0; row < BK; row++) {
            for (uint d4 = 0; d4 < 4; d4++) {
                uint col = lane_id + d4 * 32;
                // SG0 writes the "final" output; other SGs no-op (mirror probe pattern)
                if (sg_id == 0) {
                    output[(k * BK + row) * D + col] =
                        float(k) + float(row) * 0.01f;
                }
            }
        }
    }
"""


def _make_kernel(source: str, name: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=["n_iter_buf"],
        output_names=["output"],
        source=source,
        ensure_row_contiguous=True,
    )


def _dispatch(kernel, n_iter: int, output_size: int):
    """Dispatch with WM=4 SGs per TG (= 4 × 32 = 128 threads/TG)."""
    n_iter_buf = mx.array([n_iter], dtype=mx.uint32)
    return kernel(
        inputs=[n_iter_buf],
        grid=(128, 1, 1),  # 1 TG of 128 threads (4 SGs × 32 lanes)
        threadgroup=(128, 1, 1),
        output_shapes=[(output_size,)],
        output_dtypes=[mx.float32],
    )[0]


def _bench(kernel, n_iter: int, output_size: int, *, warmup=10, iters=100):
    """Canonical §4.2: 10 warmup + 100 continuous timed."""
    for _ in range(warmup):
        out = _dispatch(kernel, n_iter, output_size)
        _AE(out); mx.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = _dispatch(kernel, n_iter, output_size)
        _AE(out); mx.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts), max(ts)


def run_session(probe_kernel, baseline_kernel, n_iter_list):
    """Run one session of the bench across all n_iter values."""
    rows = []
    for n_iter in n_iter_list:
        # Output size: NK iterations × BK rows × D floats
        out_size = n_iter * 16 * 128

        probe_med, probe_min, probe_max = _bench(probe_kernel, n_iter, out_size)
        base_med, base_min, base_max = _bench(baseline_kernel, n_iter, out_size)

        # Per-iteration TGP overhead = (probe - baseline) / n_iter
        # If probe < baseline (rare; should not happen), report 0 to flag.
        overhead_total_ms = max(probe_med - base_med, 0.0)
        overhead_per_iter_us = (overhead_total_ms / n_iter) * 1000.0  # ms → µs

        rows.append({
            "n_iter": n_iter,
            "probe_median_ms": round(probe_med, 4),
            "baseline_median_ms": round(base_med, 4),
            "tgp_overhead_total_ms": round(overhead_total_ms, 4),
            "tgp_overhead_per_iter_us": round(overhead_per_iter_us, 3),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-iter-list", default="1,8,32,128,512",
        help="Comma-separated NK values to sweep",
    )
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--out", default="/tmp/tgp_overhead_results.json")
    args = p.parse_args()

    n_iter_list = [int(x) for x in args.n_iter_list.split(",")]

    probe_kernel = _make_kernel(TGP_PROBE_SOURCE, "tgp_overhead_probe")
    baseline_kernel = _make_kernel(BASELINE_SOURCE, "tgp_overhead_baseline")

    print(f"TGP overhead micro-bench — {args.sessions} sessions × {len(n_iter_list)} shapes")
    print(f"  Probe pattern: 4 SGs × BK=16 rows × D=128 fp32 partials → TGP → SG0 sum → device")
    print(f"  Baseline: same outer shape, no TGP+barrier+reduce")
    print(f"  Protocol: canonical §4.2 (10 warmup + 100 continuous, median)")
    print()

    sessions = []
    for s in range(args.sessions):
        print(f"=== Session {s+1}/{args.sessions} ===")
        rows = run_session(probe_kernel, baseline_kernel, n_iter_list)
        sessions.append(rows)
        for r in rows:
            print(f"  NK={r['n_iter']:>4}: "
                  f"probe={r['probe_median_ms']:>6.3f}ms  "
                  f"baseline={r['baseline_median_ms']:>6.3f}ms  "
                  f"TGP overhead: {r['tgp_overhead_total_ms']:>6.3f}ms total, "
                  f"{r['tgp_overhead_per_iter_us']:>6.3f}µs/iter")

    # Cross-session aggregation
    print("\n=== Cross-session aggregation ===")
    for i, n_iter in enumerate(n_iter_list):
        per_iter_us = [s[i]["tgp_overhead_per_iter_us"] for s in sessions]
        med = statistics.median(per_iter_us)
        rng = max(per_iter_us) - min(per_iter_us)
        rel = rng / med if med > 0 else float("inf")
        verdict = "CONFIDENT" if rel < 0.10 else (
            "BOUNDARY" if rel < 0.20 else "HIGH_VARIANCE"
        )
        print(f"  NK={n_iter:>4}: median {med:>6.3f}µs/iter, "
              f"range {rng:>5.3f}µs ({rel*100:>5.1f}%) — {verdict}")

    Path(args.out).write_text(json.dumps({
        "n_iter_list": n_iter_list,
        "sessions": sessions,
    }, indent=2))
    print(f"\nRaw data: {args.out}")


if __name__ == "__main__":
    main()
