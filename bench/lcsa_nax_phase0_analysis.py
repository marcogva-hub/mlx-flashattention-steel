#!/usr/bin/env python3
"""Sprint B Phase 0 — analyze 3-session baseline + compute theoretical bound + ROI.

Reads docs/lcsa-nax/lcsa-nax-phase0-baseline-data.json (raw bench data),
computes per-shape cross-session medians, theoretical NAX bound (compute
+ bandwidth bounded), and ROI ranking.

Output: docs/lcsa-nax/lcsa-nax-phase0-analysis.json
"""
import json, statistics
from pathlib import Path

# NAX calibration from Sprint C Phase 1.5 measurements:
# - Apple advertised peak: 38 TFLOPS FP16
# - Sustained measured: 24-33 TF on M-skewed shapes (Phase 1.5)
# - Median dominant: 25 TF — use as representative sustained
NAX_TFLOPS_SUSTAINED = 25.0  # FP16, calibrated from Sprint C
NAX_TFLOPS_OPTIMISTIC = 38.0  # Apple peak; never observed in production
HBM_BANDWIDTH_GB = 410.0  # M5 Max system memory bandwidth, Apple spec


def theoretical_bounds(B, H, N, D, density, dtype_bytes=2):
    """Compute compute + bandwidth bounds for a sparse attention shape.

    Effective FLOPs accounting for sparsity. Bandwidth bound assumes
    Q/K/V/O reads + output write -- block-skip doesn't reduce bandwidth
    much (still need to read all unique Q tokens once; sparsity reduces
    K/V loads but Q is fully traversed).
    """
    flops_dense = 2.0 * B * H * N * N * D
    flops_eff = flops_dense * density
    compute_bound_ms_sustained = flops_eff / (NAX_TFLOPS_SUSTAINED * 1e12) * 1e3
    compute_bound_ms_optimistic = flops_eff / (NAX_TFLOPS_OPTIMISTIC * 1e12) * 1e3
    # Bandwidth: with block-skip, K/V reads scale with density. Q + O read/write
    # always. Simplification: total bytes ≈ B*H*N*D*2 * (2 + 2*density)
    bytes_bw = B * H * N * D * dtype_bytes * (2 + 2 * density)
    bw_bound_ms = bytes_bw / (HBM_BANDWIDTH_GB * 1e9) * 1e3
    theoretical_min_ms = max(compute_bound_ms_sustained, bw_bound_ms)
    return {
        "flops_dense": flops_dense,
        "flops_effective_sparse": flops_eff,
        "compute_bound_ms_sustained": compute_bound_ms_sustained,
        "compute_bound_ms_optimistic": compute_bound_ms_optimistic,
        "bw_bound_ms": bw_bound_ms,
        "theoretical_min_ms": theoretical_min_ms,
        "density_used": density,
    }


def analyze(sessions):
    """Compute cross-session medians + theoretical bound + headroom per shape."""
    per_shape = {}
    for sess in sessions:
        for r in sess["results"]:
            if "error" in r:
                continue
            label = r["shape"]
            if label not in per_shape:
                per_shape[label] = {
                    "B": r["B"], "H": r["H"], "N": r["N"], "D": r["D"],
                    "density": r["density"],
                    "sparsity_label": r["sparsity_label"],
                    "window_size_tokens": r["window_size_tokens"],
                    "mfa_medians_per_session": [],
                    "sdpa_medians_per_session": [],
                    "ratios_per_session": [],
                }
            per_shape[label]["mfa_medians_per_session"].append(r["mfa_median_ms"])
            per_shape[label]["sdpa_medians_per_session"].append(r["sdpa_median_ms"])
            per_shape[label]["ratios_per_session"].append(r["mfa_vs_sdpa_ratio"])

    for label, s in per_shape.items():
        s["mfa_median_3s"] = statistics.median(s["mfa_medians_per_session"])
        s["sdpa_median_3s"] = statistics.median(s["sdpa_medians_per_session"])
        rng = (max(s["mfa_medians_per_session"]) - min(s["mfa_medians_per_session"])) \
              / s["mfa_median_3s"] * 100 if s["mfa_median_3s"] > 0 else 0
        s["mfa_range_pct"] = rng
        # Better baseline = whichever is faster (current production-best)
        s["best_baseline_ms"] = min(s["mfa_median_3s"], s["sdpa_median_3s"])
        s["best_baseline_path"] = "MLX SDPA+float-bias" if s["sdpa_median_3s"] < s["mfa_median_3s"] else "flash_attention_sparse"
        bounds = theoretical_bounds(s["B"], s["H"], s["N"], s["D"], s["density"])
        s.update(bounds)
        s["headroom_vs_baseline"] = (
            s["best_baseline_ms"] / s["theoretical_min_ms"]
            if s["theoretical_min_ms"] > 0 else 0
        )
        s["potential_ms_saved"] = s["best_baseline_ms"] - s["theoretical_min_ms"]
        s["potential_pct_saved"] = (
            s["potential_ms_saved"] / s["best_baseline_ms"] * 100
            if s["best_baseline_ms"] > 0 else 0
        )

    return per_shape


def main():
    path = Path("docs/lcsa-nax/lcsa-nax-phase0-baseline-data.json")
    sessions = json.loads(path.read_text())
    print(f"sessions: {len(sessions)}")
    summary = analyze(sessions)

    print(f"\n{'shape':<32} {'N':>5} {'dens':>5}  "
          f"{'MFA':>7} {'SDPA':>7} {'best':>7}  "
          f"{'bound':>7} {'headroom':>9} {'%saved':>7}")
    print('-' * 105)
    for label, s in summary.items():
        print(f"{label:<32} {s['N']:>5} {s['density']:>5.2f}  "
              f"{s['mfa_median_3s']:>6.2f}ms {s['sdpa_median_3s']:>6.2f}ms "
              f"{s['best_baseline_ms']:>6.2f}ms  "
              f"{s['theoretical_min_ms']:>6.2f}ms {s['headroom_vs_baseline']:>7.2f}× "
              f"{s['potential_pct_saved']:>6.1f}%")

    median_headroom = statistics.median(
        s["headroom_vs_baseline"] for s in summary.values())
    max_headroom = max(s["headroom_vs_baseline"] for s in summary.values())
    print(f"\nMedian headroom (baseline / theoretical-min): {median_headroom:.2f}×")
    print(f"Max headroom: {max_headroom:.2f}×")

    out = {"summary": summary,
           "median_headroom": median_headroom,
           "max_headroom": max_headroom,
           "nax_tflops_calibration": NAX_TFLOPS_SUSTAINED,
           "hbm_bandwidth_gbs": HBM_BANDWIDTH_GB}
    Path("docs/lcsa-nax/lcsa-nax-phase0-analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nanalysis -> docs/lcsa-nax/lcsa-nax-phase0-analysis.json")


if __name__ == "__main__":
    main()
