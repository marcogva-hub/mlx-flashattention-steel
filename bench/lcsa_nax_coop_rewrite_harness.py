#!/usr/bin/env python3
"""Sprint B coop-rewrite Section D §4-strict perf sweep harness.

Compares V2 (cooperative-tensor inner-GEMM) vs V1 (per-thread FA-2 baseline)
vs MLX SDPA + float-bias on the 7 LCSA shapes + density sweep.

Methodology (§4 + Sprint C precedent):
- 7 production shapes + density sweep on lcsa_mid_seq8k
- A/B/A pattern: V2 → V1 → V2 (5 runs per direction)
- §4 cooldowns: 60s inter-shape, 90s inter-round, 180s initial
- Subprocess isolation per session (Artifact #1)
- Conditions sidecar per Artifact #5 sub-rule 5b
- Pre-flight correctness smoke gate (V2 vs V1 RMSE < 1e-3)

Output appended to docs/lcsa-nax/lcsa-nax-coop-rewrite-data.json.

Three-way SDPA comparison emitted per shape for ratio reporting.
"""
import argparse, json, math, platform, statistics, subprocess
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mlx.core as mx

import os
# Ensure each version is loaded freshly per timing block — set the env var
# before each call. Restore after.
from mlx_mfa.lcsa_nax import sparse_attention_nax, _bool_mask_to_float_bias


# 7 production shapes (Phase 0 inventory + niche representative).
SHAPES = [
    # name, B, Hq, Hk, qL, kL, D, density, BT, seed
    ("lcsa_small_seq4k",          1, 12, 12,  4096,  4096, 128, 0.24, 32, 1100),
    ("lcsa_small_seq4k_sparse",   1, 12, 12,  4096,  4096, 128, 0.07, 32, 1101),
    ("lcsa_mid_seq8k",            1,  8,  8,  8192,  8192, 128, 0.12, 32, 1102),
    ("lcsa_mid_seq8k_sparse",     1,  8,  8,  8192,  8192, 128, 0.03, 32, 1103),
    ("lcsa_large_seq16k",         1,  4,  4, 16384, 16384, 128, 0.12, 32, 1104),
    ("lcsa_large_seq16k_sparse",  1,  4,  4, 16384, 16384, 128, 0.03, 32, 1105),
    ("lcsa_mid_seq8k_very_sparse", 1, 8, 8,   8192,  8192, 128, 0.01, 32, 1106),
]
# Density sweep on lcsa_mid_seq8k for break-even characterization
DENSITY_SWEEP_BASE = (1, 8, 8, 8192, 8192, 128, 32, 1200)  # B, Hq, Hk, qL, kL, D, BT, seed
DENSITY_SWEEP_VALUES = [0.01, 0.03, 0.05, 0.10, 0.20, 0.50]

SMOKE_CFG = (1, 4, 4, 4096, 4096, 128, 0.10, 32, 9999)
SMOKE_RMSE_BAR = 1e-3


def _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed):
    mx.random.seed(seed)
    Q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(mx.float16)
    K = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    V = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(seed + 1)
    bm = (rng.random((NQ, NK)) < density).astype(np.bool_)
    for q in range(NQ):
        if not bm[q].any():
            bm[q, min(q, NK - 1)] = True
    mask = mx.array(bm)
    bias = _bool_mask_to_float_bias(mask, BT, qL, kL, mx.float16)
    mx.async_eval(Q, K, V, mask, bias); mx.synchronize()
    return Q, K, V, mask, bias, float(bm.mean())


def smoke_gate():
    """V1 vs V2 correctness gate. Exit non-zero if RMSE bar exceeded."""
    B, Hq, Hk, qL, kL, D, density, BT, seed = SMOKE_CFG
    Q, K, V, mask, _, _ = _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed)
    os.environ["MFA_LCSA_KERNEL_VERSION"] = "v1"
    O1 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O1); mx.synchronize()
    os.environ["MFA_LCSA_KERNEL_VERSION"] = "v2"
    O2 = sparse_attention_nax(Q, K, V, mask, block_tile=BT)
    mx.async_eval(O2); mx.synchronize()
    err = np.abs(np.array(O1.astype(mx.float32)) -
                 np.array(O2.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    maxerr = float(err.max())
    n_nan = int(np.isnan(np.array(O2.astype(mx.float32))).sum())
    n_inf = int(np.isinf(np.array(O2.astype(mx.float32))).sum())
    passed = rmse < SMOKE_RMSE_BAR and n_nan == 0 and n_inf == 0
    return passed, {"rmse": rmse, "maxerr": maxerr, "n_nan": n_nan, "n_inf": n_inf,
                    "bar": SMOKE_RMSE_BAR, "passed": passed}


def time_call(fn, n_runs):
    for _ in range(2):
        out = fn()
        mx.async_eval(out); mx.synchronize()
    times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.async_eval(out); mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times


def run_shape(name, B, Hq, Hk, qL, kL, D, density, BT, seed, n_runs):
    Q, K, V, mask, bias, d_actual = _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed)
    scale = 1.0 / math.sqrt(D)

    def call_v1():
        os.environ["MFA_LCSA_KERNEL_VERSION"] = "v1"
        return sparse_attention_nax(Q, K, V, mask, block_tile=BT, scale=scale)
    def call_v2():
        os.environ["MFA_LCSA_KERNEL_VERSION"] = "v2"
        return sparse_attention_nax(Q, K, V, mask, block_tile=BT, scale=scale)
    def call_sdpa():
        return mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=bias)

    # A/B/A pattern: V2 → V1 → V2
    v2_a = time_call(call_v2, n_runs)
    v1_times = time_call(call_v1, n_runs)
    v2_b = time_call(call_v2, n_runs)
    sdpa_times = time_call(call_sdpa, n_runs)

    v2_all = sorted(v2_a + v2_b)
    v2_med = statistics.median(v2_all)
    v1_med = statistics.median(v1_times)
    sdpa_med = statistics.median(sdpa_times)
    a_med = statistics.median(v2_a)
    b_med = statistics.median(v2_b)
    aba_drift_pct = abs(a_med - b_med) / a_med * 100 if a_med > 0 else 0
    return {
        "shape": name,
        "B": B, "Hq": Hq, "Hk": Hk, "qL": qL, "kL": kL, "D": D, "BT": BT,
        "density_target": density, "density_actual": d_actual,
        "v2_times_a_ms": v2_a, "v2_times_b_ms": v2_b,
        "v1_times_ms": v1_times, "sdpa_times_ms": sdpa_times,
        "v2_median_ms": v2_med, "v1_median_ms": v1_med, "sdpa_median_ms": sdpa_med,
        "v1_over_v2_ratio": v1_med / v2_med if v2_med > 0 else 0,
        "sdpa_over_v2_ratio": sdpa_med / v2_med if v2_med > 0 else 0,
        "sdpa_over_v1_ratio": sdpa_med / v1_med if v1_med > 0 else 0,
        "aba_drift_pct": aba_drift_pct,
    }


def capture_conditions():
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "platform": platform.platform()}
    for n, c in [("sw_vers", ["sw_vers"]), ("uptime", ["uptime"]),
                 ("uname", ["uname", "-a"]),
                 ("boottime", ["sysctl", "-n", "kern.boottime"])]:
        try:
            out[n] = subprocess.run(c, check=True, capture_output=True,
                                    text=True, timeout=5).stdout.strip()
        except Exception as e:
            out[f"{n}_error"] = str(e)
    try:
        out["mlx_version"] = mx.__version__
    except Exception:
        out["mlx_version"] = "unknown"
    try:
        from mlx_mfa import __version__ as mfa_ver
        out["mlx_mfa_version"] = mfa_ver
    except Exception:
        out["mlx_mfa_version"] = "unknown"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--output",
                    default="docs/lcsa-nax/lcsa-nax-coop-rewrite-data.json")
    ap.add_argument("--cooldown-inter-round", type=float, default=90.0)
    ap.add_argument("--cooldown-inter-shape", type=float, default=60.0)
    ap.add_argument("--cooldown-initial", type=float, default=180.0)
    ap.add_argument("--runs-per-direction", type=int, default=5)
    ap.add_argument("--skip-initial-cooldown", action="store_true")
    args = ap.parse_args()

    print(f"[coop §4 harness] session={args.session_id}", flush=True)

    print("[coop §4 harness] correctness smoke (V1 vs V2)...", flush=True)
    ok, diag = smoke_gate()
    print(f"  smoke: rmse={diag['rmse']:.4e} maxerr={diag['maxerr']:.4e} "
          f"NaN={diag['n_nan']} Inf={diag['n_inf']} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("[coop §4 harness] STATUS: SMOKE_FAILED", file=sys.stderr, flush=True)
        sys.exit(2)

    if not args.skip_initial_cooldown:
        print(f"[coop §4 harness] initial cooldown {args.cooldown_initial}s",
              flush=True)
        time.sleep(args.cooldown_initial)

    record = {
        "session_id": args.session_id,
        "phase": "B-coop-rewrite §4-strict (V2 vs V1 vs SDPA+bias)",
        "cooldowns": {
            "initial_s": args.cooldown_initial,
            "inter_shape_s": args.cooldown_inter_shape,
            "inter_round_s": args.cooldown_inter_round,
        },
        "runs_per_direction": args.runs_per_direction,
        "smoke_gate": diag,
        "conditions": capture_conditions(),
        "production_results": [],
        "density_sweep_results": [],
    }

    # Production shapes
    print("\n[coop §4 harness] production shapes A/B/A V2→V1→V2", flush=True)
    for i, spec in enumerate(SHAPES):
        name = spec[0]
        try:
            res = run_shape(*spec, n_runs=args.runs_per_direction)
        except Exception as e:
            res = {"shape": name, "error": str(e)[:300]}
        record["production_results"].append(res)
        if "error" in res:
            print(f"  {name:<32} ERROR: {res['error'][:80]}", flush=True)
        else:
            print(f"  {name:<32} d={res['density_actual']:.3f} "
                  f"V2={res['v2_median_ms']:>7.2f}ms "
                  f"V1={res['v1_median_ms']:>7.2f}ms "
                  f"SDPA={res['sdpa_median_ms']:>7.2f}ms "
                  f"V1/V2={res['v1_over_v2_ratio']:>5.2f}× "
                  f"SDPA/V2={res['sdpa_over_v2_ratio']:>5.2f}× "
                  f"drift={res['aba_drift_pct']:>4.1f}%", flush=True)
        if i < len(SHAPES) - 1:
            time.sleep(args.cooldown_inter_shape)

    # Density sweep
    time.sleep(args.cooldown_inter_round)
    print("\n[coop §4 harness] density sweep on lcsa_mid_seq8k", flush=True)
    B, Hq, Hk, qL, kL, D, BT, seed_base = DENSITY_SWEEP_BASE
    for i, d in enumerate(DENSITY_SWEEP_VALUES):
        try:
            res = run_shape(f"lcsa_mid_seq8k_d{int(d*1000):03d}",
                             B, Hq, Hk, qL, kL, D, d, BT,
                             seed_base + int(d * 1000),
                             n_runs=args.runs_per_direction)
        except Exception as e:
            res = {"shape": f"density_{d}", "error": str(e)[:300]}
        record["density_sweep_results"].append(res)
        if "error" in res:
            print(f"  d={d:.2f} ERROR: {res['error'][:80]}", flush=True)
        else:
            print(f"  d={d:.2f} d_act={res['density_actual']:.3f} "
                  f"V2={res['v2_median_ms']:>7.2f}ms "
                  f"V1={res['v1_median_ms']:>7.2f}ms "
                  f"SDPA={res['sdpa_median_ms']:>7.2f}ms "
                  f"V1/V2={res['v1_over_v2_ratio']:>5.2f}× "
                  f"SDPA/V2={res['sdpa_over_v2_ratio']:>5.2f}×", flush=True)
        if i < len(DENSITY_SWEEP_VALUES) - 1:
            time.sleep(args.cooldown_inter_shape)

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))
    print(f"\n[coop §4 harness] session '{args.session_id}' → {p}", flush=True)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
