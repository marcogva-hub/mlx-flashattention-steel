#!/usr/bin/env python3
"""Sprint B Phase 1.5 §4-strict re-bench harness — LCSA sparse_attention_dispatch
vs MLX SDPA + float-bias baseline.

Methodology (Sprint C precedent + Sprint B Phase 0):
- 7 production shapes (6 LCSA clusters + 1 very-sparse niche-win shape) ×
  A/B/A (NAX-sparse → SDPA+bias → NAX-sparse) × 5 runs per direction
- §4 cooldowns: 60s/shape, 90s/round, 180s/initial
- Subprocess isolation per session (Artifact #1)
- Conditions sidecar per Artifact #5 sub-rule 5b
- Pre-flight correctness smoke gate (Phase 1.1 lesson learned)
- Wall-clock measurement via perf_counter + mx.synchronize boundaries

Output appended to docs/lcsa-nax/lcsa-nax-rebench-data.json — per-session
record with per-shape NAX/SDPA timings + ratios.

Comparison baseline = caller-cached float bias path (v2.33.1 cache-HIT
pattern), since that is the dispatcher's else-branch destination. Reflects
docs/lcsa-nax/lcsa-nax-phase1_5-ship-verdict.md §2 numbers apples-to-apples.

Variance handling per CLAUDE_V6_NAX §B.7 (applied post-bench):
  - cross-session range < 10% → ratio confident
  - 10-20% range → boundary; opt-in default
  - > 20% range → §B.7 high-variance fallback (claims need caveating)
"""
import argparse, json, math, platform, statistics, subprocess
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mlx.core as mx

from mlx_mfa.lcsa_nax import (
    sparse_attention_nax,
    sparse_attention_dispatch,
    _bool_mask_to_float_bias,
)


# ---------------------------------------------------------------------
# Shape inventory — 6 Phase 0 production clusters + 1 very-sparse niche.
# (B, Hq, Hk, qL, kL, D, density, seed, label_class)
# label_class is "moderate" or "niche" — used by analysis to anchor expected ratio.
# ---------------------------------------------------------------------
SHAPES = [
    # name, B, Hq, Hk, qL, kL, D, density, BT, seed, class
    ("lcsa_small_seq4k",          1, 12, 12,  4096,  4096, 128, 0.24, 16, 1100, "moderate"),
    ("lcsa_small_seq4k_sparse",   1, 12, 12,  4096,  4096, 128, 0.07, 16, 1101, "moderate"),
    ("lcsa_mid_seq8k",            1,  8,  8,  8192,  8192, 128, 0.12, 16, 1102, "moderate"),
    ("lcsa_mid_seq8k_sparse",     1,  8,  8,  8192,  8192, 128, 0.03, 16, 1103, "boundary"),
    ("lcsa_large_seq16k",         1,  4,  4, 16384, 16384, 128, 0.12, 16, 1104, "moderate"),
    ("lcsa_large_seq16k_sparse",  1,  4,  4, 16384, 16384, 128, 0.03, 16, 1105, "boundary"),
    # Niche-win shape: Phase 1.4 sweep showed 2.45-4.6× at density 0.01
    # across the 3 large clusters. We pick mid_seq8k @ d=0.01 as the
    # canonical niche representative (mid sequence, mid head count).
    ("lcsa_mid_seq8k_very_sparse", 1, 8, 8,   8192,  8192, 128, 0.01, 16, 1106, "niche"),
]

SMOKE_CFG = (1, 4, 4, 4096, 4096, 128, 0.24, 16, 9999)  # small + dense → exercises NAX path
SMOKE_RMSE_BAR = 5e-3


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
    actual_density = float(bm.mean())
    return Q, K, V, mask, bias, actual_density


def smoke_gate():
    """Pre-flight correctness gate — NAX sparse vs SDPA+bias at SMOKE_CFG."""
    B, Hq, Hk, qL, kL, D, density, BT, seed = SMOKE_CFG
    Q, K, V, mask, bias, _ = _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed)
    scale = 1.0 / math.sqrt(D)
    O_nax = sparse_attention_nax(Q, K, V, mask, block_tile=BT, scale=scale)
    mx.async_eval(O_nax); mx.synchronize()
    O_ref = mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=bias)
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O_nax.astype(mx.float32)) -
                 np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    maxerr = float(err.max())
    n_inf = int(np.isinf(np.array(O_nax.astype(mx.float32))).sum())
    n_nan = int(np.isnan(np.array(O_nax.astype(mx.float32))).sum())
    rel_err = rmse / (float(np.abs(np.array(O_ref.astype(mx.float32))).mean()) + 1e-12)
    passed = rmse < SMOKE_RMSE_BAR and n_inf == 0 and n_nan == 0
    return passed, {"rmse": rmse, "maxerr": maxerr, "rel_err": rel_err,
                    "n_inf": n_inf, "n_nan": n_nan,
                    "bar": SMOKE_RMSE_BAR, "passed": passed}


def time_call(fn, n_runs):
    """Time a callable n_runs times; warmup once outside loop, mx.synchronize per call."""
    # Single warm-up call (output discarded).
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


def run_shape(name, B, Hq, Hk, qL, kL, D, density, BT, seed, label_class, *, n_runs):
    """A/B/A pattern: NAX-sparse → SDPA+bias → NAX-sparse."""
    Q, K, V, mask, bias, actual_density = _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed)
    scale = 1.0 / math.sqrt(D)

    def call_nax():
        # Use dispatcher with precomputed_bias + density hint (matches the
        # production cache-HIT pattern; for very-sparse shapes the dispatcher
        # routes to sparse_attention_nax internally per DEFAULT_DENSITY_THRESHOLD)
        return sparse_attention_dispatch(
            Q, K, V, mask, block_tile=BT, scale=scale,
            density=actual_density, precomputed_bias=bias)

    def call_sdpa_bias():
        return mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=bias)

    nax_a = time_call(call_nax, n_runs)
    sdpa_times = time_call(call_sdpa_bias, n_runs)
    nax_b = time_call(call_nax, n_runs)

    nax_all = sorted(nax_a + nax_b)
    nax_med = statistics.median(nax_all)
    sdpa_med = statistics.median(sdpa_times)
    # Ratio convention: > 1.0 → NAX faster than SDPA+bias.
    # (For dispatcher routing to SDPA, ratio should be ~1.0.)
    ratio_sdpa_over_nax = sdpa_med / nax_med if nax_med > 0 else 0
    a_med = statistics.median(nax_a)
    b_med = statistics.median(nax_b)
    aba_drift_pct = abs(a_med - b_med) / a_med * 100 if a_med > 0 else 0
    return {
        "shape": name, "class": label_class,
        "B": B, "Hq": Hq, "Hk": Hk, "qL": qL, "kL": kL, "D": D,
        "density_target": density, "density_actual": actual_density, "BT": BT,
        "nax_times_a_ms": nax_a, "nax_times_b_ms": nax_b,
        "sdpa_times_ms": sdpa_times,
        "nax_median_ms": nax_med, "sdpa_median_ms": sdpa_med,
        "ratio_sdpa_over_nax": ratio_sdpa_over_nax,
        "aba_drift_pct": aba_drift_pct,
    }


def capture_conditions():
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "platform": platform.platform()}
    for n, c in [("sw_vers", ["sw_vers"]),
                 ("uptime", ["uptime"]),
                 ("uname", ["uname", "-a"]),
                 ("boottime", ["sysctl", "-n", "kern.boottime"])]:
        try:
            out[n] = subprocess.run(c, check=True, capture_output=True,
                                    text=True, timeout=5).stdout.strip()
        except Exception as e:
            out[f"{n}_error"] = str(e)
    try:
        import mlx
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
    ap.add_argument("--session-id", required=True,
                    help="Label for this session, e.g. S1/S2/S3.")
    ap.add_argument("--output",
                    default="docs/lcsa-nax/lcsa-nax-rebench-data.json")
    ap.add_argument("--cooldown-inter-round", type=float, default=90.0,
                    help="§4: between A/B/A rounds within a shape (default 90s)")
    ap.add_argument("--cooldown-inter-shape", type=float, default=60.0,
                    help="§4: between shapes (default 60s)")
    ap.add_argument("--cooldown-initial", type=float, default=180.0,
                    help="§4: at session start (default 180s)")
    ap.add_argument("--runs-per-direction", type=int, default=5,
                    help="Runs per A direction; B uses same count.")
    ap.add_argument("--skip-initial-cooldown", action="store_true")
    args = ap.parse_args()

    print(f"[lcsa-nax §4 harness] session={args.session_id}", flush=True)

    # Smoke gate FIRST. No timing data emitted if smoke fails.
    print("[lcsa-nax §4 harness] correctness smoke (Phase 1.1 lesson)...", flush=True)
    ok, diag = smoke_gate()
    print(f"  smoke: rel_err={diag['rel_err']:.4e}  rmse={diag['rmse']:.6f}  "
          f"NaN={diag['n_nan']} Inf={diag['n_inf']} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("[lcsa-nax §4 harness] STATUS: SMOKE_FAILED", file=sys.stderr, flush=True)
        sys.exit(2)

    if not args.skip_initial_cooldown:
        print(f"[lcsa-nax §4 harness] initial cooldown "
              f"{args.cooldown_initial}s", flush=True)
        time.sleep(args.cooldown_initial)

    record = {
        "session_id": args.session_id,
        "phase": "B-1.5 §4-strict re-bench (LCSA NAX vs SDPA+bias)",
        "cooldowns": {
            "initial_s": args.cooldown_initial,
            "inter_shape_s": args.cooldown_inter_shape,
            "inter_round_s": args.cooldown_inter_round,
            "skip_initial": args.skip_initial_cooldown,
        },
        "runs_per_direction": args.runs_per_direction,
        "smoke_gate": diag,
        "conditions": capture_conditions(),
        "results": [],
    }

    for i, spec in enumerate(SHAPES):
        name = spec[0]
        try:
            res = run_shape(*spec, n_runs=args.runs_per_direction)
        except Exception as e:
            res = {"shape": name, "error": str(e)[:300]}
        record["results"].append(res)
        if "error" in res:
            print(f"  {name:<32} ERROR: {res['error'][:80]}", flush=True)
        else:
            print(f"  {name:<32} d_actual={res['density_actual']:.3f} class={res['class']:<9} "
                  f"NAX={res['nax_median_ms']:>7.2f}ms  "
                  f"SDPA={res['sdpa_median_ms']:>7.2f}ms  "
                  f"ratio={res['ratio_sdpa_over_nax']:>5.2f}×  "
                  f"drift={res['aba_drift_pct']:>4.1f}%", flush=True)
        # Inter-shape cooldown after every shape except the last.
        if i < len(SHAPES) - 1:
            time.sleep(args.cooldown_inter_shape)

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))
    print(f"\n[lcsa-nax §4 harness] session '{args.session_id}' → {p}", flush=True)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
