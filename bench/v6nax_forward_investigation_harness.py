#!/usr/bin/env python3
"""V6NAX forward investigation — unified §4-strict harness for 5 hypotheses.

Probes each hypothesis (A-E) via MFA_V6_* env-var toggles on the existing
V6NAX production code rather than building variant source-gen functions.

Each probe is a single-session §4-strict bench (§4 cooldowns 180/60/90s
A/B/A pattern). Pattern: ALT (probe config) → BASELINE (V6NAX stock) → ALT
to characterize the mechanism's contribution to the V6NAX baseline gain.

Hypotheses:
  - A: TGP occupancy via MFA_V6_EXEC_SG (1, 2, 4, 8)
  - B+C bundled: V6NAX vs predecessor via MFA_V6_USE_NAX (0/1)
  - D: register pressure via MFA_V6_BLOCK_R (64 vs default 32)
  - E: Apple defaults — bundled with B+C measurement

Output: docs/v6-nax/v6nax-forward-investigation-data.json
"""
import argparse, json, math, os, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mlx.core as mx

# 4 shapes per inventory DI4
SHAPES = [
    # name, B, Hq, Hk, qL, kL, D, seed
    ("v6nax_small_d64",  1,  8,  8,  1024,  1024,  64, 9001),
    ("v6nax_small_d128", 1,  8,  8,  1024,  1024, 128, 9002),
    ("v6nax_mid_d128",   1, 16, 16,  4096,  4096, 128, 9003),
    ("v6nax_large_d128", 1, 16, 16,  8192,  8192, 128, 9004),
]

# Smoke shape: small enough for fast smoke gate
SMOKE_CFG = (1, 4, 4, 1024, 1024, 128, 99999)
SMOKE_RMSE_BAR = 5e-3


def _build_inputs(B, Hq, Hk, qL, kL, D, seed):
    mx.random.seed(seed)
    Q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(mx.float16)
    K = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    V = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    mx.async_eval(Q, K, V); mx.synchronize()
    return Q, K, V


def smoke_gate():
    """Output-sanity axis-1 gate before any timing."""
    B, Hq, Hk, qL, kL, D, seed = SMOKE_CFG
    Q, K, V = _build_inputs(B, Hq, Hk, qL, kL, D, seed)
    scale = 1.0 / math.sqrt(D)
    # V6NAX attention via mlx flash_attention (uses V6NAX dispatch when env set)
    from mlx_mfa._ext import v6_nax_forward
    O, _ = v6_nax_forward(Q, K, V, False)
    mx.async_eval(O); mx.synchronize()
    # Reference: dense SDPA
    O_ref = mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale)
    mx.async_eval(O_ref); mx.synchronize()
    err = np.abs(np.array(O.astype(mx.float32)) - np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    maxerr = float(err.max())
    n_nan = int(np.isnan(np.array(O.astype(mx.float32))).sum())
    n_inf = int(np.isinf(np.array(O.astype(mx.float32))).sum())
    passed = rmse < SMOKE_RMSE_BAR and n_nan == 0 and n_inf == 0
    return passed, {"rmse": rmse, "maxerr": maxerr, "n_nan": n_nan, "n_inf": n_inf,
                    "bar": SMOKE_RMSE_BAR, "passed": passed}


def time_call(fn, n_runs):
    # 2 warmups
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


def run_shape_aba(name, B, Hq, Hk, qL, kL, D, seed, n_runs, cooldown_inter_round):
    """A/B/A pattern: ALT → BASELINE → ALT.

    The ALT and BASELINE env config is set OUTSIDE this function by the
    caller; this function just times the current env state.
    Returns 3 lists of timings: alt_1, baseline, alt_2.
    """
    Q, K, V = _build_inputs(B, Hq, Hk, qL, kL, D, seed)
    scale = 1.0 / math.sqrt(D)
    from mlx_mfa._ext import v6_nax_forward

    def call():
        return v6_nax_forward(Q, K, V, False)[0]

    return time_call(call, n_runs)


def run_probe(probe_name, baseline_env_overrides, alt_env_overrides, shapes,
              n_runs, cooldown_inter_round, cooldown_inter_shape):
    """Run a single hypothesis probe across all shapes.

    A/B/A pattern: ALT → BASELINE → ALT.
    Env vars are set/unset between rounds. Each round is 5 runs.
    """
    results = []
    for i, spec in enumerate(shapes):
        name = spec[0]
        # ALT round 1
        for k, v in alt_env_overrides.items():
            os.environ[k] = v
        try:
            alt_a = run_shape_aba(*spec, n_runs=n_runs,
                                   cooldown_inter_round=cooldown_inter_round)
        except Exception as e:
            results.append({"shape": name, "error_alt_a": str(e)[:300]})
            continue
        time.sleep(cooldown_inter_round)

        # BASELINE round
        for k, v in baseline_env_overrides.items():
            os.environ[k] = v
        # Clear ALT-specific env vars NOT in baseline (back to default)
        for k in alt_env_overrides:
            if k not in baseline_env_overrides:
                os.environ.pop(k, None)
        try:
            baseline = run_shape_aba(*spec, n_runs=n_runs,
                                      cooldown_inter_round=cooldown_inter_round)
        except Exception as e:
            results.append({"shape": name, "error_baseline": str(e)[:300]})
            continue
        time.sleep(cooldown_inter_round)

        # ALT round 2
        for k, v in alt_env_overrides.items():
            os.environ[k] = v
        try:
            alt_b = run_shape_aba(*spec, n_runs=n_runs,
                                   cooldown_inter_round=cooldown_inter_round)
        except Exception as e:
            results.append({"shape": name, "error_alt_b": str(e)[:300]})
            continue
        # Clear ALT env back to baseline default
        for k in alt_env_overrides:
            if k not in baseline_env_overrides:
                os.environ.pop(k, None)

        alt_all = sorted(alt_a + alt_b)
        alt_med = statistics.median(alt_all)
        baseline_med = statistics.median(baseline)
        a_med = statistics.median(alt_a)
        b_med = statistics.median(alt_b)
        drift = abs(a_med - b_med) / a_med * 100 if a_med > 0 else 0
        # Ratio convention: > 1.0 means ALT is SLOWER (i.e., V6NAX baseline is faster)
        ratio = alt_med / baseline_med if baseline_med > 0 else 0

        results.append({
            "shape": name, "B": spec[1], "Hq": spec[2], "Hk": spec[3],
            "qL": spec[4], "kL": spec[5], "D": spec[6],
            "alt_times_a_ms": alt_a, "alt_times_b_ms": alt_b,
            "baseline_times_ms": baseline,
            "alt_median_ms": alt_med, "baseline_median_ms": baseline_med,
            "alt_over_baseline_ratio": ratio,  # > 1 means baseline faster
            "aba_drift_pct": drift,
        })

        if i < len(shapes) - 1:
            time.sleep(cooldown_inter_shape)

    return results


def capture_conditions():
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
    for n, c in [("sw_vers", ["sw_vers"]), ("uptime", ["uptime"]),
                 ("uname", ["uname", "-a"]),
                 ("boottime", ["sysctl", "-n", "kern.boottime"])]:
        try:
            out[n] = subprocess.run(c, check=True, capture_output=True,
                                    text=True, timeout=5).stdout.strip()
        except Exception as e:
            out[f"{n}_error"] = str(e)
    out["mlx_version"] = mx.__version__
    try:
        from mlx_mfa import __version__ as mfa_ver
        out["mlx_mfa_version"] = mfa_ver
    except Exception:
        out["mlx_mfa_version"] = "unknown"
    return out


# Probe definitions — DI2 hypothesis-to-env mapping
PROBES = [
    {
        "name": "B+C+E_aggregate_predecessor_vs_v6nax",
        "description": ("V6NAX vs predecessor path: includes hypotheses B "
                        "(cross-SG sync elim), C (simd_shuffle_xor vs MPP "
                        "reduce), and E (Apple defaults). Aggregate measurement."),
        "baseline_env": {"MFA_V6_USE_NAX": "1"},
        "alt_env": {"MFA_V6_USE_NAX": "0"},  # predecessor path
    },
    {
        "name": "A_tgp_low_sg2",
        "description": ("Hyp A: lower TGP occupancy. MFA_V6_EXEC_SG=2 "
                        "(default=4)."),
        "baseline_env": {"MFA_V6_USE_NAX": "1"},
        "alt_env": {"MFA_V6_USE_NAX": "1", "MFA_V6_EXEC_SG": "2"},
    },
    {
        "name": "A_tgp_high_sg8",
        "description": ("Hyp A: higher TGP occupancy. MFA_V6_EXEC_SG=8."),
        "baseline_env": {"MFA_V6_USE_NAX": "1"},
        "alt_env": {"MFA_V6_USE_NAX": "1", "MFA_V6_EXEC_SG": "8"},
    },
    {
        "name": "D_block_r_64",
        "description": ("Hyp D: larger tile = more register pressure. "
                        "MFA_V6_BLOCK_R=64 (default 32)."),
        "baseline_env": {"MFA_V6_USE_NAX": "1"},
        "alt_env": {"MFA_V6_USE_NAX": "1", "MFA_V6_BLOCK_R": "64"},
    },
    {
        "name": "D_block_c_64",
        "description": ("Hyp D companion: larger K-tile. "
                        "MFA_V6_BLOCK_C=64 (default 32)."),
        "baseline_env": {"MFA_V6_USE_NAX": "1"},
        "alt_env": {"MFA_V6_USE_NAX": "1", "MFA_V6_BLOCK_C": "64"},
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="*", default=None,
                    help="Probe names to run (default all)")
    ap.add_argument("--output",
                    default="docs/v6-nax/v6nax-forward-investigation-data.json")
    ap.add_argument("--cooldown-inter-round", type=float, default=90.0)
    ap.add_argument("--cooldown-inter-shape", type=float, default=60.0)
    ap.add_argument("--cooldown-initial", type=float, default=180.0)
    ap.add_argument("--runs-per-direction", type=int, default=5)
    ap.add_argument("--skip-initial-cooldown", action="store_true")
    args = ap.parse_args()

    # Force V6NAX for smoke gate
    os.environ["MFA_V6_USE_NAX"] = "1"

    print("[v6nax-investig] correctness smoke...", flush=True)
    ok, diag = smoke_gate()
    print(f"  smoke: rmse={diag['rmse']:.4e} maxerr={diag['maxerr']:.4e} "
          f"NaN={diag['n_nan']} Inf={diag['n_inf']} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("[v6nax-investig] STATUS: SMOKE_FAILED", file=sys.stderr, flush=True)
        sys.exit(2)

    if not args.skip_initial_cooldown:
        print(f"[v6nax-investig] initial cooldown {args.cooldown_initial}s",
              flush=True)
        time.sleep(args.cooldown_initial)

    probes_to_run = (PROBES if args.probes is None else
                      [p for p in PROBES if p["name"] in args.probes])

    record = {
        "phase": "V6NAX forward investigation — §4-strict hypothesis isolation",
        "cooldowns": {
            "initial_s": args.cooldown_initial,
            "inter_round_s": args.cooldown_inter_round,
            "inter_shape_s": args.cooldown_inter_shape,
        },
        "runs_per_direction": args.runs_per_direction,
        "smoke_gate": diag,
        "conditions": capture_conditions(),
        "probes": [],
    }

    for i, probe in enumerate(probes_to_run):
        print(f"\n[v6nax-investig] PROBE {i+1}/{len(probes_to_run)}: {probe['name']}",
              flush=True)
        print(f"  {probe['description']}", flush=True)
        print(f"  baseline_env={probe['baseline_env']} alt_env={probe['alt_env']}",
              flush=True)

        results = run_probe(probe["name"], probe["baseline_env"], probe["alt_env"],
                             SHAPES, args.runs_per_direction,
                             args.cooldown_inter_round,
                             args.cooldown_inter_shape)
        for r in results:
            if "error" in r or "error_alt_a" in r:
                err = r.get("error") or r.get("error_alt_a") or r.get("error_baseline") or r.get("error_alt_b")
                print(f"  {r['shape']:<22} ERROR: {err[:80] if err else 'unknown'}",
                      flush=True)
            else:
                print(f"  {r['shape']:<22} "
                      f"ALT={r['alt_median_ms']:>7.3f}ms "
                      f"BASE={r['baseline_median_ms']:>7.3f}ms "
                      f"ALT/BASE={r['alt_over_baseline_ratio']:>5.2f}× "
                      f"drift={r['aba_drift_pct']:>4.1f}%", flush=True)
        record["probes"].append({
            "name": probe["name"],
            "description": probe["description"],
            "baseline_env": probe["baseline_env"],
            "alt_env": probe["alt_env"],
            "results": results,
        })
        if i < len(probes_to_run) - 1:
            time.sleep(args.cooldown_inter_round)

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2))
    print(f"\n[v6nax-investig] wrote: {p}", flush=True)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
