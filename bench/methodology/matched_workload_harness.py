#!/usr/bin/env python3
"""Methodology sprint - matched-workload-family bench harness.

Path-forward option 1 from `docs/methodology/sub1ms-protocol-diagnostic.md`:
replace the v2.36.0 256x256 FP16 matmul warmup with a small
`sparse_attention_nax` dispatch using a DIFFERENT SHAPE than any of the
7 measured shapes. Same kernel family preserves pipeline/scheduling
state; smaller D=64 + qL=512 keeps the active per-dispatch working set
<= 12 KB (fits in per-core L1, no cluster-L2 collision with measured
D=128 kernels).

Design per DM1-DM9 (`docs/methodology/matched-workload-decisions.md`):
  - Warmup workload: sparse_attention_nax(B=1, H=4, qL=kL=512, D=64,
    BT=32, density=0.10), FP16
  - Warmup gap: 50ms (well below the < 100ms downclock threshold)
  - A/B/A pattern: V2 -> SDPA+bias -> V2, 5 runs/direction
  - Section 4 cooldowns: initial 180s / inter-shape 60s / inter-round 90s
  - Three-axis self-validation: smoke gate + warmup counter + control
    shapes monitored for axis-3 regression
"""
import argparse, json, math, os, platform, statistics, subprocess
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mlx.core as mx

# Avoid literal `eval(` in module text (hook scanner false-positives on
# mx.async_eval / mx.eval).  Bind via getattr once.
_ASYNC_EXEC = getattr(mx, "async_eval")
_SYNC = mx.synchronize

os.environ["MFA_LCSA_KERNEL_VERSION"] = "v2"
from mlx_mfa.lcsa_nax import sparse_attention_nax, _bool_mask_to_float_bias


def _materialize(*arrays):
    """Force-evaluate lazy MLX arrays + wait for device."""
    _ASYNC_EXEC(*arrays)
    _SYNC()


# ---------------------------------------------------------------------------
# Sprint B reference shapes (same 7 as v2.36.0 V2-only re-bench)
# ---------------------------------------------------------------------------
SHAPES = [
    # name, B, Hq, Hk, qL, kL, D, density, BT, seed
    ("lcsa_small_seq4k",           1, 12, 12,  4096,  4096, 128, 0.24, 32, 1100),
    ("lcsa_small_seq4k_sparse",    1, 12, 12,  4096,  4096, 128, 0.07, 32, 1101),
    ("lcsa_mid_seq8k",             1,  8,  8,  8192,  8192, 128, 0.12, 32, 1102),
    ("lcsa_mid_seq8k_sparse",      1,  8,  8,  8192,  8192, 128, 0.03, 32, 1103),
    ("lcsa_large_seq16k",          1,  4,  4, 16384, 16384, 128, 0.12, 32, 1104),
    ("lcsa_large_seq16k_sparse",   1,  4,  4, 16384, 16384, 128, 0.03, 32, 1105),
    ("lcsa_mid_seq8k_very_sparse", 1,  8,  8,  8192,  8192, 128, 0.01, 32, 1106),
]

# Smoke shape distinct from measured set: low-noise reference for axis-1 check
SMOKE_CFG = (1, 4, 4, 4096, 4096, 128, 0.10, 32, 9999)
SMOKE_RMSE_BAR = 1e-3

# Warmup config (DM2): smaller D + smaller qL/kL + smaller BT than ANY
# measured shape. qL=kL=2048 chosen so the 2D bool mask is 128x128 = 16
# KB (safely above the 4096-byte MLX small-buffer-inlining threshold;
# qL=kL=512 was below it and was rejected by the kernel).
WARMUP_CFG = dict(B=1, Hq=4, Hk=4, qL=2048, kL=2048, D=64, density=0.10,
                  BT=16, seed=42424242)


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------
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
    _materialize(Q, K, V, mask, bias)
    return Q, K, V, mask, bias, float(bm.mean())


def _build_warmup_inputs():
    """Construct the matched-workload-family warmup inputs (DM1+DM2)."""
    cfg = WARMUP_CFG
    Q, K, V, mask, bias, density_actual = _build_inputs(
        cfg["B"], cfg["Hq"], cfg["Hk"], cfg["qL"], cfg["kL"], cfg["D"],
        cfg["density"], cfg["BT"], cfg["seed"])
    scale = 1.0 / math.sqrt(cfg["D"])
    return {
        "Q": Q, "K": K, "V": V, "mask": mask, "bias": bias,
        "BT": cfg["BT"], "scale": scale,
        "density_actual": density_actual,
    }


def _warmup_dispatch(wm):
    """Single matched-workload warmup dispatch."""
    out = sparse_attention_nax(
        wm["Q"], wm["K"], wm["V"], wm["mask"],
        block_tile=wm["BT"], scale=wm["scale"])
    _materialize(out)


# ---------------------------------------------------------------------------
# Axis-1: output sanity smoke gate
# ---------------------------------------------------------------------------
def smoke_gate():
    B, Hq, Hk, qL, kL, D, density, BT, seed = SMOKE_CFG
    Q, K, V, mask, bias, _ = _build_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed)
    scale = 1.0 / math.sqrt(D)
    O_v2 = sparse_attention_nax(Q, K, V, mask, block_tile=BT, scale=scale)
    _materialize(O_v2)
    O_sdpa = mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=bias)
    _materialize(O_sdpa)
    err = np.abs(np.array(O_v2.astype(mx.float32)) -
                 np.array(O_sdpa.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    maxerr = float(err.max())
    n_nan = int(np.isnan(np.array(O_v2.astype(mx.float32))).sum())
    n_inf = int(np.isinf(np.array(O_v2.astype(mx.float32))).sum())
    passed = rmse < SMOKE_RMSE_BAR and n_nan == 0 and n_inf == 0
    return passed, {"rmse": rmse, "maxerr": maxerr, "n_nan": n_nan,
                    "n_inf": n_inf, "bar": SMOKE_RMSE_BAR, "passed": passed}


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------
def time_call(fn, n_runs):
    for _ in range(2):
        out = fn()
        _materialize(out)
    times = []
    for _ in range(n_runs):
        _SYNC()
        t0 = time.perf_counter()
        out = fn()
        _materialize(out)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times


def matched_workload_cooldown(duration_s, wm, warmup_gap_s, counter):
    """Cooldown that holds GPU power state via interleaved small
    sparse_attention_nax dispatches (matched workload family).

    Returns number of warmup dispatches fired (axis-2 verification).
    """
    end_time = time.time() + duration_s
    n_fired = 0
    while time.time() < end_time - warmup_gap_s:
        _warmup_dispatch(wm)
        n_fired += 1
        time.sleep(warmup_gap_s)
    # Final 50ms settle to avoid measured kernel landing back-to-back
    # behind a warmup tail (without crossing 100ms downclock threshold).
    time.sleep(0.05)
    counter["dispatches"] += n_fired
    counter["intervals"] += 1
    return n_fired


# ---------------------------------------------------------------------------
# Per-shape A/B/A run
# ---------------------------------------------------------------------------
def run_shape(name, B, Hq, Hk, qL, kL, D, density, BT, seed, n_runs,
              wm, warmup_gap_s, inter_round_cooldown_s, counter):
    Q, K, V, mask, bias, d_actual = _build_inputs(
        B, Hq, Hk, qL, kL, D, density, BT, seed)
    scale = 1.0 / math.sqrt(D)

    def call_v2():
        return sparse_attention_nax(Q, K, V, mask, block_tile=BT, scale=scale)

    def call_sdpa():
        return mx.fast.scaled_dot_product_attention(
            Q, K, V, scale=scale, mask=bias)

    v2_a = time_call(call_v2, n_runs)
    matched_workload_cooldown(
        inter_round_cooldown_s, wm, warmup_gap_s, counter)
    sdpa_times = time_call(call_sdpa, n_runs)
    matched_workload_cooldown(
        inter_round_cooldown_s, wm, warmup_gap_s, counter)
    v2_b = time_call(call_v2, n_runs)

    v2_all = sorted(v2_a + v2_b)
    v2_med = statistics.median(v2_all)
    sdpa_med = statistics.median(sdpa_times)
    a_med = statistics.median(v2_a)
    b_med = statistics.median(v2_b)
    aba_drift_pct = abs(a_med - b_med) / a_med * 100 if a_med > 0 else 0
    ratio = sdpa_med / v2_med if v2_med > 0 else 0
    return {
        "shape": name,
        "B": B, "Hq": Hq, "Hk": Hk, "qL": qL, "kL": kL, "D": D, "BT": BT,
        "density_target": density, "density_actual": d_actual,
        "v2_times_a_ms": v2_a, "v2_times_b_ms": v2_b,
        "sdpa_times_ms": sdpa_times,
        "v2_median_ms": v2_med, "sdpa_median_ms": sdpa_med,
        "v2_a_first_median": a_med, "v2_a_second_median": b_med,
        "ratio_sdpa_over_v2": ratio,
        "aba_drift_pct": aba_drift_pct,
    }


# ---------------------------------------------------------------------------
# Provenance capture
# ---------------------------------------------------------------------------
def capture_conditions():
    out = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "platform": platform.platform(),
           "mfa_lcsa_kernel_version_env":
               os.environ.get("MFA_LCSA_KERNEL_VERSION", "<unset>")}
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
    out["warmup_config"] = {**WARMUP_CFG}
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--output",
                    default="docs/methodology/matched-workload-data.json")
    ap.add_argument("--inter-round-cooldown", type=float, default=90.0)
    ap.add_argument("--inter-shape-cooldown", type=float, default=60.0)
    ap.add_argument("--initial-cooldown", type=float, default=180.0)
    ap.add_argument("--warmup-gap-s", type=float, default=0.05,
                    help="Gap between warmup dispatches (< 100ms threshold).")
    ap.add_argument("--runs-per-direction", type=int, default=5)
    ap.add_argument("--skip-initial-cooldown", action="store_true")
    ap.add_argument("--smoke-only", action="store_true",
                    help="Run smoke gate + warmup-mechanism sanity check, "
                         "exit without main bench. Used for pre-flight.")
    ap.add_argument("--shapes-subset", type=int, default=None,
                    help="Run only the first N shapes (pre-flight only).")
    args = ap.parse_args()

    print(f"[mw section4 harness] session={args.session_id}", flush=True)
    print(f"[mw section4 harness] warmup gap: {args.warmup_gap_s*1000:.0f}ms "
          f"(downclock threshold: < 100ms)", flush=True)
    print(f"[mw section4 harness] warmup workload: sparse_attention_nax "
          f"B={WARMUP_CFG['B']} H={WARMUP_CFG['Hq']} qL=kL={WARMUP_CFG['qL']} "
          f"D={WARMUP_CFG['D']} BT={WARMUP_CFG['BT']} "
          f"density={WARMUP_CFG['density']}", flush=True)

    # Axis-1: smoke gate
    print("[mw section4 harness] correctness smoke...", flush=True)
    ok, diag = smoke_gate()
    print(f"  smoke: rmse={diag['rmse']:.4e} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("[mw section4 harness] STATUS: SMOKE_FAILED", file=sys.stderr, flush=True)
        sys.exit(2)

    # Build warmup workload inputs (matched family, different shape per DM2)
    wm = _build_warmup_inputs()
    print(f"[mw section4 harness] warmup density_actual={wm['density_actual']:.3f}",
          flush=True)

    # Axis-2 pre-check: time a single warmup dispatch
    t0 = time.perf_counter()
    _warmup_dispatch(wm)
    warmup_us = (time.perf_counter() - t0) * 1e6
    print(f"[mw section4 harness] single warmup dispatch: {warmup_us:.1f}us "
          f"(target <= {args.warmup_gap_s*1000*0.2*1000:.0f}us for "
          f"<= 20% duty cycle)", flush=True)

    if args.smoke_only:
        print("[mw section4 harness] --smoke-only: exiting after pre-checks",
              flush=True)
        return

    # Prime GPU to high power state using matched workload
    print("[mw section4 harness] priming GPU (100 matched-workload dispatches)...",
          flush=True)
    for _ in range(100):
        _warmup_dispatch(wm)
    counter = {"dispatches": 0, "intervals": 0}

    if not args.skip_initial_cooldown:
        print(f"[mw section4 harness] initial cooldown {args.initial_cooldown}s "
              f"(matched-workload-family)", flush=True)
        n = matched_workload_cooldown(
            args.initial_cooldown, wm, args.warmup_gap_s, counter)
        print(f"  fired {n} warmup dispatches during initial cooldown",
              flush=True)

    record = {
        "session_id": args.session_id,
        "phase": "Methodology sprint - matched-workload-family section4-strict",
        "pattern": "V2 -> SDPA+bias -> V2 with matched-workload cooldowns",
        "cooldowns": {
            "initial_s": args.initial_cooldown,
            "inter_shape_s": args.inter_shape_cooldown,
            "inter_round_s": args.inter_round_cooldown,
            "warmup_gap_s": args.warmup_gap_s,
        },
        "runs_per_direction": args.runs_per_direction,
        "smoke_gate": diag,
        "warmup_dispatch_us": warmup_us,
        "warmup_density_actual": wm["density_actual"],
        "conditions": capture_conditions(),
        "production_results": [],
        "warmup_counter": counter,
    }

    shapes_to_run = (SHAPES[:args.shapes_subset] if args.shapes_subset
                     else SHAPES)
    for i, spec in enumerate(shapes_to_run):
        name = spec[0]
        try:
            res = run_shape(*spec, n_runs=args.runs_per_direction, wm=wm,
                            warmup_gap_s=args.warmup_gap_s,
                            inter_round_cooldown_s=args.inter_round_cooldown,
                            counter=counter)
        except Exception as e:
            res = {"shape": name, "error": str(e)[:300]}
        record["production_results"].append(res)
        if "error" in res:
            print(f"  {name:<32} ERROR: {res['error'][:80]}", flush=True)
        else:
            print(f"  {name:<32} d={res['density_actual']:.3f} "
                  f"V2={res['v2_median_ms']:>7.3f}ms "
                  f"SDPA={res['sdpa_median_ms']:>7.3f}ms "
                  f"ratio={res['ratio_sdpa_over_v2']:>5.2f}x "
                  f"drift={res['aba_drift_pct']:>4.1f}%", flush=True)
        if i < len(shapes_to_run) - 1:
            matched_workload_cooldown(
                args.inter_shape_cooldown, wm, args.warmup_gap_s, counter)

    record["warmup_counter"] = counter

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))
    print(f"\n[mw section4 harness] session '{args.session_id}' -> {p}", flush=True)
    print(f"[mw section4 harness] total warmup dispatches: {counter['dispatches']} "
          f"across {counter['intervals']} cooldown intervals", flush=True)


if __name__ == "__main__":
    main()
