#!/usr/bin/env python3
"""Phase 1.5 perf-sweep harness — NAX conv3d vs MLX conv_general.

Methodology (Sprint A precedent):
- 6 production shapes × A/B/A (NAX → MLX → NAX) × 5 runs per direction
- §4 cooldowns: 60s/shape, 90s/round, 180s/initial
- Subprocess isolation per session (Artifact #1)
- Conditions sidecar per Artifact #5 sub-rule 5b
- Pre-flight correctness smoke gate (Phase 1.1 lesson learned)
- Wall-clock measurement via perf_counter + mx.synchronize boundaries

Output: docs/conv-nax/conv-nax-phase1_5-perfsweep.json — per-session
record with per-shape NAX/MLX timings + ratios.

Variance handling per Sprint A §B.7 (applied post-bench):
  - cross-session range < 10% → ratio confident
  - 10-20% range → boundary; opt-in default
  - > 20% range → §B.7 high-variance fallback (shelve possible)
"""
import argparse, json, math, os, platform, statistics, subprocess
import sys, time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_mfa.conv_nax import conv3d_nax_forward, estimate_working_set


# ---------------------------------------------------------------------
# 6 production shapes per design §3.1 + Phase 1.5 prompt §F.1.
# ---------------------------------------------------------------------
SHAPES = [
    # (label, B, T, H, W, C_in, C_out, K_T, K_H, K_W)
    ("mid_resnet",             1,  5, 64,  64,  512, 512, 3, 3, 3),
    ("up1_resnet",             1,  9, 128, 128, 512, 512, 3, 3, 3),
    ("up2_resnet0_chunk_cap",  1, 11, 150, 180, 512, 256, 3, 3, 3),
    ("up3_resnet_chunk_cap",   1, 24, 128, 193, 128, 128, 3, 3, 3),
    ("up2_resnet_full",        1, 17, 256, 256, 256, 256, 3, 3, 3),
    ("up2_resnet0_peakflops",  1, 17, 256, 256, 512, 256, 3, 3, 3),
]

# Smoke shape (small, fast, correctness check).
SMOKE_CFG = (1, 3, 32, 32, 64, 64, 3, 3, 3)
SMOKE_RMSE_BAR = 1e-3


def make_inputs(B, T, H, W, C_in, C_out, K_T, K_H, K_W, seed=0):
    mx.random.seed(seed)
    x = (mx.random.uniform(shape=(B, T, H, W, C_in)) * 0.1).astype(mx.float16)
    w = (mx.random.uniform(shape=(C_out, K_T, K_H, K_W, C_in)) * 0.1).astype(mx.float16)
    mx.async_eval(x, w); mx.synchronize()
    return x, w


def smoke_gate():
    """Phase 1.1 lesson: correctness smoke BEFORE any timing.

    Returns (passed: bool, diag: dict). Smoke shape is small; FP16 round-off
    is negligible.
    """
    B, T, H, W, C_in, C_out, K_T, K_H, K_W = SMOKE_CFG
    x, w = make_inputs(B, T, H, W, C_in, C_out, K_T, K_H, K_W, seed=0)
    pad = (K_T // 2, K_H // 2, K_W // 2)
    y_nax = conv3d_nax_forward(x, w, stride=(1,1,1), padding=pad, dilation=(1,1,1))
    y_mlx = mx.conv_general(x, w, stride=[1,1,1], padding=list(pad), kernel_dilation=[1,1,1])
    mx.async_eval(y_nax, y_mlx); mx.synchronize()
    n_nan = int(mx.sum(mx.isnan(y_nax.astype(mx.float32))))
    n_inf = int(mx.sum(mx.isinf(y_nax.astype(mx.float32))))
    err = mx.abs(y_nax.astype(mx.float32) - y_mlx.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(err*err)))
    mag = float(mx.max(mx.abs(y_mlx.astype(mx.float32))))
    rel = rmse / mag if mag > 0 else 0
    passed = rel < SMOKE_RMSE_BAR and n_nan == 0 and n_inf == 0
    return passed, {"rel_err": rel, "rmse": rmse, "mag": mag,
                    "n_nan": n_nan, "n_inf": n_inf, "passed": passed,
                    "bar": SMOKE_RMSE_BAR}


def time_call(func, n_runs):
    """Time a callable n_runs times, return per-run ms list."""
    # Warmup (not timed)
    func(); mx.synchronize()
    times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        y = func()
        mx.async_eval(y); mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times


def run_shape(label, B, T, H, W, C_in, C_out, K_T, K_H, K_W, *, n_runs):
    """Time NAX → MLX → NAX (A/B/A) on this shape."""
    pad = (K_T // 2, K_H // 2, K_W // 2)
    x, w = make_inputs(B, T, H, W, C_in, C_out, K_T, K_H, K_W, seed=42)

    def call_nax():
        return conv3d_nax_forward(x, w, stride=(1,1,1), padding=pad, dilation=(1,1,1))
    def call_mlx():
        return mx.conv_general(x, w, stride=[1,1,1], padding=list(pad),
                                kernel_dilation=[1,1,1])

    # A/B/A pattern per Sprint A precedent: NAX → MLX → NAX
    nax_times_a = time_call(call_nax, n_runs)
    mlx_times   = time_call(call_mlx, n_runs)
    nax_times_b = time_call(call_nax, n_runs)

    # Combine both NAX runs for the canonical timing.
    nax_all = sorted(nax_times_a + nax_times_b)
    nax_med = statistics.median(nax_all)
    mlx_med = statistics.median(mlx_times)
    ratio = mlx_med / nax_med if nax_med > 0 else 0
    # A/B/A drift: |median(A) - median(B)| / median(A) for within-session
    a_med = statistics.median(nax_times_a)
    b_med = statistics.median(nax_times_b)
    aba_drift_pct = abs(a_med - b_med) / a_med * 100 if a_med > 0 else 0

    M = B*T*H*W
    K = K_T*K_H*K_W*C_in
    flops = 2.0 * M * K * C_out
    nax_tflops = flops / (nax_med * 1e-3) / 1e12
    mlx_tflops = flops / (mlx_med * 1e-3) / 1e12

    return {
        "shape": label,
        "B": B, "T": T, "H": H, "W": W, "C_in": C_in, "C_out": C_out,
        "K_T": K_T, "K_H": K_H, "K_W": K_W,
        "M": M, "K": K, "N": C_out,
        "nax_times_a_ms": nax_times_a,
        "nax_times_b_ms": nax_times_b,
        "mlx_times_ms":   mlx_times,
        "nax_median_ms": nax_med,
        "mlx_median_ms": mlx_med,
        "ratio_mlx_over_nax": ratio,  # > 1.0 → NAX faster
        "aba_drift_pct": aba_drift_pct,
        "nax_TFLOPS": nax_tflops,
        "mlx_TFLOPS": mlx_tflops,
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_label")
    ap.add_argument("--data_path",
                    default="docs/conv-nax/conv-nax-phase1_5-perfsweep.json")
    ap.add_argument("--cooldown_shape", type=float, default=60.0)
    ap.add_argument("--cooldown_initial", type=float, default=180.0)
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--skip_initial_cooldown", action="store_true")
    args = ap.parse_args()

    print(f"[phase1.5 harness] session={args.session_label}")

    # Pre-flight smoke gate.
    print("[phase1.5 harness] correctness smoke (Phase 1.1 lesson)...")
    passed, diag = smoke_gate()
    print(f"  smoke: rel_err={diag['rel_err']:.4e}  rmse={diag['rmse']:.6f}  "
          f"NaN={diag['n_nan']} -> {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("[phase1.5 harness] STATUS: SMOKE_FAILED", file=sys.stderr)
        sys.exit(2)

    if not args.skip_initial_cooldown:
        print(f"[phase1.5 harness] initial cooldown {args.cooldown_initial}s")
        time.sleep(args.cooldown_initial)

    record = {
        "session_label": args.session_label,
        "phase": "C-1.5 perf-sweep (NAX vs MLX conv_general)",
        "cooldowns": {"shape_s": args.cooldown_shape,
                      "initial_s": args.cooldown_initial,
                      "skip_initial": args.skip_initial_cooldown},
        "n_runs": args.n_runs,
        "smoke_gate": diag,
        "conditions": capture_conditions(),
        "results": [],
    }

    for spec in SHAPES:
        label = spec[0]
        try:
            res = run_shape(*spec, n_runs=args.n_runs)
        except Exception as e:
            res = {"shape": label, "error": str(e)[:300]}
        record["results"].append(res)
        if "error" in res:
            print(f"  {label:<28} ERROR: {res['error'][:80]}")
        else:
            print(f"  {label:<28} M={res['M']:>8} K={res['K']:>6}  "
                  f"NAX={res['nax_median_ms']:>8.2f}ms ({res['nax_TFLOPS']:>5.2f} TF)  "
                  f"MLX={res['mlx_median_ms']:>8.2f}ms ({res['mlx_TFLOPS']:>5.2f} TF)  "
                  f"ratio={res['ratio_mlx_over_nax']:>5.2f}× "
                  f"drift={res['aba_drift_pct']:>4.1f}%")
        time.sleep(args.cooldown_shape)

    p = Path(args.data_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))
    print(f"\n[phase1.5 harness] session '{args.session_label}' → {p}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
