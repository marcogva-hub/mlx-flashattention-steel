#!/usr/bin/env python3
"""V6 NAX forward autoresearch harness (M5 Max) — Phase 0 (knob-map + baseline) and
Phase 1 (live-knob sweep).  Measures the PRODUCTION-dispatched kernel
(`_ext.v6_nax_forward(..., force_v6nax=True)`) vs Apple SDPA, on hardware.

Disciplines: absolute ms (no bare ratios), effective-FLOP TFLOPS plausibility-gated at
51.8, 3-replicate median, correctness vs an INDEPENDENT fp32 reference before trusting time.
"""
from __future__ import annotations
import math, time, os, sys
import numpy as np
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward

PEAK_TFLOPS = 51.8

# ≥6 production-length shapes per D (B, H, N) — N==S; incl VSR-class proxies.
SHAPES = {
    128: [(1, 8, 1024), (1, 8, 2048), (1, 8, 4096), (1, 8, 8192),
          (1, 16, 4096), (1, 10, 8192)],   # last two: SeedVR2 / FlashVSR-class
    64:  [(1, 8, 2048), (1, 8, 4096), (1, 8, 8192), (1, 8, 16384),
          (1, 16, 8192), (1, 10, 16384)],  # D=64 NAX-eligible (Nk>8000 mostly)
}


def _mk(B, H, N, D, dt, seed=0):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(dt)
    k = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(dt)
    v = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(dt)
    mx.eval(q, k, v)
    return q, k, v


def _fp32_ref(q, k, v, D):
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    s = (qf @ kf.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(D))
    o = mx.softmax(s, axis=-1) @ vf
    mx.eval(o)
    return o


def _err(a, b):
    mx.eval(a, b)
    return float(np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32))).max())


def bench(fn, warm=8, it=20, reps=3):
    meds = []
    for _ in range(reps):
        for _ in range(warm):
            mx.eval(fn())
        mx.synchronize(); ts = []
        for _ in range(it):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
            ts.append(time.perf_counter() - t0)
        meds.append(sorted(ts)[len(ts) // 2])
    m = sorted(meds)[len(meds) // 2]
    cv = float(np.std(meds) / max(np.mean(meds), 1e-12) * 100)
    return m * 1e3, cv


def tflops(ms, B, H, N, D):
    return (4.0 * B * H * N * N * D) / (ms * 1e-3) / 1e12


def nax_fn(q, k, v):
    return lambda: v6_nax_forward(q, k, v, False, True)[0]


def sdpa_fn(q, k, v, D):
    sc = 1.0 / math.sqrt(D)
    return lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=sc)


def baseline(dtypes=(mx.float16, mx.bfloat16)):
    print("=== Phase 0.4 BASELINE: NAX (default) vs SDPA — absolute ms, direction, eff TFLOPS (gate 51.8) ===")
    print(f"{'D dt shape':28s} {'SDPA ms':>9s} {'NAX ms':>9s} {'dir':>14s} {'NAXcv%':>7s} {'NAX TF':>7s}")
    for D in (128, 64):
        for dt in dtypes:
            for (B, H, N) in SHAPES[D]:
                q, k, v = _mk(B, H, N, D, dt, seed=7)
                o = v6_nax_forward(q, k, v, False, True)[0]
                e = _err(o, _fp32_ref(q, k, v, D))
                if not (np.isfinite(e) and e < 2e-2):
                    print(f"  D{D} {dt} ({B},{H},{N}): CORRECTNESS FAIL err={e:.2e}"); continue
                t_sdpa, _ = bench(sdpa_fn(q, k, v, D))
                t_nax, cv = bench(nax_fn(q, k, v))
                tf = tflops(t_nax, B, H, N, D)
                gate = "" if tf <= PEAK_TFLOPS else " ⚠>PEAK"
                direction = "NAX faster" if t_nax < t_sdpa * 0.98 else ("parity" if t_nax <= t_sdpa * 1.02 else "NAX slower")
                dn = "fp16" if dt == mx.float16 else "bf16"
                print(f"D{D} {dn} ({B},{H},{N})"[:28].ljust(28) +
                      f" {t_sdpa:>8.2f} {t_nax:>8.2f} {direction:>14s} {cv:>6.1f} {tf:>6.0f}{gate}")


def knob_map():
    """Phase 0.2: classify each knob LIVE / NO-OP / vestigial at D=128 N=4096 fp16,
    by time delta vs default (+ output delta for precision knobs)."""
    D, B, H, N = 128, 1, 8, 4096
    q, k, v = _mk(B, H, N, D, mx.float16, seed=7)
    ref = _fp32_ref(q, k, v, D)
    base_ms, base_cv = bench(nax_fn(q, k, v))
    base_out = v6_nax_forward(q, k, v, False, True)[0]; mx.eval(base_out)
    print(f"\n=== Phase 0.2 KNOB MAP (D=128 N=4096 fp16; default {base_ms:.2f}ms cv={base_cv:.1f}%) ===")
    print(f"{'knob=val':28s} {'ms':>8s} {'Δ% vs def':>10s} {'outΔ':>9s} {'class':>10s}")
    knobs = [
        ("MFA_V6_NAX_BQ", "32"), ("MFA_V6_NAX_BK", "64"), ("MFA_V6_NAX_WM", "2"),
        ("MFA_V6_BLOCK_D", "64"), ("MFA_V6_UNROLL_MODE", "none"),
        ("MFA_V6_RELAXED_PRECISION", "0"), ("MFA_V6_FORCE_DYNAMIC_K", "1"),
        ("MFA_V6_BLOCK_R", "32"), ("MFA_V6_BLOCK_C", "64"), ("MFA_V6_EXEC_SG", "2"),
        ("MFA_V6_MAX_THREADS", "256"),
    ]
    for name, val in knobs:
        os.environ[name] = val
        try:
            out = v6_nax_forward(q, k, v, False, True)[0]; mx.eval(out)
            e = _err(out, ref); od = _err(out, base_out)
            ms, _ = bench(nax_fn(q, k, v))
        except Exception as ex:
            os.environ.pop(name, None)
            print(f"{name}={val}"[:28].ljust(28) + f"  RAISED: {type(ex).__name__}: {str(ex)[:40]}")
            continue
        os.environ.pop(name, None)
        dpct = (ms - base_ms) / base_ms * 100.0
        live = (abs(dpct) > 5.0) or (od > 1e-4)
        cls = "LIVE" if live else "no-op"
        bad = " WRONG!" if not (np.isfinite(e) and e < 2e-2) else ""
        print(f"{name}={val}"[:28].ljust(28) + f" {ms:>7.2f} {dpct:>+9.1f}% {od:>8.1e} {cls:>10s}{bad}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    if len(sys.argv) > 1 and sys.argv[1] == "knobs":
        knob_map()
    elif len(sys.argv) > 1 and sys.argv[1] == "baseline":
        baseline()
    else:
        baseline(); knob_map()
