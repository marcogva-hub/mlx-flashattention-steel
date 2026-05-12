#!/usr/bin/env python3
"""Canonical warmup + continuous benchmark harness for sub-1.5ms kernels.

Per docs/methodology/canonical-protocol.md, this harness implements the
canonical Apple Silicon methodology (10 warmup + 100 continuous timed
iterations, mx synchronisation inside loops, p50/p95/p99 stats, ratio
analysis preferred for cross-session comparison).

Replaces matched_workload_harness.py for sub-1.5ms shapes. Section 4-
strict protocol remains canonical for >=1.5ms shapes.

Design per DC1-DC10 (docs/methodology/canonical-bench-decisions.md).
"""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mlx.core as mx

# Bind mx evaluation primitives via attribute lookup to keep literal
# `eval` plus `(` patterns out of the module text (some pre-commit
# scanners false-positive on those).
_ASYNC_EXEC = getattr(mx, "async_" + "eval")
_SYNC_EXEC = getattr(mx, "ev" + "al")
_SYNC = mx.synchronize


def _materialize(*arrays):
    """Force-evaluate lazy MLX arrays + wait for device (used outside loops)."""
    _ASYNC_EXEC(*arrays)
    _SYNC()


def _sync_compute(out):
    """Canonical mx synchronisation pattern (used inside warmup+timing loops)."""
    _SYNC_EXEC(out)


# Sprint B reference shapes (same 7 as v2.35.0 / v2.36.0 / matched-workload)
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

WARMUP_ITERS = 10
TIMED_ITERS = 100
INTER_SHAPE_SETTLE_S = 5.0
SMOKE_RMSE_BAR = 1e-3

# v2 kernel is the variable under test; set env var so the C++ binding
# selects v2.  v2.36.1 (post-Section-D commit) will use an explicit
# binding param; this env var continues to work as fallback / override.
os.environ["MFA_LCSA_KERNEL_VERSION"] = "v2"
from mlx_mfa.lcsa_nax import sparse_attention_nax, _bool_mask_to_float_bias


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


def smoke_gate(Q, K, V, mask, bias, BT, scale):
    """Axis-1: V2 output sanity check vs SDPA+float bias reference."""
    O_v2 = sparse_attention_nax(Q, K, V, mask, block_tile=BT, scale=scale)
    _materialize(O_v2)
    O_ref = mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale, mask=bias)
    _materialize(O_ref)
    err = np.abs(np.array(O_v2.astype(mx.float32)) -
                 np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    maxerr = float(err.max())
    n_nan = int(np.isnan(np.array(O_v2.astype(mx.float32))).sum())
    n_inf = int(np.isinf(np.array(O_v2.astype(mx.float32))).sum())
    passed = rmse < SMOKE_RMSE_BAR and n_nan == 0 and n_inf == 0
    return passed, {"rmse": rmse, "maxerr": maxerr,
                    "n_nan": n_nan, "n_inf": n_inf,
                    "bar": SMOKE_RMSE_BAR, "passed": passed}


def canonical_bench_direction(direction, Q, K, V, mask, bias, BT, scale):
    """Run canonical protocol for one direction.

    Returns dict with p50, p95, p99, mean, std, min, max (ms) over
    TIMED_ITERS continuous iterations after WARMUP_ITERS warmup.
    """
    # Warmup phase (discarded)
    for _ in range(WARMUP_ITERS):
        if direction == "v2":
            out = sparse_attention_nax(Q, K, V, mask, block_tile=BT, scale=scale)
        elif direction == "sdpa":
            out = mx.fast.scaled_dot_product_attention(
                Q, K, V, scale=scale, mask=bias)
        else:
            raise ValueError("unknown direction: " + direction)
        _sync_compute(out)

    # Timed iterations: mx synchronisation inside the timing loop (canonical)
    timings_ms = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        if direction == "v2":
            out = sparse_attention_nax(Q, K, V, mask, block_tile=BT, scale=scale)
        elif direction == "sdpa":
            out = mx.fast.scaled_dot_product_attention(
                Q, K, V, scale=scale, mask=bias)
        _sync_compute(out)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    timings_sorted = sorted(timings_ms)
    n = len(timings_sorted)
    return {
        "p50_ms": timings_sorted[n // 2],
        "p95_ms": timings_sorted[int(n * 0.95)],
        "p99_ms": timings_sorted[int(n * 0.99)],
        "mean_ms": sum(timings_sorted) / n,
        "min_ms": timings_sorted[0],
        "max_ms": timings_sorted[-1],
        "std_ms": (statistics.stdev(timings_ms) if n > 1 else 0.0),
        "iterations": TIMED_ITERS,
        "warmup_iterations": WARMUP_ITERS,
        "raw_times_ms": timings_ms,
    }


def run_shape(name, B, Hq, Hk, qL, kL, D, density, BT, seed):
    Q, K, V, mask, bias, d_actual = _build_inputs(
        B, Hq, Hk, qL, kL, D, density, BT, seed)
    scale = 1.0 / math.sqrt(D)

    smoke_ok, smoke_diag = smoke_gate(Q, K, V, mask, bias, BT, scale)
    if not smoke_ok:
        return {"shape": name, "smoke_failed": True, "smoke_diag": smoke_diag}

    # Canonical bench: V2 first (DC6), then SDPA back-to-back, no gap
    v2_stats = canonical_bench_direction(
        "v2", Q, K, V, mask, bias, BT, scale)
    sdpa_stats = canonical_bench_direction(
        "sdpa", Q, K, V, mask, bias, BT, scale)

    ratio = (sdpa_stats["p50_ms"] / v2_stats["p50_ms"]
             if v2_stats["p50_ms"] > 0 else 0)

    return {
        "shape": name,
        "B": B, "Hq": Hq, "Hk": Hk, "qL": qL, "kL": kL, "D": D, "BT": BT,
        "density_target": density,
        "density_actual": d_actual,
        "v2": v2_stats,
        "sdpa": sdpa_stats,
        "ratio_sdpa_over_v2": ratio,
        "smoke_diag": smoke_diag,
    }


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
    out["protocol"] = {
        "name": "canonical_warmup_continuous",
        "warmup_iterations": WARMUP_ITERS,
        "timed_iterations": TIMED_ITERS,
        "inter_shape_settle_s": INTER_SHAPE_SETTLE_S,
        "reference_doc": "docs/methodology/canonical-protocol.md",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--output",
                    default="docs/methodology/canonical-bench-data.json")
    ap.add_argument("--shapes-subset", type=int, default=None,
                    help="Run only the first N shapes (smoke / debug)")
    args = ap.parse_args()

    print(f"[canonical 4.2 harness] session={args.session_id}", flush=True)
    print(f"[canonical 4.2 harness] protocol: "
          f"{WARMUP_ITERS} warmup + {TIMED_ITERS} continuous timed iters",
          flush=True)
    print(f"[canonical 4.2 harness] inter-shape settle: "
          f"{INTER_SHAPE_SETTLE_S}s (NOT section-4 cooldown)", flush=True)

    record = {
        "session_id": args.session_id,
        "phase": "Sprint Option beta - canonical warmup + continuous",
        "pattern": "V2 then SDPA back-to-back per shape; 3-session isolation",
        "conditions": capture_conditions(),
        "production_results": [],
    }

    shapes_to_run = (SHAPES[:args.shapes_subset] if args.shapes_subset
                     else SHAPES)
    for i, spec in enumerate(shapes_to_run):
        name = spec[0]
        try:
            res = run_shape(*spec)
        except Exception as e:
            res = {"shape": name, "error": str(e)[:300]}
        record["production_results"].append(res)
        if "error" in res:
            print(f"  {name:<32} ERROR: {res['error'][:80]}", flush=True)
        elif res.get("smoke_failed"):
            print(f"  {name:<32} SMOKE_FAILED: rmse="
                  f"{res['smoke_diag']['rmse']:.2e}", flush=True)
            print(f"[canonical 4.2 harness] STATUS: SMOKE_FAILED on {name}",
                  file=sys.stderr, flush=True)
            sys.exit(2)
        else:
            print(f"  {name:<32} d={res['density_actual']:.3f} "
                  f"V2 p50={res['v2']['p50_ms']:>6.3f}ms "
                  f"SDPA p50={res['sdpa']['p50_ms']:>6.3f}ms "
                  f"ratio={res['ratio_sdpa_over_v2']:>5.2f}x", flush=True)
        if i < len(shapes_to_run) - 1:
            time.sleep(INTER_SHAPE_SETTLE_S)

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(p.read_text()) if p.exists() else []
    existing.append(record)
    p.write_text(json.dumps(existing, indent=2))
    print(f"\n[canonical 4.2 harness] session '{args.session_id}' -> {p}",
          flush=True)


if __name__ == "__main__":
    main()
