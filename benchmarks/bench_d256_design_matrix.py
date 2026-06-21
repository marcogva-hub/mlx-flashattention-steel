#!/usr/bin/env python3
"""D=256 design-space benchmark matrix (separate-family pass).

This benchmark intentionally focuses only on D=256 and compares currently
relevant routed paths against SDPA:
  - SDPA baseline
  - MFA V1 fallback route (`MFA_DISABLE_V2=1`)
  - MFA V2 D-split default
  - MFA V2 D-split with BK overrides (`MFA_V2_FORCE_BK=32|64`)
  - AUTO backend route
  - split-K force toggles (expected no-op for D=256, recorded explicitly)

Profiles:
  - production-like: B=2, H=8
  - under-occupied:  B=1, H=1

Shapes:
  - N = {2048, 4096, 8192, 16384}
  - causal in {False, True}
  - dtype in {f16, bf16-if-supported}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager

import mlx.core as mx

from mlx_mfa import __version__, flash_attention, get_device_info
from mlx_mfa.attention import _fallback_sdpa, _mfa_forward
from mlx_mfa.dispatch_policy import should_use_mfa


@contextmanager
def _env(name: str, value: str | None):
    prev = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def _measure(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return float(times[len(times) // 2])


def _dtype_supported(dtype: mx.Dtype) -> bool:
    if dtype == mx.float16:
        return True
    if dtype == mx.bfloat16:
        try:
            q = mx.random.normal([1, 1, 16, 64]).astype(mx.bfloat16)
            k = mx.random.normal([1, 1, 16, 64]).astype(mx.bfloat16)
            v = mx.random.normal([1, 1, 16, 64]).astype(mx.bfloat16)
            out = flash_attention(
                q,
                k,
                v,
                scale=1.0 / math.sqrt(64.0),
                causal=True,
                backend="mfa",
            )
            mx.eval(out)
            return True
        except Exception:
            return False
    return False


def _dtype_name(dtype: mx.Dtype) -> str:
    if dtype == mx.float16:
        return "f16"
    if dtype == mx.bfloat16:
        return "bf16"
    return str(dtype)


def _infer_auto_path(
    auto_ms: float,
    sdpa_ms: float,
    v1_ms: float,
    v2_default_ms: float,
    v2_bk32_ms: float,
    v2_bk64_ms: float,
    policy_use_mfa: bool,
) -> str:
    if not policy_use_mfa:
        return "sdpa(policy)"
    candidates = {
        "v1": v1_ms,
        "v2_dsplit_default": v2_default_ms,
        "v2_dsplit_bk32": v2_bk32_ms,
        "v2_dsplit_bk64": v2_bk64_ms,
        "sdpa": sdpa_ms,
    }
    best = min(candidates.items(), key=lambda kv: abs(kv[1] - auto_ms))[0]
    return f"inferred:{best}"


def _row_classification(best_ratio_vs_sdpa: float) -> str:
    if best_ratio_vs_sdpa >= 1.02:
        return "maybe_win"
    if best_ratio_vs_sdpa >= 0.95:
        return "neutral"
    return "losing"


def run_case(
    profile_name: str,
    B: int,
    H: int,
    N: int,
    causal: bool,
    dtype: mx.Dtype,
    is_m3_plus: bool,
    gpu_cores: int,
    warmup: int,
    iters: int,
) -> dict:
    D = 256
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(42)
    q = mx.random.normal([B, H, N, D]).astype(dtype)
    k = mx.random.normal([B, H, N, D]).astype(dtype)
    v = mx.random.normal([B, H, N, D]).astype(dtype)
    mx.eval(q, k, v)

    # Baseline
    sdpa_ms = _measure(lambda: _fallback_sdpa(q, k, v, scale, causal), warmup, iters)

    # V1 route
    with _env("MFA_DISABLE_V2", "1"):
        v1_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), warmup, iters)

    # V2 D-split default
    with _env("MFA_DISABLE_V2", None), _env("MFA_V2_FORCE_BK", None):
        v2_default_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), warmup, iters)

    # Candidate strategy hook: D256 D-split BK override
    with _env("MFA_DISABLE_V2", None), _env("MFA_V2_FORCE_BK", "32"):
        v2_bk32_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), warmup, iters)
    with _env("MFA_DISABLE_V2", None), _env("MFA_V2_FORCE_BK", "64"):
        v2_bk64_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), warmup, iters)

    # Auto dispatch path
    auto_ms = _measure(
        lambda: flash_attention(q, k, v, scale=scale, causal=causal, backend="auto"),
        warmup,
        iters,
    )

    # split-K force toggles (D=256 expected no-op in C++ eligibility)
    with _env("MFA_DISABLE_V2", None), _env("MFA_FORCE_SPLITK", "0"):
        splitk0_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), warmup, iters)
    with _env("MFA_DISABLE_V2", None), _env("MFA_FORCE_SPLITK", "1"):
        splitk1_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), warmup, iters)

    # Occupancy estimate for D256 V2 D-split: BQ=32 by design.
    bq_d256 = 32
    n_q_tiles = (N + bq_d256 - 1) // bq_d256
    total_tgs = n_q_tiles * H * B
    occupancy_ratio = (float(total_tgs) / float(gpu_cores)) if gpu_cores > 0 else 0.0

    policy_use_mfa = should_use_mfa(
        D,
        N,
        causal,
        is_m3_plus,
        dtype=dtype,
        backend="auto",
    )
    auto_path = _infer_auto_path(
        auto_ms=auto_ms,
        sdpa_ms=sdpa_ms,
        v1_ms=v1_ms,
        v2_default_ms=v2_default_ms,
        v2_bk32_ms=v2_bk32_ms,
        v2_bk64_ms=v2_bk64_ms,
        policy_use_mfa=policy_use_mfa,
    )

    ratios = {
        "v1_vs_sdpa": sdpa_ms / v1_ms if v1_ms > 0 else 0.0,
        "v2_default_vs_sdpa": sdpa_ms / v2_default_ms if v2_default_ms > 0 else 0.0,
        "v2_bk32_vs_sdpa": sdpa_ms / v2_bk32_ms if v2_bk32_ms > 0 else 0.0,
        "v2_bk64_vs_sdpa": sdpa_ms / v2_bk64_ms if v2_bk64_ms > 0 else 0.0,
        "auto_vs_sdpa": sdpa_ms / auto_ms if auto_ms > 0 else 0.0,
    }
    best_ratio = max(
        ratios["v1_vs_sdpa"],
        ratios["v2_default_vs_sdpa"],
        ratios["v2_bk32_vs_sdpa"],
        ratios["v2_bk64_vs_sdpa"],
    )

    return {
        "profile": profile_name,
        "B": B,
        "H": H,
        "N": N,
        "D": D,
        "dtype": _dtype_name(dtype),
        "causal": causal,
        "sdpa_ms": sdpa_ms,
        "v1_ms": v1_ms,
        "v2_dsplit_default_ms": v2_default_ms,
        "v2_dsplit_bk32_ms": v2_bk32_ms,
        "v2_dsplit_bk64_ms": v2_bk64_ms,
        "auto_ms": auto_ms,
        "splitk_force0_ms": splitk0_ms,
        "splitk_force1_ms": splitk1_ms,
        "splitk_applicable": False,
        "policy_use_mfa": policy_use_mfa,
        "auto_internal_path_inferred": auto_path,
        "n_q_tiles": n_q_tiles,
        "total_tgs": total_tgs,
        "gpu_cores": gpu_cores,
        "occupancy_ratio": occupancy_ratio,
        **ratios,
        "best_mfa_ratio_vs_sdpa": best_ratio,
        "classification": _row_classification(best_ratio),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="D=256 design-space benchmark matrix")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--output", type=str, default="devnotes/d256_design_matrix_latest.json")
    args = ap.parse_args()

    dev = get_device_info()
    is_m3_plus = bool(dev.get("is_m3_plus", False))
    gpu_cores = int(dev.get("gpu_cores", 0))

    profiles = [
        ("prod_b2h8", 2, 8),
        ("under_b1h1", 1, 1),
    ]
    dtypes = [mx.float16]
    if _dtype_supported(mx.bfloat16):
        dtypes.append(mx.bfloat16)

    rows = []
    for profile_name, B, H in profiles:
        for dtype in dtypes:
            for causal in (False, True):
                for N in (2048, 4096, 8192, 16384):
                    row = run_case(
                        profile_name=profile_name,
                        B=B,
                        H=H,
                        N=N,
                        causal=causal,
                        dtype=dtype,
                        is_m3_plus=is_m3_plus,
                        gpu_cores=gpu_cores,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    rows.append(row)
                    print(
                        f"{profile_name:>10} {_dtype_name(dtype):>4} "
                        f"N={N:<5} causal={causal!s:<5} "
                        f"best={row['best_mfa_ratio_vs_sdpa']:.2f}x "
                        f"auto={row['auto_internal_path_inferred']}"
                    )

    counts = {
        "maybe_win": sum(1 for r in rows if r["classification"] == "maybe_win"),
        "neutral": sum(1 for r in rows if r["classification"] == "neutral"),
        "losing": sum(1 for r in rows if r["classification"] == "losing"),
    }

    print("\nSummary (best MFA route vs SDPA):")
    print(
        f"maybe_win={counts['maybe_win']} "
        f"neutral={counts['neutral']} "
        f"losing={counts['losing']}"
    )

    out = {
        "date": time.strftime("%Y-%m-%d"),
        "mlx_mfa_version": __version__,
        "device": dev,
        "warmup": args.warmup,
        "iters": args.iters,
        "profiles": [p[0] for p in profiles],
        "rows": rows,
        "counts": counts,
        "notes": {
            "d256_splitk_applicable": False,
            "path_compare": [
                "sdpa",
                "mfa_v1",
                "mfa_v2_dsplit_default",
                "mfa_v2_dsplit_bk32",
                "mfa_v2_dsplit_bk64",
                "auto",
            ],
        },
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
