#!/usr/bin/env python3
"""D=512 decision-pass benchmark matrix (benchmark-backed production status).

This benchmark intentionally treats D=512 as a separate decision family and
compares only currently wired routes:
  - SDPA baseline
  - MFA V1 fallback (`MFA_DISABLE_V2=1`)
  - MFA V2 D-split default
  - MFA with V5 opt-in (`MFA_ENABLE_V5=1`) to verify whether V5 can compete
    on D=512 (expected to fall through because V5 eligibility is D in {64,128})
  - `flash_attention(..., backend="auto")`

Each timed route is run in a separate subprocess to isolate shader caches and
environment toggles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import mlx.core as mx

from mlx_mfa import __version__, get_device_info
from mlx_mfa.dispatch_policy import should_use_mfa


D512 = 512
ROUTES = ("sdpa", "mfa_v1", "mfa_v2_dsplit", "mfa_v5_optin", "auto")


@dataclass(frozen=True)
class Profile:
    name: str
    batch: int
    heads: int


def _dtype_supported(dtype: mx.Dtype) -> bool:
    if dtype == mx.float16:
        return True
    if dtype == mx.bfloat16:
        try:
            q = mx.random.normal([1, 1, 16, 64]).astype(mx.bfloat16)
            k = mx.random.normal([1, 1, 16, 64]).astype(mx.bfloat16)
            v = mx.random.normal([1, 1, 16, 64]).astype(mx.bfloat16)
            mx.eval(q, k, v)
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


def _median_ms(times_ms: list[float]) -> float:
    times_ms.sort()
    return float(times_ms[len(times_ms) // 2])


def _measure(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return _median_ms(times_ms)


def _prepare_env_for_route(route: str) -> None:
    os.environ.pop("MFA_DISABLE_V2", None)
    os.environ.pop("MFA_ENABLE_V5", None)

    if route == "mfa_v1":
        os.environ["MFA_DISABLE_V2"] = "1"
    elif route == "mfa_v2_dsplit":
        pass
    elif route == "mfa_v5_optin":
        os.environ["MFA_ENABLE_V5"] = "1"
    elif route in {"sdpa", "auto"}:
        pass
    else:
        raise ValueError(f"Unknown route: {route}")


def _run_subprocess_mode(args: argparse.Namespace) -> None:
    from mlx_mfa import flash_attention
    from mlx_mfa.attention import _fallback_sdpa, _mfa_forward

    route = args.route
    dtype = mx.float16 if args.dtype == "f16" else mx.bfloat16
    causal = bool(args.causal)
    B, H, N, D = args.batch, args.heads, args.seq_len, D512
    scale = 1.0 / math.sqrt(D)

    _prepare_env_for_route(route)

    mx.random.seed(args.seed)
    q = mx.random.normal([B, H, N, D]).astype(dtype)
    k = mx.random.normal([B, H, N, D]).astype(dtype)
    v = mx.random.normal([B, H, N, D]).astype(dtype)
    mx.eval(q, k, v)

    if route == "sdpa":
        fn = lambda: _fallback_sdpa(q, k, v, scale, causal)
    elif route in {"mfa_v1", "mfa_v2_dsplit", "mfa_v5_optin"}:
        fn = lambda: _mfa_forward(q, k, v, scale, causal)
    else:
        fn = lambda: flash_attention(q, k, v, scale=scale, causal=causal, backend="auto")

    ms = _measure(fn, warmup=args.warmup, iters=args.iters)
    print(f"{ms:.6f}")


def _run_route_subprocess(
    *,
    route: str,
    dtype_name: str,
    profile: Profile,
    causal: bool,
    seq_len: int,
    warmup: int,
    iters: int,
    seed: int,
) -> float:
    cmd = [
        sys.executable,
        __file__,
        "--subprocess-mode",
        "--route",
        route,
        "--dtype",
        dtype_name,
        "--batch",
        str(profile.batch),
        "--heads",
        str(profile.heads),
        "--seq-len",
        str(seq_len),
        "--causal",
        "1" if causal else "0",
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(
            f"route={route} failed for profile={profile.name} N={seq_len} "
            f"causal={causal} dtype={dtype_name}\nstdout={stdout}\nstderr={stderr}"
        )
    return float(proc.stdout.strip())


def _classification(best_ratio_vs_sdpa: float) -> str:
    if best_ratio_vs_sdpa >= 1.02:
        return "maybe_win"
    if best_ratio_vs_sdpa >= 0.95:
        return "no_win"
    return "losing"


def _auto_path_label(policy_use_mfa: bool) -> str:
    if policy_use_mfa:
        return "mfa(policy)->steel_v2_dsplit512"
    return "sdpa(policy)"


def _route_path_label(route: str, policy_use_mfa: bool | None = None) -> str:
    if route == "sdpa":
        return "sdpa"
    if route == "mfa_v1":
        return "steel_v1_dense"
    if route == "mfa_v2_dsplit":
        return "steel_v2_dsplit512"
    if route == "mfa_v5_optin":
        return "v5_ineligible_for_d512->steel_v2_dsplit512"
    if route == "auto":
        assert policy_use_mfa is not None
        return _auto_path_label(policy_use_mfa)
    return "unknown"


def run_matrix(args: argparse.Namespace) -> dict:
    dev = get_device_info()
    is_m3_plus = bool(dev.get("is_m3_plus", False))
    gpu_cores = int(dev.get("gpu_cores", 0))

    profiles = [
        Profile("prod_b2h8", batch=2, heads=8),
        Profile("under_b1h1", batch=1, heads=1),
    ]

    dtypes: list[mx.Dtype] = [mx.float16]
    if _dtype_supported(mx.bfloat16):
        dtypes.append(mx.bfloat16)

    rows = []
    for profile in profiles:
        for dtype in dtypes:
            dtype_key = _dtype_name(dtype)
            for causal in (False, True):
                for seq_len in (1024, 2048, 4096, 8192):
                    policy_use_mfa = should_use_mfa(
                        D512,
                        seq_len,
                        causal,
                        is_m3_plus,
                        dtype=dtype,
                        backend="auto",
                    )

                    timings = {}
                    for route in ROUTES:
                        timings[route] = _run_route_subprocess(
                            route=route,
                            dtype_name=dtype_key,
                            profile=profile,
                            causal=causal,
                            seq_len=seq_len,
                            warmup=args.warmup,
                            iters=args.iters,
                            seed=args.seed,
                        )

                    sdpa_ms = timings["sdpa"]
                    ratios = {
                        "mfa_v1_vs_sdpa": sdpa_ms / timings["mfa_v1"] if timings["mfa_v1"] > 0 else 0.0,
                        "mfa_v2_dsplit_vs_sdpa": sdpa_ms / timings["mfa_v2_dsplit"]
                        if timings["mfa_v2_dsplit"] > 0
                        else 0.0,
                        "mfa_v5_optin_vs_sdpa": sdpa_ms / timings["mfa_v5_optin"]
                        if timings["mfa_v5_optin"] > 0
                        else 0.0,
                        "auto_vs_sdpa": sdpa_ms / timings["auto"] if timings["auto"] > 0 else 0.0,
                    }
                    best_ratio = max(
                        ratios["mfa_v1_vs_sdpa"],
                        ratios["mfa_v2_dsplit_vs_sdpa"],
                        ratios["mfa_v5_optin_vs_sdpa"],
                    )

                    bq_dsplit = 32
                    n_q_tiles = (seq_len + bq_dsplit - 1) // bq_dsplit
                    total_tgs = n_q_tiles * profile.heads * profile.batch
                    occupancy_ratio = (float(total_tgs) / float(gpu_cores)) if gpu_cores > 0 else 0.0

                    row = {
                        "profile": profile.name,
                        "B": profile.batch,
                        "H": profile.heads,
                        "N": seq_len,
                        "D": D512,
                        "dtype": dtype_key,
                        "causal": causal,
                        "policy_use_mfa": policy_use_mfa,
                        "path_sdpa": _route_path_label("sdpa"),
                        "path_mfa_v1": _route_path_label("mfa_v1"),
                        "path_mfa_v2_dsplit": _route_path_label("mfa_v2_dsplit"),
                        "path_mfa_v5_optin": _route_path_label("mfa_v5_optin"),
                        "path_auto": _route_path_label("auto", policy_use_mfa=policy_use_mfa),
                        "sdpa_ms": sdpa_ms,
                        "mfa_v1_ms": timings["mfa_v1"],
                        "mfa_v2_dsplit_ms": timings["mfa_v2_dsplit"],
                        "mfa_v5_optin_ms": timings["mfa_v5_optin"],
                        "auto_ms": timings["auto"],
                        **ratios,
                        "best_mfa_ratio_vs_sdpa": best_ratio,
                        "classification": _classification(best_ratio),
                        "n_q_tiles": n_q_tiles,
                        "total_tgs": total_tgs,
                        "gpu_cores": gpu_cores,
                        "occupancy_ratio": occupancy_ratio,
                    }
                    rows.append(row)

                    print(
                        f"{profile.name:>10} {dtype_key:>4} N={seq_len:<5} causal={causal!s:<5} "
                        f"best={best_ratio:.2f}x auto={row['path_auto']} "
                        f"v2={ratios['mfa_v2_dsplit_vs_sdpa']:.2f}x"
                    )

    counts = {
        "maybe_win": sum(1 for r in rows if r["classification"] == "maybe_win"),
        "no_win": sum(1 for r in rows if r["classification"] == "no_win"),
        "losing": sum(1 for r in rows if r["classification"] == "losing"),
    }

    return {
        "date": time.strftime("%Y-%m-%d"),
        "mlx_mfa_version": __version__,
        "device": dev,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "profiles": [p.__dict__ for p in profiles],
        "rows": rows,
        "counts": counts,
        "notes": {
            "decision_family": "D=512",
            "v5_d512_support": "not eligible in current kernel (D in {64,128})",
            "routes": list(ROUTES),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="D=512 decision-pass benchmark matrix")
    ap.add_argument("--subprocess-mode", action="store_true")
    ap.add_argument("--route", type=str, default="sdpa")
    ap.add_argument("--dtype", type=str, default="f16", choices=["f16", "bf16"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        type=str,
        default="devnotes/d512_decision_matrix_latest.json",
    )
    args = ap.parse_args()

    if args.subprocess_mode:
        _run_subprocess_mode(args)
        return

    results = run_matrix(args)

    print("\nSummary (best MFA route vs SDPA):")
    print(
        f"maybe_win={results['counts']['maybe_win']} "
        f"no_win={results['counts']['no_win']} "
        f"losing={results['counts']['losing']}"
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
