#!/usr/bin/env python3
"""Experimental-path triage matrix + advanced-kernel cold-start probes.

This pass benchmarks experimental forward routes (V3/V4/V5) against V2/SDPA,
and measures first-call compile overhead for selected advanced kernels that are
possible AOT candidates.

All timed runs execute in separate subprocesses to avoid ShaderCache bleed.
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
from typing import Any, Optional

import mlx.core as mx

from mlx_mfa import __version__, get_device_info


ROUTES = ("sdpa", "v2", "v3", "v4", "v5")
COLDSTART_CANDIDATES = (
    "sage_decode_d128_gqa2",
    "paged_gather_d128",
    "paged_steel_d128",
)


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
            q = mx.random.normal([1, 1, 8, 64]).astype(mx.bfloat16)
            mx.eval(q)
            return True
        except Exception:
            return False
    return False


def _dtype_name(dtype: mx.Dtype) -> str:
    return "bf16" if dtype == mx.bfloat16 else "f16"


def _median_ms(values: list[float]) -> float:
    values.sort()
    return float(values[len(values) // 2])


def _measure(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()

    values = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        values.append((time.perf_counter() - t0) * 1000.0)
    return _median_ms(values)


def _prepare_route_env(route: str) -> None:
    for key in (
        "MFA_DISABLE_V2",
        "MFA_ENABLE_V3",
        "MFA_ENABLE_V4",
        "MFA_ENABLE_V5",
        "MFA_FORCE_GEN",
    ):
        os.environ.pop(key, None)

    if route == "sdpa":
        return
    if route == "v2":
        return
    if route == "v3":
        os.environ["MFA_ENABLE_V3"] = "1"
        return
    if route == "v4":
        os.environ["MFA_ENABLE_V4"] = "1"
        return
    if route == "v5":
        os.environ["MFA_ENABLE_V5"] = "1"
        return
    if route == "v4_sim_m3":
        os.environ["MFA_ENABLE_V4"] = "1"
        os.environ["MFA_FORCE_GEN"] = "15"
        return
    raise ValueError(f"unknown route: {route}")


def _expected_path(route: str, *, D: int, is_m3_plus: bool, simulated_m3: bool = False) -> str:
    if route == "sdpa":
        return "sdpa"
    if route == "v2":
        return "steel_v2"
    if route == "v3":
        if D == 64 or (D == 128 and not is_m3_plus):
            return "steel_v3"
        return "steel_v2_fallback(v3_ineligible)"
    if route == "v4":
        if is_m3_plus and D in (64, 128):
            return "steel_v4"
        return "steel_v2_fallback(v4_non_m3_or_ineligible)"
    if route == "v4_sim_m3":
        if simulated_m3 and D in (64, 128):
            return "steel_v4(simulated_m3)"
        return "steel_v2_fallback(v4_sim_ineligible)"
    if route == "v5":
        if D in (64, 128):
            return "steel_v5"
        return "steel_v2_fallback(v5_ineligible)"
    return "unknown"


def _classify_ratio(*, ratio_vs_v2: float, ratio_vs_sdpa: float) -> str:
    if ratio_vs_v2 >= 1.03 and ratio_vs_sdpa >= 1.0:
        return "clear_win"
    if ratio_vs_v2 >= 0.97 and ratio_vs_sdpa >= 0.95:
        return "neutral"
    return "losing"


def _classify_route(
    *,
    expected_path: str,
    ratio_vs_v2: float,
    ratio_vs_sdpa: float,
) -> str:
    if "fallback" in expected_path:
        return "ineligible"
    return _classify_ratio(ratio_vs_v2=ratio_vs_v2, ratio_vs_sdpa=ratio_vs_sdpa)


def _run_subprocess_route(args: argparse.Namespace) -> None:
    from mlx_mfa.attention import _fallback_sdpa, _mfa_forward

    route = args.route
    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float16
    causal = bool(args.causal)
    B, H, N, D = args.batch, args.heads, args.seq_len, args.head_dim
    scale = 1.0 / math.sqrt(D)

    _prepare_route_env(route)

    mx.random.seed(args.seed)
    q = mx.random.normal([B, H, N, D]).astype(dtype)
    k = mx.random.normal([B, H, N, D]).astype(dtype)
    v = mx.random.normal([B, H, N, D]).astype(dtype)
    mx.eval(q, k, v)

    if route == "sdpa":
        fn = lambda: _fallback_sdpa(q, k, v, scale, causal)
    else:
        fn = lambda: _mfa_forward(q, k, v, scale, causal)

    ms = _measure(fn, warmup=args.warmup, iters=args.iters)
    print(f"{ms:.6f}")


def _run_subprocess_coldstart(args: argparse.Namespace) -> None:
    import mlx.core as mx_local

    dtype = mx_local.float16
    candidate = args.candidate

    def _first_and_steady(fn):
        mx_local.synchronize()
        t0 = time.perf_counter()
        mx_local.eval(fn())
        mx_local.synchronize()
        first_ms = (time.perf_counter() - t0) * 1000.0
        steady_ms = _measure(fn, warmup=args.warmup, iters=args.iters)
        return first_ms, steady_ms

    if candidate == "sage_decode_d128_gqa2":
        from mlx_mfa.attention import sage_attention_prequantized
        from mlx_mfa.quantize import quantize_per_block

        B, Hq, Hkv, D, Nq, S = 2, 8, 4, 128, 1, 4096
        scale = 1.0 / math.sqrt(D)
        q = mx_local.random.normal([B, Hq, Nq, D]).astype(dtype)
        k = mx_local.random.normal([B, Hkv, S, D]).astype(dtype)
        v = mx_local.random.normal([B, Hkv, S, D]).astype(dtype)
        k_i8, k_s = quantize_per_block(k, block_size=32)
        k_s = k_s.squeeze(-1)
        mx_local.eval(q, k_i8, k_s, v)

        first_ms, steady_ms = _first_and_steady(
            lambda: sage_attention_prequantized(q, k_i8, k_s, v, scale=scale, causal=True)
        )
    elif candidate == "paged_gather_d128":
        from mlx_mfa._ext import mfa_paged_kv_gather

        B, H, D = 1, 4, 128
        block_size = 16
        max_kv_len = 4096
        max_blocks = max_kv_len // block_size
        num_blocks = max_blocks + 32

        pool = mx_local.random.normal([num_blocks, block_size, H, D]).astype(dtype)
        table = mx_local.array([list(range(max_blocks))], dtype=mx_local.int32)
        lens = mx_local.array([max_kv_len], dtype=mx_local.int32)
        mx_local.eval(pool, table, lens)

        first_ms, steady_ms = _first_and_steady(
            lambda: mfa_paged_kv_gather(pool, table, lens, max_kv_len)
        )
    elif candidate == "paged_steel_d128":
        from mlx_mfa._ext import mfa_paged_steel_forward

        B, Hq, Hkv, D = 1, 8, 4, 128
        Nq, S = 16, 4096
        block_size = 16
        max_blocks = S // block_size
        num_blocks = max_blocks + 16
        scale = 1.0 / math.sqrt(D)

        q = mx_local.random.normal([B, Hq, Nq, D]).astype(dtype)
        pool_k = mx_local.random.normal([num_blocks, block_size, Hkv, D]).astype(dtype)
        pool_v = mx_local.random.normal([num_blocks, block_size, Hkv, D]).astype(dtype)
        table = mx_local.array([list(range(max_blocks))], dtype=mx_local.int32)
        lens = mx_local.array([S], dtype=mx_local.int32)
        mx_local.eval(q, pool_k, pool_v, table, lens)

        first_ms, steady_ms = _first_and_steady(
            lambda: mfa_paged_steel_forward(
                q,
                pool_k,
                pool_v,
                table,
                lens,
                scale,
                True,
                -1,
                -1,
                block_size,
            )[0]
        )
    else:
        raise ValueError(f"unknown candidate: {candidate}")

    payload = {
        "candidate": candidate,
        "first_call_ms": first_ms,
        "steady_ms": steady_ms,
        "first_over_steady": first_ms / steady_ms if steady_ms > 0 else 0.0,
    }
    print(json.dumps(payload))


def _run_route_subprocess(
    *,
    route: str,
    dtype_name: str,
    profile: Profile,
    D: int,
    N: int,
    causal: bool,
    warmup: int,
    iters: int,
    seed: int,
) -> float:
    cmd = [
        sys.executable,
        __file__,
        "--subprocess-mode",
        "route",
        "--route",
        route,
        "--dtype",
        dtype_name,
        "--batch",
        str(profile.batch),
        "--heads",
        str(profile.heads),
        "--head-dim",
        str(D),
        "--seq-len",
        str(N),
        "--causal",
        "1" if causal else "0",
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(
            f"route={route} failed D={D} N={N} causal={causal} profile={profile.name}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return float(proc.stdout.strip())


def _run_coldstart_subprocess(*, candidate: str, warmup: int, iters: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        __file__,
        "--subprocess-mode",
        "coldstart",
        "--candidate",
        candidate,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(
            f"candidate={candidate} failed\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return json.loads(proc.stdout.strip())


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    dev = get_device_info()
    is_m3_plus = bool(dev.get("is_m3_plus", False))

    profiles = [
        Profile("prod_b2h8", batch=2, heads=8),
        Profile("under_b1h1", batch=1, heads=1),
    ]

    dtypes: list[mx.Dtype] = [mx.float16]
    if args.include_bf16 and _dtype_supported(mx.bfloat16):
        dtypes.append(mx.bfloat16)

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for dtype in dtypes:
            dtype_key = _dtype_name(dtype)
            for D in (64, 128):
                for N in (2048, 8192):
                    for causal in (False, True):
                        timings: dict[str, float] = {}
                        for route in ROUTES:
                            timings[route] = _run_route_subprocess(
                                route=route,
                                dtype_name=dtype_key,
                                profile=profile,
                                D=D,
                                N=N,
                                causal=causal,
                                warmup=args.warmup,
                                iters=args.iters,
                                seed=args.seed,
                            )

                        sdpa_ms = timings["sdpa"]
                        v2_ms = timings["v2"]
                        for route in ("v3", "v4", "v5"):
                            ratio_vs_sdpa = sdpa_ms / timings[route] if timings[route] > 0 else 0.0
                            ratio_vs_v2 = v2_ms / timings[route] if timings[route] > 0 else 0.0
                            expected_path = _expected_path(
                                route,
                                D=D,
                                is_m3_plus=is_m3_plus,
                            )
                            row = {
                                "profile": profile.name,
                                "B": profile.batch,
                                "H": profile.heads,
                                "D": D,
                                "N": N,
                                "causal": causal,
                                "dtype": dtype_key,
                                "route": route,
                                "expected_path": expected_path,
                                "sdpa_ms": sdpa_ms,
                                "v2_ms": v2_ms,
                                "experimental_ms": timings[route],
                                "ratio_vs_sdpa": ratio_vs_sdpa,
                                "ratio_vs_v2": ratio_vs_v2,
                                "classification": _classify_route(
                                    expected_path=expected_path,
                                    ratio_vs_v2=ratio_vs_v2,
                                    ratio_vs_sdpa=ratio_vs_sdpa,
                                ),
                            }
                            rows.append(row)
                            print(
                                f"{route:>2} {profile.name:>9} D={D:<3} N={N:<5} causal={causal!s:<5} "
                                f"vs_v2={ratio_vs_v2:.2f}x class={row['classification']}"
                            )

    # One V4 simulated-M3 probe to capture hardware-specific potential note.
    v4_sim_probe: Optional[dict[str, Any]] = None
    try:
        sim_profile = profiles[0]
        sim_dtype = _dtype_name(dtypes[0])
        sdpa_ms = _run_route_subprocess(
            route="sdpa",
            dtype_name=sim_dtype,
            profile=sim_profile,
            D=128,
            N=4096,
            causal=True,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )
        v2_ms = _run_route_subprocess(
            route="v2",
            dtype_name=sim_dtype,
            profile=sim_profile,
            D=128,
            N=4096,
            causal=True,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )
        v4_sim_ms = _run_route_subprocess(
            route="v4_sim_m3",
            dtype_name=sim_dtype,
            profile=sim_profile,
            D=128,
            N=4096,
            causal=True,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )
        v4_sim_probe = {
            "profile": sim_profile.name,
            "D": 128,
            "N": 4096,
            "causal": True,
            "dtype": sim_dtype,
            "expected_path": _expected_path("v4_sim_m3", D=128, is_m3_plus=True, simulated_m3=True),
            "sdpa_ms": sdpa_ms,
            "v2_ms": v2_ms,
            "v4_sim_m3_ms": v4_sim_ms,
            "ratio_vs_sdpa": sdpa_ms / v4_sim_ms if v4_sim_ms > 0 else 0.0,
            "ratio_vs_v2": v2_ms / v4_sim_ms if v4_sim_ms > 0 else 0.0,
            "classification": _classify_ratio(
                ratio_vs_v2=v2_ms / v4_sim_ms if v4_sim_ms > 0 else 0.0,
                ratio_vs_sdpa=sdpa_ms / v4_sim_ms if v4_sim_ms > 0 else 0.0,
            ),
            "note": "simulated_m3_only (MFA_FORCE_GEN=15)",
        }
        print(
            "v4_sim probe D=128 N=4096 causal=True "
            f"vs_v2={v4_sim_probe['ratio_vs_v2']:.2f}x"
        )
    except Exception as exc:
        v4_sim_probe = {"error": str(exc)}

    summary_by_route = {}
    for route in ("v3", "v4", "v5"):
        subset = [r for r in rows if r["route"] == route]
        active_subset = [r for r in subset if r["classification"] != "ineligible"]
        summary_by_route[route] = {
            "ineligible": sum(1 for r in subset if r["classification"] == "ineligible"),
            "clear_win": sum(1 for r in active_subset if r["classification"] == "clear_win"),
            "neutral": sum(1 for r in active_subset if r["classification"] == "neutral"),
            "losing": sum(1 for r in active_subset if r["classification"] == "losing"),
            "best_ratio_vs_v2": max((r["ratio_vs_v2"] for r in active_subset), default=0.0),
            "best_ratio_vs_sdpa": max((r["ratio_vs_sdpa"] for r in active_subset), default=0.0),
        }

    coldstarts = []
    for candidate in COLDSTART_CANDIDATES:
        row = _run_coldstart_subprocess(candidate=candidate, warmup=args.cold_warmup, iters=args.cold_iters)
        coldstarts.append(row)
        print(
            f"coldstart {candidate:>22} first={row['first_call_ms']:.2f}ms "
            f"steady={row['steady_ms']:.2f}ms x{row['first_over_steady']:.1f}"
        )

    return {
        "date": time.strftime("%Y-%m-%d"),
        "mlx_mfa_version": __version__,
        "device": dev,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "rows": rows,
        "summary_by_route": summary_by_route,
        "v4_sim_m3_probe": v4_sim_probe,
        "coldstart_candidates": coldstarts,
        "notes": {
            "experimental_routes": ["v3", "v4", "v5"],
            "v4_hardware_dependency": "M3+ only (simulated probe uses MFA_FORCE_GEN=15)",
            "coldstart_candidates": list(COLDSTART_CANDIDATES),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimental path triage benchmark")
    ap.add_argument("--subprocess-mode", nargs=1, choices=["route", "coldstart"])

    ap.add_argument("--route", type=str, default="sdpa")
    ap.add_argument("--candidate", type=str, default=COLDSTART_CANDIDATES[0])
    ap.add_argument("--dtype", type=str, default="f16", choices=["f16", "bf16"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--cold-warmup", type=int, default=1)
    ap.add_argument("--cold-iters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-bf16", action="store_true")
    ap.add_argument(
        "--output",
        type=str,
        default="notes/experimental_path_triage_latest.json",
    )
    args = ap.parse_args()

    if args.subprocess_mode:
        mode = args.subprocess_mode[0]
        if mode == "route":
            _run_subprocess_route(args)
        else:
            _run_subprocess_coldstart(args)
        return

    results = run_matrix(args)
    print("\nSummary by route:")
    for route, values in results["summary_by_route"].items():
        print(
            f"{route}: win={values['clear_win']} neutral={values['neutral']} "
            f"losing={values['losing']} best_vs_v2={values['best_ratio_vs_v2']:.2f}x"
        )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
