#!/usr/bin/env python3
"""Targeted native-backward viability map for winning forward shapes.

Scope (narrow by design):
  - D in {64, 128}
  - N in {2048, 4096, 8192, 16384}
  - causal=True only
  - f16 + bf16 (bf16 skipped if unsupported)

Compares:
  1) Current production path: flash_attention(..., backend="mfa") VJP
  2) Direct native STEEL backward: mfa_forward_with_lse + mfa_steel_backward
  3) SDPA reference VJP: mx.vjp(_fallback_sdpa, ...)

Reports:
  - total forward+backward wall time
  - native backward-only wall time (SDPA backward-only not separable here)
  - gradient max-abs error vs SDPA reference
  - per-shape classification: promising / neutral / losing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, ".")
import mlx.core as mx

from mlx_mfa import flash_attention, get_device_info
from mlx_mfa.attention import _fallback_sdpa
from mlx_mfa._ext import mfa_forward_with_lse, mfa_steel_backward


def _materialize(value) -> None:
    if isinstance(value, (tuple, list)):
        if value:
            mx.eval(*value)
    else:
        mx.eval(value)
    mx.synchronize()


def _timed_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _materialize(fn())
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _materialize(fn())
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times_ms))


def _to_np32(x: mx.array) -> np.ndarray:
    # numpy does not support bf16; cast within MLX first.
    return np.array(x.astype(mx.float32))


def _max_abs_err(a: mx.array, b: mx.array) -> float:
    return float(np.max(np.abs(_to_np32(a) - _to_np32(b))))


def _dtype_name(dtype: mx.Dtype) -> str:
    if dtype == mx.float16:
        return "f16"
    if dtype == mx.bfloat16:
        return "bf16"
    return str(dtype)


@dataclass(frozen=True)
class ShapeCfg:
    B: int
    H: int
    N: int
    D: int
    dtype: mx.Dtype
    causal: bool = True


def _bf16_supported() -> bool:
    try:
        q = mx.random.normal((1, 1, 32, 64)).astype(mx.bfloat16)
        k = mx.random.normal((1, 1, 32, 64)).astype(mx.bfloat16)
        v = mx.random.normal((1, 1, 32, 64)).astype(mx.bfloat16)
        scale = 1.0 / math.sqrt(64.0)
        out = flash_attention(q, k, v, scale=scale, causal=True, backend="mfa")
        _materialize(out)
        return True
    except Exception:
        return False


def _build_configs(batch: int, heads: int, include_bf16: bool) -> list[ShapeCfg]:
    dtypes = [mx.float16]
    if include_bf16:
        dtypes.append(mx.bfloat16)
    cfgs: list[ShapeCfg] = []
    for dtype in dtypes:
        for D in (64, 128):
            for N in (2048, 4096, 8192, 16384):
                cfgs.append(ShapeCfg(B=batch, H=heads, N=N, D=D, dtype=dtype))
    return cfgs


def _classify(speedup: float, max_abs_err: float, dtype_name: str) -> str:
    tol = 5e-2 if dtype_name == "f16" else 8e-2
    if max_abs_err > tol:
        return "losing"
    if speedup >= 1.05:
        return "promising"
    if speedup >= 0.95:
        return "neutral"
    return "losing"


def _run_shape(cfg: ShapeCfg, warmup: int, iters: int) -> dict:
    B, H, N, D = cfg.B, cfg.H, cfg.N, cfg.D
    dtype = cfg.dtype
    causal = cfg.causal
    scale = 1.0 / math.sqrt(float(D))

    mx.random.seed(42)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    dO = mx.random.normal((B, H, N, D)).astype(dtype)
    _materialize((q, k, v, dO))

    # Precompute once for backward-only native timing.
    O_native, L_native = mfa_forward_with_lse(q, k, v, scale, causal)
    _materialize((O_native, L_native))

    def current_total():
        _, grads = mx.vjp(
            lambda qi, ki, vi: flash_attention(
                qi, ki, vi, scale=scale, causal=causal, backend="mfa"
            ),
            [q, k, v],
            [dO],
        )
        return grads

    def native_total():
        O, L = mfa_forward_with_lse(q, k, v, scale, causal)
        return mfa_steel_backward(q, k, v, O, L, dO, scale, causal)

    def native_bwd_only():
        return mfa_steel_backward(q, k, v, O_native, L_native, dO, scale, causal)

    def sdpa_total():
        _, grads = mx.vjp(
            lambda qi, ki, vi: _fallback_sdpa(qi, ki, vi, scale, causal),
            [q, k, v],
            [dO],
        )
        return grads

    current_total_ms = _timed_ms(current_total, warmup, iters)
    native_total_ms = _timed_ms(native_total, warmup, iters)
    native_bwd_only_ms = _timed_ms(native_bwd_only, warmup, iters)
    sdpa_total_ms = _timed_ms(sdpa_total, warmup, iters)

    # Correctness against SDPA reference gradients.
    dq_ref, dk_ref, dv_ref = sdpa_total()
    dq_nat, dk_nat, dv_nat = native_bwd_only()
    _materialize((dq_ref, dk_ref, dv_ref, dq_nat, dk_nat, dv_nat))

    err_dq = _max_abs_err(dq_nat, dq_ref)
    err_dk = _max_abs_err(dk_nat, dk_ref)
    err_dv = _max_abs_err(dv_nat, dv_ref)
    max_abs_err = max(err_dq, err_dk, err_dv)

    native_vs_sdpa = (sdpa_total_ms / native_total_ms) if native_total_ms > 0 else float("nan")
    current_vs_sdpa = (sdpa_total_ms / current_total_ms) if current_total_ms > 0 else float("nan")
    cls = _classify(native_vs_sdpa, max_abs_err, _dtype_name(dtype))

    return {
        "B": B,
        "H": H,
        "N": N,
        "D": D,
        "dtype": _dtype_name(dtype),
        "causal": causal,
        "current_total_ms": current_total_ms,
        "native_total_ms": native_total_ms,
        "native_backward_only_ms": native_bwd_only_ms,
        "sdpa_total_ms": sdpa_total_ms,
        "native_vs_sdpa": native_vs_sdpa,
        "current_vs_sdpa": current_vs_sdpa,
        "max_abs_err": max_abs_err,
        "max_abs_err_dq": err_dq,
        "max_abs_err_dk": err_dk,
        "max_abs_err_dv": err_dv,
        "classification": cls,
    }


def _print_table(rows: Iterable[dict]) -> None:
    print(
        f"{'dtype':<5} {'D':>4} {'N':>6} "
        f"{'current':>10} {'native':>10} {'sdpa':>10} "
        f"{'nv/sdpa':>8} {'err':>9} {'class':>10}"
    )
    print("-" * 90)
    for r in rows:
        if "error" in r:
            print(
                f"{r['dtype']:<5} {r['D']:>4} {r['N']:>6} "
                f"{'nan':>10} {'nan':>10} {'nan':>10} "
                f"{'nan':>8} {'nan':>9} {'error':>10}"
            )
            continue
        print(
            f"{r['dtype']:<5} {r['D']:>4} {r['N']:>6} "
            f"{r['current_total_ms']:>9.2f} "
            f"{r['native_total_ms']:>9.2f} "
            f"{r['sdpa_total_ms']:>9.2f} "
            f"{r['native_vs_sdpa']:>8.2f} "
            f"{r['max_abs_err']:>9.3e} "
            f"{r['classification']:>10}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument(
        "--output",
        type=str,
        default="notes/native_backward_targeted_latest.json",
    )
    args = ap.parse_args()

    bf16_ok = _bf16_supported()
    cfgs = _build_configs(args.batch, args.heads, include_bf16=bf16_ok)
    dev = get_device_info()

    print(f"Native backward targeted benchmark — {date.today()}")
    print(
        f"Device={dev.get('device_name','?')} gen={dev.get('gpu_family_gen','?')} "
        f"B={args.batch} H={args.heads} warmup={args.warmup} iters={args.iters}"
    )
    print(f"bf16_supported={bf16_ok}")
    print()

    rows = []
    total = len(cfgs)
    for idx, cfg in enumerate(cfgs, start=1):
        label = f"{_dtype_name(cfg.dtype)} D={cfg.D} N={cfg.N}"
        try:
            row = _run_shape(cfg, warmup=args.warmup, iters=args.iters)
            rows.append(row)
            print(
                f"[{idx:02d}/{total:02d}] {label}: "
                f"native={row['native_total_ms']:.2f}ms "
                f"sdpa={row['sdpa_total_ms']:.2f}ms "
                f"ratio={row['native_vs_sdpa']:.2f} "
                f"class={row['classification']}"
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "B": cfg.B,
                    "H": cfg.H,
                    "N": cfg.N,
                    "D": cfg.D,
                    "dtype": _dtype_name(cfg.dtype),
                    "causal": cfg.causal,
                    "current_total_ms": float("nan"),
                    "native_total_ms": float("nan"),
                    "native_backward_only_ms": float("nan"),
                    "sdpa_total_ms": float("nan"),
                    "native_vs_sdpa": float("nan"),
                    "current_vs_sdpa": float("nan"),
                    "max_abs_err": float("nan"),
                    "max_abs_err_dq": float("nan"),
                    "max_abs_err_dk": float("nan"),
                    "max_abs_err_dv": float("nan"),
                    "classification": "losing",
                    "error": str(exc),
                }
            )
            print(f"[{idx:02d}/{total:02d}] {label}: ERROR {exc}")

    _print_table(rows)
    print()

    counts = {
        "promising": sum(1 for r in rows if r["classification"] == "promising"),
        "neutral": sum(1 for r in rows if r["classification"] == "neutral"),
        "losing": sum(1 for r in rows if r["classification"] == "losing"),
    }
    print(
        f"Classification counts: promising={counts['promising']} "
        f"neutral={counts['neutral']} losing={counts['losing']}"
    )

    payload = {
        "date": str(date.today()),
        "device": dev,
        "batch": args.batch,
        "heads": args.heads,
        "warmup": args.warmup,
        "iters": args.iters,
        "bf16_supported": bf16_ok,
        "results": rows,
        "counts": counts,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
