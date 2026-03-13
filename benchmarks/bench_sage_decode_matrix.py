#!/usr/bin/env python3
"""Decode-focused Sage benchmark matrix.

Compares decode-time routes on D={64,128} with N_q<=4 and long KV cache:
  1) STEEL dense decode path (flash_attention backend='mfa')
  2) Sage with QuantizedKVCache + sage_attention_prequantized
  3) Sage prequantized direct path (no QuantizedKVCache wrapper)
  4) Sage re-quantize-each-call path (sage_attention)
  5) SDPA fallback reference (where applicable: no window)

Outputs JSON summary for policy decisions around narrow Sage auto-routing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from mlx_mfa import __version__, flash_attention, get_device_info
from mlx_mfa.attention import (
    QuantizedKVCache,
    _ext_available,
    _fallback_sdpa,
    sage_attention,
    sage_attention_prequantized,
)
from mlx_mfa.quantize import quantize_per_block, sage_block_sizes


@dataclass(frozen=True)
class Profile:
    name: str
    B: int
    Hq: int
    Hkv: int


def _dtype_supported(dtype: mx.Dtype) -> bool:
    if dtype == mx.float16:
        return True
    if dtype == mx.bfloat16:
        try:
            x = mx.zeros([1, 1, 1, 64], dtype=mx.bfloat16)
            mx.eval(x)
            return True
        except Exception:
            return False
    return False


def _dtype_name(dtype: mx.Dtype) -> str:
    return "bf16" if dtype == mx.bfloat16 else "f16"


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
    values.sort()
    return float(values[len(values) // 2])


def _expected_dense_path(D: int, Nq: int, N_cache: int) -> str:
    if D in (64, 128) and Nq <= 4 and N_cache >= 256:
        return "flash_decode_expected"
    return "steel_dense_expected"


def _classify(ratio_vs_steel: float) -> str:
    if ratio_vs_steel >= 1.05:
        return "sage_win"
    if ratio_vs_steel >= 0.97:
        return "maybe"
    return "losing"


def run_case(
    *,
    profile: Profile,
    D: int,
    dtype: mx.Dtype,
    Nq: int,
    N_cache: int,
    window_size: Optional[tuple[int, int]],
    warmup: int,
    iters: int,
) -> dict:
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(123)
    q = mx.random.normal([profile.B, profile.Hq, Nq, D]).astype(dtype)
    k = mx.random.normal([profile.B, profile.Hkv, N_cache, D]).astype(dtype)
    v = mx.random.normal([profile.B, profile.Hkv, N_cache, D]).astype(dtype)
    mx.eval(q, k, v)

    dense_ms = _measure(
        lambda: flash_attention(
            q,
            k,
            v,
            scale=scale,
            causal=True,
            window_size=window_size,
            backend="mfa",
        ),
        warmup,
        iters,
    )

    # QuantizedKVCache path (production candidate for decode reuse).
    cache = QuantizedKVCache(
        profile.B,
        profile.Hkv,
        D,
        max_seq_len=N_cache + Nq + 8,
        dtype=dtype,
    )
    cache.append(k, v)
    sage_cache_ms = _measure(
        lambda: sage_attention_prequantized(
            q,
            cache.k_int8,
            cache.k_scale,
            cache.v,
            scale=scale,
            causal=True,
            window_size=window_size,
        ),
        warmup,
        iters,
    )

    # Prequantized direct path (no QuantizedKVCache wrapper).
    _, BK = sage_block_sizes(D)
    k_int8, k_scale = quantize_per_block(k, BK)
    k_scale = k_scale.squeeze(-1)
    mx.eval(k_int8, k_scale)
    sage_preq_direct_ms = _measure(
        lambda: sage_attention_prequantized(
            q,
            k_int8,
            k_scale,
            v,
            scale=scale,
            causal=True,
            window_size=window_size,
        ),
        warmup,
        iters,
    )

    # Re-quantize path (decode-unfriendly baseline).
    sage_requant_ms = _measure(
        lambda: sage_attention(
            q,
            k,
            v,
            scale=scale,
            causal=True,
            window_size=window_size,
        ),
        warmup,
        iters,
    )

    # SDPA reference only for full (non-windowed) decode.
    sdpa_ms: Optional[float] = None
    if window_size is None:
        sdpa_ms = _measure(lambda: _fallback_sdpa(q, k, v, scale, True), warmup, iters)

    ratio_sage_cache_vs_steel = dense_ms / sage_cache_ms if sage_cache_ms > 0 else 0.0
    ratio_sage_direct_vs_steel = dense_ms / sage_preq_direct_ms if sage_preq_direct_ms > 0 else 0.0
    ratio_sage_requant_vs_steel = dense_ms / sage_requant_ms if sage_requant_ms > 0 else 0.0
    ratio_sdpa_vs_steel = (sdpa_ms / dense_ms) if (sdpa_ms is not None and dense_ms > 0) else None
    ratio_sage_cache_vs_sdpa = (sdpa_ms / sage_cache_ms) if (sdpa_ms is not None and sage_cache_ms > 0) else None
    quantized_reuse_speedup = sage_requant_ms / sage_cache_ms if sage_cache_ms > 0 else 0.0

    return {
        "profile": profile.name,
        "B": profile.B,
        "Hq": profile.Hq,
        "Hkv": profile.Hkv,
        "gqa_ratio": (profile.Hq // profile.Hkv),
        "D": D,
        "dtype": _dtype_name(dtype),
        "N_q": Nq,
        "N_cache": N_cache,
        "causal": True,
        "window_size": list(window_size) if window_size is not None else None,
        "dense_internal_path_expected": _expected_dense_path(D, Nq, N_cache),
        "dense_ms": dense_ms,
        "sage_cache_ms": sage_cache_ms,
        "sage_prequant_direct_ms": sage_preq_direct_ms,
        "sage_requant_each_call_ms": sage_requant_ms,
        "sdpa_ms": sdpa_ms,
        "ratio_sage_cache_vs_steel": ratio_sage_cache_vs_steel,
        "ratio_sage_direct_vs_steel": ratio_sage_direct_vs_steel,
        "ratio_sage_requant_vs_steel": ratio_sage_requant_vs_steel,
        "ratio_sdpa_vs_steel": ratio_sdpa_vs_steel,
        "ratio_sage_cache_vs_sdpa": ratio_sage_cache_vs_sdpa,
        "quantized_reuse_speedup": quantized_reuse_speedup,
        "classification": _classify(ratio_sage_cache_vs_steel),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sage decode regime matrix benchmark")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument(
        "--output",
        type=str,
        default="devnotes/sage_decode_matrix_latest.json",
    )
    args = ap.parse_args()

    if not _ext_available():
        raise RuntimeError("MFA extension is required for Sage decode matrix benchmark")

    profiles = [
        Profile("prod_gqa_b2_hq8_hkv4", B=2, Hq=8, Hkv=4),
        Profile("under_b1_hq1_hkv1", B=1, Hq=1, Hkv=1),
    ]
    windows: list[Optional[tuple[int, int]]] = [None, (256, 0)]
    dtypes = [mx.float16]
    if _dtype_supported(mx.bfloat16):
        dtypes.append(mx.bfloat16)

    rows = []
    for profile in profiles:
        for D in (64, 128):
            for dtype in dtypes:
                for Nq in (1, 2, 4):
                    for N_cache in (512, 1024, 2048, 4096, 8192):
                        for window_size in windows:
                            row = run_case(
                                profile=profile,
                                D=D,
                                dtype=dtype,
                                Nq=Nq,
                                N_cache=N_cache,
                                window_size=window_size,
                                warmup=args.warmup,
                                iters=args.iters,
                            )
                            rows.append(row)
                            w = "none" if window_size is None else f"{window_size[0]},{window_size[1]}"
                            print(
                                f"{profile.name:>20} D={D:<3} {_dtype_name(dtype):>4} "
                                f"Nq={Nq} Nc={N_cache:<5} win={w:<7} "
                                f"Sage/STEEL={row['ratio_sage_cache_vs_steel']:.2f}x "
                                f"reuse={row['quantized_reuse_speedup']:.2f}x "
                                f"{row['classification']}"
                            )

    counts = {
        "sage_win": sum(1 for r in rows if r["classification"] == "sage_win"),
        "maybe": sum(1 for r in rows if r["classification"] == "maybe"),
        "losing": sum(1 for r in rows if r["classification"] == "losing"),
    }

    payload = {
        "date": time.strftime("%Y-%m-%d"),
        "mlx_mfa_version": __version__,
        "device": get_device_info(),
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
        "counts": counts,
        "notes": {
            "decode_only": True,
            "profiles": [p.name for p in profiles],
            "windows": [None, [256, 0]],
            "dtypes": [_dtype_name(d) for d in dtypes],
        },
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(
        "\nSummary: "
        f"sage_win={counts['sage_win']} maybe={counts['maybe']} losing={counts['losing']}"
    )
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
