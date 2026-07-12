#!/usr/bin/env python3
"""Gate A for SeedVR2's fp32-to-bf16 varlen boundary.

This is a benchmark-only overlay. It does not import or mutate a SeedVR2 model.
It transposes the active SeedVR2 WindowPartitioner geometry, constructs packed
QKV at the exact T_lat=38 / 27x33 / 20-head / D=128 shapes, and compares:

* current: fp32 packed public API -> per-segment MLX SDPA split-concat
* native:  same QKV cast to bf16 -> public API -> STEEL varlen extension

Run each order in a separate foreground process. Question B is deliberately not
implemented here: the quality gate is allowed to run only after Gate A wins.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np


DEFAULT_SEEDVR2_ROOT = Path(
    "/Users/marcomarcelino/code/ComfyUI-SeedVR2_VideoUpscaler-ORIGINAL"
)
DEFAULT_T_LAT = 38
DEFAULT_PATCH_H = 27
DEFAULT_PATCH_W = 33
DEFAULT_TEXT_TOKENS = 58
DEFAULT_HEADS = 20
DEFAULT_HEAD_DIM = 128
DEFAULT_WINDOW = (4, 3, 3)
DEFAULT_SESSIONS = 5
DEFAULT_WARMUP = 3
DEFAULT_COOLDOWN_S = 1.0


def _stats(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(statistics.median(samples)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(statistics.mean(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "sample_count": len(samples),
        "samples_ms": samples,
    }


def _eval(value: Any) -> None:
    arrays: list[mx.array] = []

    def collect(item: Any) -> None:
        if isinstance(item, mx.array):
            arrays.append(item)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)

    collect(value)
    if arrays:
        mx.eval(*arrays)
    mx.synchronize()


def _time_arm(
    fn: Callable[[], mx.array], *, warmup: int, sessions: int, cooldown_s: float
) -> dict[str, Any]:
    for _ in range(warmup):
        _eval(fn())
    samples: list[float] = []
    for session in range(sessions):
        mx.synchronize()
        start = time.perf_counter()
        _eval(fn())
        samples.append((time.perf_counter() - start) * 1000.0)
        if cooldown_s > 0 and session + 1 < sessions:
            time.sleep(cooldown_s)
    return _stats(samples)


def _setup_seedvr2(root: Path) -> None:
    sys.path.insert(0, str(root / "src" / "mlx_native"))
    sys.path.insert(0, str(root / "src"))


def _geometry(root: Path, shift: bool) -> dict[str, Any]:
    _setup_seedvr2(root)
    from mflux.models.seedvr2.model.seedvr2_transformer.window import (
        WindowPartitioner,
    )

    # Static method is pure Python. Calling the source implementation avoids a
    # second, drifting copy of SeedVR2's window formula in this repository.
    slices = WindowPartitioner._make_windows(
        (DEFAULT_T_LAT, DEFAULT_PATCH_H, DEFAULT_PATCH_W), DEFAULT_WINDOW, shift
    )
    shapes = [
        (st.stop - st.start, sh.stop - sh.start, sw.stop - sw.start)
        for st, sh, sw in slices
    ]
    video_lengths = [math.prod(shape) for shape in shapes]
    segment_lengths = [length + DEFAULT_TEXT_TOKENS for length in video_lengths]
    cumulative = [0]
    tile_offsets = [0]
    for length in segment_lengths:
        cumulative.append(cumulative[-1] + length)
        tile_offsets.append(tile_offsets[-1] + math.ceil(length / 32))
    return {
        "shift": shift,
        "window_count": len(shapes),
        "window_shape_counts": {
            "x".join(map(str, key)): count for key, count in Counter(shapes).items()
        },
        "segment_length_counts": {
            str(key): count for key, count in Counter(segment_lengths).items()
        },
        "video_token_count": sum(video_lengths),
        "packed_token_count": sum(segment_lengths),
        "segment_min": min(segment_lengths),
        "segment_max": max(segment_lengths),
        "segment_lengths": segment_lengths,
        "cu_seqlens": cumulative,
        "tile_offsets": tile_offsets,
        "total_q_tiles": tile_offsets[-1],
    }


def _cosine_and_max_abs(a: mx.array, b: mx.array) -> tuple[float, float]:
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    dot = mx.sum(af * bf)
    denom = mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    max_abs = mx.max(mx.abs(af - bf))
    mx.eval(dot, denom, max_abs)
    denom_value = float(denom.item())
    cosine = float(dot.item()) / denom_value if denom_value else 1.0
    return cosine, float(max_abs.item())


def _per_segment_metrics(
    actual: mx.array, reference: mx.array, cu: list[int]
) -> dict[str, Any]:
    cosines: list[float] = []
    max_abs_values: list[float] = []
    for start, end in zip(cu[:-1], cu[1:]):
        cosine, max_abs = _cosine_and_max_abs(
            actual[:, :, start:end, :], reference[:, :, start:end, :]
        )
        cosines.append(cosine)
        max_abs_values.append(max_abs)
    return {
        "min_cosine": min(cosines),
        "median_cosine": float(statistics.median(cosines)),
        "max_abs": max(max_abs_values),
        "cosines": cosines,
        "max_abs_per_segment": max_abs_values,
    }


def _stamp() -> dict[str, Any]:
    from mlx_mfa import _ext

    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_mfa": importlib.metadata.version("mlx-mfa"),
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "device_info": dict(_ext.get_device_info()),
        "has_nax": bool(_ext.device_has_neural_accelerators()),
    }


def _run(args: argparse.Namespace, geometry: dict[str, Any]) -> dict[str, Any]:
    import mlx_mfa
    from mlx_mfa import _ext

    cu = mx.array(geometry["cu_seqlens"], dtype=mx.int32)
    total = geometry["packed_token_count"]
    mx.random.seed(args.seed)
    qkv_fp32 = (
        mx.random.normal((1, DEFAULT_HEADS, total, 3, DEFAULT_HEAD_DIM)) * 0.1
    ).astype(mx.float32)
    qkv_bf16 = qkv_fp32.astype(mx.bfloat16)
    mx.eval(qkv_fp32, qkv_bf16, cu)

    scale = DEFAULT_HEAD_DIM**-0.5

    def current() -> mx.array:
        return mlx_mfa.flash_attention_varlen_qkv_packed(
            qkv_fp32,
            cu,
            cu,
            geometry["segment_max"],
            geometry["segment_max"],
            scale=scale,
            causal=False,
        )

    def native() -> mx.array:
        return mlx_mfa.flash_attention_varlen_qkv_packed(
            qkv_bf16,
            cu,
            cu,
            geometry["segment_max"],
            geometry["segment_max"],
            scale=scale,
            causal=False,
        )

    # Independent fp32 per-segment oracle, not the public varlen wrapper.
    q = qkv_fp32[:, :, :, 0, :]
    k = qkv_fp32[:, :, :, 1, :]
    v = qkv_fp32[:, :, :, 2, :]
    oracle_parts = [
        mx.fast.scaled_dot_product_attention(
            q[:, :, start:end, :],
            k[:, :, start:end, :],
            v[:, :, start:end, :],
            scale=scale,
        )
        for start, end in zip(geometry["cu_seqlens"][:-1], geometry["cu_seqlens"][1:])
    ]
    oracle = mx.concatenate(oracle_parts, axis=2)
    current_probe = current()
    native_probe = native()
    _eval((oracle, current_probe, native_probe))

    current_cos, current_max_abs = _cosine_and_max_abs(current_probe, oracle)
    native_cos, native_max_abs = _cosine_and_max_abs(native_probe, oracle)
    correctness = {
        "current_vs_fp32_oracle": {
            "cosine": current_cos,
            "max_abs": current_max_abs,
            "per_segment": _per_segment_metrics(
                current_probe, oracle, geometry["cu_seqlens"]
            ),
        },
        "native_bf16_vs_fp32_oracle": {
            "cosine": native_cos,
            "max_abs": native_max_abs,
            "per_segment": _per_segment_metrics(
                native_probe, oracle, geometry["cu_seqlens"]
            ),
        },
    }
    if current_cos < 0.999 or native_cos < 0.999:
        raise RuntimeError(f"correctness gate failed: {correctness}")

    native_calls = 0
    sdpa_calls = 0
    original_native = _ext.mfa_attention_varlen_forward
    original_sdpa = mx.fast.scaled_dot_product_attention

    def counted_native(*call_args, **call_kwargs):
        nonlocal native_calls
        native_calls += 1
        return original_native(*call_args, **call_kwargs)

    def counted_sdpa(*call_args, **call_kwargs):
        nonlocal sdpa_calls
        sdpa_calls += 1
        return original_sdpa(*call_args, **call_kwargs)

    _ext.mfa_attention_varlen_forward = counted_native
    mx.fast.scaled_dot_product_attention = counted_sdpa
    try:
        _eval(current())
        current_engagement = {
            "native_symbol_calls": native_calls,
            "sdpa_calls": sdpa_calls,
        }
        native_calls = 0
        sdpa_calls = 0
        _eval(native())
        native_engagement = {
            "native_symbol_calls": native_calls,
            "sdpa_calls": sdpa_calls,
        }
    finally:
        _ext.mfa_attention_varlen_forward = original_native
        mx.fast.scaled_dot_product_attention = original_sdpa

    if current_engagement != {
        "native_symbol_calls": 0,
        "sdpa_calls": geometry["window_count"],
    }:
        raise RuntimeError(f"current which-binary failed: {current_engagement}")
    if native_engagement != {"native_symbol_calls": 1, "sdpa_calls": 0}:
        raise RuntimeError(f"native which-binary failed: {native_engagement}")

    arms = {"current": current, "native": native}
    order = args.order.split("-")
    timings: dict[str, Any] = {}
    for name in order:
        timings[name] = _time_arm(
            arms[name],
            warmup=args.warmup,
            sessions=args.sessions,
            cooldown_s=args.cooldown,
        )

    ratio = timings["current"]["median_ms"] / timings["native"]["median_ms"]
    return {
        "geometry": geometry,
        "methodology": {
            "order": args.order,
            "sessions_per_arm": args.sessions,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
            "per_arm_sustained": True,
            "same_values_before_dtype_cast": True,
        },
        "stamp": _stamp(),
        "correctness": correctness,
        "engagement": {
            "current": current_engagement,
            "native": native_engagement,
        },
        "timings": timings,
        "speedup_current_over_native": ratio,
        "native_wins": ratio > 1.05,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seedvr2-root", type=Path, default=DEFAULT_SEEDVR2_ROOT)
    parser.add_argument("--shift", choices=("aligned", "shifted"), default="aligned")
    parser.add_argument(
        "--order", choices=("current-native", "native-current"), default="current-native"
    )
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    root = args.seedvr2_root.resolve()
    geometry = _geometry(root, args.shift == "shifted")
    if geometry["video_token_count"] != DEFAULT_T_LAT * DEFAULT_PATCH_H * DEFAULT_PATCH_W:
        raise RuntimeError(f"window partition does not cover video exactly: {geometry}")

    if args.geometry_only:
        result: dict[str, Any] = {
            "schema": "mlx-mfa.seedvr2-varlen-dtype.v1",
            "geometry": geometry,
            "gpu_measurement": "not run (--geometry-only)",
        }
    else:
        result = {
            "schema": "mlx-mfa.seedvr2-varlen-dtype.v1",
            "gate": "A-performance",
            **_run(args, geometry),
        }

    out = args.out
    if out is None:
        suffix = "geometry" if args.geometry_only else args.order
        out = Path("benchmarks/results") / f"seedvr2_varlen_{args.shift}_{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
