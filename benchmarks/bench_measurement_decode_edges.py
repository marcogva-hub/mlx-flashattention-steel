#!/usr/bin/env python3
"""Consolidate decode-edge measurements with explicit engagement gates."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx_mfa
from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import flash_attention


REPO = Path(__file__).resolve().parents[1]
H_Q = 16
DTYPES = ("fp16", "bf16")
K_LENGTHS = (16384, 32768, 65536)
SESSIONS = 5
SAMPLES_PER_SESSION = 5
WARMUP_SAMPLES = 2
DISPATCHES_PER_SAMPLE = 20


@dataclass(frozen=True)
class Shape:
    family: str
    q_len: int
    k_len: int
    gqa: int
    head_dim: int
    dtype: str
    causal: bool

    @property
    def h_kv(self) -> int:
        return H_Q // self.gqa


def _boundary_shapes() -> list[Shape]:
    shapes: list[Shape] = []
    for k_len in K_LENGTHS:
        for dtype in DTYPES:
            shapes.append(Shape("q16", 16, k_len, 8, 64, dtype, False))
            for gqa in (4, 16):
                shapes.append(Shape(f"gqa{gqa}", 8, k_len, gqa, 64, dtype, False))
            shapes.append(Shape("d128", 8, k_len, 8, 128, dtype, False))
            shapes.append(Shape("causal", 8, k_len, 8, 64, dtype, True))
    return shapes


def _cross_shapes() -> list[Shape]:
    return [
        Shape(f"q16_gqa{gqa}", 16, k_len, gqa, 64, dtype, False)
        for gqa in (4, 16)
        for k_len in K_LENGTHS
        for dtype in DTYPES
    ]


def _dtype(name: str) -> mx.Dtype:
    return mx.float16 if name == "fp16" else mx.bfloat16


def _make_qkv(shape: Shape):
    seed = (920000 + shape.q_len * 31 + shape.k_len + shape.gqa * 17
            + shape.head_dim * 11 + int(shape.causal) * 7
            + (1 if shape.dtype == "bf16" else 0))
    mx.random.seed(seed)
    dtype = _dtype(shape.dtype)
    q = (mx.random.normal((1, H_Q, shape.q_len, shape.head_dim)) * 0.25).astype(dtype)
    k = (mx.random.normal((1, shape.h_kv, shape.k_len, shape.head_dim)) * 0.25).astype(dtype)
    v = (mx.random.normal((1, shape.h_kv, shape.k_len, shape.head_dim)) * 0.25).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _call(q, k, v, shape: Shape, backend: str):
    return flash_attention(
        q, k, v, scale=1.0 / math.sqrt(shape.head_dim),
        causal=shape.causal, backend=backend,
    )


def _reference(q, k, v, shape: Shape):
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=1.0 / math.sqrt(shape.head_dim),
        mask="causal" if shape.causal else None,
    )


def _metrics(value, reference) -> dict[str, float | bool]:
    got = np.asarray(value.astype(mx.float32)).reshape(-1).astype(np.float64)
    ref = np.asarray(reference.astype(mx.float32)).reshape(-1).astype(np.float64)
    denom = np.linalg.norm(got) * np.linalg.norm(ref)
    return {
        "cos": float(np.dot(got, ref) / denom) if denom else 1.0,
        "max_abs": float(np.max(np.abs(got - ref))),
        "finite": bool(np.isfinite(got).all()),
    }


def _trace(fn) -> tuple[object, list[list[str]]]:
    with dtrace.capture() as trace:
        out = fn()
        mx.eval(out)
    return out, [list(item) for item in trace]


def _stats(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * .95) - 1)],
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
        "n": len(values),
    }


def _bench(fn) -> dict[str, object]:
    all_samples: list[float] = []
    sessions = []
    for _ in range(SESSIONS):
        for _ in range(WARMUP_SAMPLES):
            for _ in range(DISPATCHES_PER_SAMPLE):
                mx.eval(fn())
            mx.synchronize()
        samples: list[float] = []
        for _ in range(SAMPLES_PER_SESSION):
            mx.synchronize()
            started = time.perf_counter()
            for _ in range(DISPATCHES_PER_SAMPLE):
                mx.eval(fn())
            mx.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0 / DISPATCHES_PER_SAMPLE)
        all_samples.extend(samples)
        sessions.append(_stats(samples))
    result = _stats(all_samples)
    result["sessions"] = sessions
    result["dispatches_per_sample"] = DISPATCHES_PER_SAMPLE
    return result


def _run(shape: Shape, order: list[str]) -> dict[str, object]:
    q, k, v = _make_qkv(shape)
    fns = {
        "sdpa": lambda: _call(q, k, v, shape, "sdpa"),
        "mfa": lambda: _call(q, k, v, shape, "mfa"),
    }
    reference = _reference(q, k, v, shape)
    sdpa, sdpa_trace = _trace(fns["sdpa"])
    mfa, mfa_trace = _trace(fns["mfa"])
    mx.eval(reference, sdpa, mfa)
    validation = {"sdpa": _metrics(sdpa, reference), "mfa": _metrics(mfa, reference)}
    if validation["sdpa"]["cos"] < .999 or validation["mfa"]["cos"] < .999:
        raise RuntimeError(f"correction failed for {shape}: {validation}")
    if not sdpa_trace or sdpa_trace[-1][0] != "sdpa":
        raise RuntimeError(f"SDPA engagement failed for {shape}: {sdpa_trace}")
    if not mfa_trace or mfa_trace[-1][0] != "mfa_primitive":
        raise RuntimeError(f"MFA engagement failed for {shape}: {mfa_trace}")
    delta = float(np.max(np.abs(
        np.asarray(mfa.astype(mx.float32)) - np.asarray(sdpa.astype(mx.float32))
    )))
    if delta == 0.0:
        raise RuntimeError(f"binary fingerprint collapsed for {shape}")
    timing = {arm: _bench(fns[arm]) for arm in order}
    ratio = timing["sdpa"]["median_ms"] / timing["mfa"]["median_ms"]
    print(f"{shape.family:10s} q={shape.q_len:2d} k={shape.k_len:5d} "
          f"gqa={shape.gqa:2d} d={shape.head_dim:3d} {shape.dtype} "
          f"c={int(shape.causal)} SDPA/MFA={ratio:.3f}x")
    return {
        "shape": asdict(shape), "validation": validation,
        "which_binary": {"sdpa_trace": sdpa_trace, "mfa_trace": mfa_trace,
                         "mfa_vs_sdpa_max_abs": delta},
        "timing": timing, "sdpa_over_mfa": ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("boundaries", "cross"), required=True)
    parser.add_argument("--order", choices=("sdpa,mfa", "mfa,sdpa"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    shapes = _boundary_shapes() if args.profile == "boundaries" else _cross_shapes()
    order = args.order.split(",")
    payload = {
        "schema": "mlx-mfa.decode-edge-consolidation.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "python": sys.executable, "mlx": getattr(mx, "__version__", importlib.metadata.version("mlx")),
        "platform": platform.platform(), "profile": args.profile, "order": order,
        "method": {"sessions": SESSIONS, "samples_per_session": SAMPLES_PER_SESSION,
                   "warmup_samples": WARMUP_SAMPLES,
                   "dispatches_per_sample": DISPATCHES_PER_SAMPLE,
                   "sampling_asymmetry": "none"},
        "results": [_run(shape, order) for shape in shapes],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
