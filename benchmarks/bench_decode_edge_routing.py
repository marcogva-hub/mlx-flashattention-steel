#!/usr/bin/env python3
"""Confirm the narrow M5 decode edge before changing public routing.

Each timing sample contains multiple individually evaluated dispatches.  Run
the two arm orders in separate foreground processes to expose warm-cache and
thermal order bias:

  /Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python \
    benchmarks/bench_decode_edge_routing.py --profile edge --arm-order sdpa,mfa
"""

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
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx_mfa
from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import flash_attention


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO / "benchmarks" / "results"
DEFAULT_BATCH = 1
DEFAULT_QUERY_HEADS = 16
DEFAULT_STDDEV = 0.25
DEFAULT_SESSIONS = 5
DEFAULT_WARMUP_SAMPLES = 2
DEFAULT_SAMPLES_PER_SESSION = 5
DEFAULT_DISPATCHES_PER_SAMPLE = 20


@dataclass(frozen=True)
class Shape:
    name: str
    q_len: int
    kv_len: int
    query_heads: int
    kv_heads: int
    head_dim: int
    dtype_name: str
    causal: bool

    @property
    def gqa_factor(self) -> int:
        return self.query_heads // self.kv_heads


def _dtype(name: str):
    if name == "fp16":
        return mx.float16
    if name == "bf16":
        return mx.bfloat16
    raise ValueError(f"unsupported dtype name: {name}")


def _scale(shape: Shape) -> float:
    return 1.0 / math.sqrt(shape.head_dim)


def _seed(shape: Shape) -> int:
    return (
        10_000
        + shape.q_len * 13
        + shape.kv_len
        + shape.head_dim * 17
        + shape.query_heads * 19
        + shape.kv_heads * 23
        + int(shape.causal) * 29
        + (31 if shape.dtype_name == "bf16" else 0)
    )


def _make_qkv(shape: Shape):
    mx.random.seed(_seed(shape))
    dtype = _dtype(shape.dtype_name)
    q = (mx.random.normal(
        (DEFAULT_BATCH, shape.query_heads, shape.q_len, shape.head_dim)
    ) * DEFAULT_STDDEV).astype(dtype)
    k = (mx.random.normal(
        (DEFAULT_BATCH, shape.kv_heads, shape.kv_len, shape.head_dim)
    ) * DEFAULT_STDDEV).astype(dtype)
    v = (mx.random.normal(
        (DEFAULT_BATCH, shape.kv_heads, shape.kv_len, shape.head_dim)
    ) * DEFAULT_STDDEV).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _call(q, k, v, shape: Shape, backend: str):
    return flash_attention(
        q,
        k,
        v,
        scale=_scale(shape),
        causal=shape.causal,
        backend=backend,
    )


def _reference(q, k, v, shape: Shape):
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32),
        k.astype(mx.float32),
        v.astype(mx.float32),
        scale=_scale(shape),
        mask=("causal" if shape.causal else None),
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    left = a.astype(np.float64, copy=False).reshape(-1)
    right = b.astype(np.float64, copy=False).reshape(-1)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denom) if denom else 1.0


def _output_metrics(got, reference) -> dict[str, float | bool]:
    got_np = np.asarray(got.astype(mx.float32))
    ref_np = np.asarray(reference.astype(mx.float32))
    return {
        "cos": _cosine(got_np, ref_np),
        "max_abs": float(np.max(np.abs(got_np - ref_np))),
        "finite": bool(np.isfinite(got_np).all()),
    }


def _trace(fn) -> list[list[str]]:
    with dtrace.capture() as trace:
        out = fn()
        mx.eval(out)
    return [list(event) for event in trace]


def _terminal(trace: list[list[str]]) -> str:
    return trace[-1][0] if trace else "<none>"


def _validate(shape: Shape, expected_auto: str) -> dict:
    q, k, v = _make_qkv(shape)
    reference = _reference(q, k, v, shape)
    sdpa = _call(q, k, v, shape, "sdpa")
    mfa = _call(q, k, v, shape, "mfa")
    auto = _call(q, k, v, shape, "auto")
    mx.eval(reference, sdpa, mfa, auto)

    sdpa_trace = _trace(lambda: _call(q, k, v, shape, "sdpa"))
    mfa_trace = _trace(lambda: _call(q, k, v, shape, "mfa"))
    auto_trace = _trace(lambda: _call(q, k, v, shape, "auto"))
    sdpa_metrics = _output_metrics(sdpa, reference)
    mfa_metrics = _output_metrics(mfa, reference)
    auto_metrics = _output_metrics(auto, reference)
    auto_vs_sdpa = float(np.max(np.abs(
        np.asarray(auto.astype(mx.float32)) - np.asarray(sdpa.astype(mx.float32))
    )))
    mfa_vs_sdpa = float(np.max(np.abs(
        np.asarray(mfa.astype(mx.float32)) - np.asarray(sdpa.astype(mx.float32))
    )))

    if mfa_metrics["cos"] < 0.999 or not mfa_metrics["finite"]:
        raise RuntimeError(f"{shape.name}: MFA oracle validation failed: {mfa_metrics}")
    if sdpa_metrics["cos"] < 0.999 or not sdpa_metrics["finite"]:
        raise RuntimeError(f"{shape.name}: SDPA oracle validation failed: {sdpa_metrics}")
    if _terminal(sdpa_trace) != "sdpa":
        raise RuntimeError(f"{shape.name}: expected SDPA trace, got {sdpa_trace}")
    if _terminal(mfa_trace) != "mfa_primitive":
        raise RuntimeError(f"{shape.name}: expected mfa_primitive trace, got {mfa_trace}")
    if _terminal(auto_trace) != expected_auto:
        raise RuntimeError(
            f"{shape.name}: expected auto terminal {expected_auto}, got {auto_trace}"
        )
    return {
        "sdpa_vs_fp32": sdpa_metrics,
        "mfa_vs_fp32": mfa_metrics,
        "auto_vs_fp32": auto_metrics,
        "auto_vs_sdpa_max_abs": auto_vs_sdpa,
        "mfa_vs_sdpa_max_abs": mfa_vs_sdpa,
        "sdpa_trace": sdpa_trace,
        "mfa_trace": mfa_trace,
        "auto_trace": auto_trace,
    }


def _stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "mean_ms": statistics.fmean(samples),
        "samples_ms": samples,
        "n": len(samples),
    }


def _bench_arm(fn, *, sessions: int, warmup_samples: int,
               samples_per_session: int, dispatches_per_sample: int) -> dict:
    samples: list[float] = []
    session_stats: list[dict] = []
    for _ in range(sessions):
        for _ in range(warmup_samples):
            for _ in range(dispatches_per_sample):
                mx.eval(fn())
            mx.synchronize()
        session_samples: list[float] = []
        for _ in range(samples_per_session):
            mx.synchronize()
            started = time.perf_counter()
            for _ in range(dispatches_per_sample):
                mx.eval(fn())
            mx.synchronize()
            session_samples.append(
                (time.perf_counter() - started) * 1000.0 / dispatches_per_sample
            )
        session_stats.append(_stats(session_samples))
        samples.extend(session_samples)
    result = _stats(samples)
    result["sessions"] = session_stats
    result["dispatches_per_sample"] = dispatches_per_sample
    return result


def _shape(name: str, q_len: int, kv_len: int, gqa: int, head_dim: int,
           dtype_name: str, causal: bool) -> Shape:
    if DEFAULT_QUERY_HEADS % gqa:
        raise ValueError(f"GQA {gqa} does not divide {DEFAULT_QUERY_HEADS}")
    return Shape(name, q_len, kv_len, DEFAULT_QUERY_HEADS,
                 DEFAULT_QUERY_HEADS // gqa, head_dim, dtype_name, causal)


def _edge_shapes() -> list[Shape]:
    return [
        _shape(f"edge_q8_k{k_len}_gqa8_d64_{dtype}_nc", 8, k_len, 8, 64, dtype, False)
        for k_len in (4096, 8192, 16384, 32768)
        for dtype in ("fp16", "bf16")
    ]


def _boundary_shapes() -> list[Shape]:
    shapes: dict[str, Shape] = {shape.name: shape for shape in _edge_shapes()}
    for q_len in (4, 8, 16):
        for dtype in ("fp16", "bf16"):
            shape = _shape(f"boundary_q{q_len}_k4096_gqa8_d64_{dtype}_nc", q_len, 4096, 8, 64, dtype, False)
            shapes[shape.name] = shape
    for gqa in (4, 8, 16):
        for dtype in ("fp16", "bf16"):
            shape = _shape(f"boundary_q8_k4096_gqa{gqa}_d64_{dtype}_nc", 8, 4096, gqa, 64, dtype, False)
            shapes[shape.name] = shape
    for head_dim in (64, 128):
        for dtype in ("fp16", "bf16"):
            shape = _shape(f"boundary_q8_k4096_gqa8_d{head_dim}_{dtype}_nc", 8, 4096, 8, head_dim, dtype, False)
            shapes[shape.name] = shape
    for causal in (False, True):
        for dtype in ("fp16", "bf16"):
            suffix = "c" if causal else "nc"
            shape = _shape(f"boundary_q8_k4096_gqa8_d64_{dtype}_{suffix}", 8, 4096, 8, 64, dtype, causal)
            shapes[shape.name] = shape
    for k_len in (2048, 4096, 32768, 65536):
        for dtype in ("fp16", "bf16"):
            shape = _shape(f"boundary_q8_k{k_len}_gqa8_d64_{dtype}_nc", 8, k_len, 8, 64, dtype, False)
            shapes[shape.name] = shape
    return list(shapes.values())


def _run_shape(shape: Shape, args) -> dict:
    validation = _validate(shape, args.expect_auto)
    q, k, v = _make_qkv(shape)
    arms = {
        "sdpa": lambda: _call(q, k, v, shape, "sdpa"),
        "mfa": lambda: _call(q, k, v, shape, "mfa"),
    }
    timing = {}
    for arm in args.arm_order.split(","):
        timing[arm] = _bench_arm(
            arms[arm],
            sessions=args.sessions,
            warmup_samples=args.warmup_samples,
            samples_per_session=args.samples_per_session,
            dispatches_per_sample=args.dispatches_per_sample,
        )
    ratio = timing["sdpa"]["median_ms"] / timing["mfa"]["median_ms"]
    print(
        f"{shape.name:53s} SDPA/MFA={ratio:.3f}x "
        f"MFA={timing['mfa']['median_ms']:.4f}ms "
        f"SDPA={timing['sdpa']['median_ms']:.4f}ms "
        f"cos={validation['mfa_vs_fp32']['cos']:.8f}"
    )
    return {
        "shape": asdict(shape),
        "validation": validation,
        "timing": timing,
        "sdpa_over_mfa": ratio,
    }


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("edge", "boundaries"), default="edge")
    parser.add_argument("--arm-order", choices=("sdpa,mfa", "mfa,sdpa"), required=True)
    parser.add_argument(
        "--expect-auto", choices=("sdpa", "mfa_primitive"), default="mfa_primitive"
    )
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--warmup-samples", type=int, default=DEFAULT_WARMUP_SAMPLES)
    parser.add_argument("--samples-per-session", type=int, default=DEFAULT_SAMPLES_PER_SESSION)
    parser.add_argument("--dispatches-per-sample", type=int, default=DEFAULT_DISPATCHES_PER_SAMPLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.dispatches_per_sample < DEFAULT_DISPATCHES_PER_SAMPLE:
        parser.error(f"--dispatches-per-sample must be >= {DEFAULT_DISPATCHES_PER_SAMPLE}")

    shapes = _edge_shapes() if args.profile == "edge" else _boundary_shapes()
    results = [_run_shape(shape, args) for shape in shapes]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "created_at": stamp,
        "commit": _git_head(),
        "python": sys.executable,
        "mlx_version": getattr(mx, "__version__", None) or importlib.metadata.version("mlx"),
        "mlx_mfa_version": getattr(mlx_mfa, "__version__", "unknown"),
        "platform": platform.platform(),
        "device": str(mx.default_device()),
        "profile": args.profile,
        "arm_order": args.arm_order,
        "expect_auto": args.expect_auto,
        "sessions": args.sessions,
        "warmup_samples": args.warmup_samples,
        "samples_per_session": args.samples_per_session,
        "dispatches_per_sample": args.dispatches_per_sample,
        "results": results,
    }
    path = args.out_dir / f"decode_edge_{args.profile}_{args.arm_order.replace(',', '_')}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
