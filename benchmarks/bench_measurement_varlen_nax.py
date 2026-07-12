#!/usr/bin/env python3
"""Complete the packed-varlen V6NAX GQA/causal performance map.

Run this against the expert candidate worktree.  Every arm is a distinct raw
symbol: V6NAX varlen, STEEL varlen, or per-segment MLX SDPA.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _ext


REPO = Path(__file__).resolve().parents[1]
GEOMETRIES = {
    "seed_aligned": [3226, 3226, 3226, 3226, 2434, 1642, 1642, 1642, 1642,
                     1246, 1642, 1642, 1642, 1642, 1246, 850, 850, 850, 850, 652],
    "seed_shifted": [454, 850, 850, 850, 850, 256, 850, 1642, 1642, 1642,
                     1642, 454, 850, 1642, 1642, 1642, 1642, 454, 1642, 3226,
                     3226, 3226, 3226, 850],
    "equal_1024": [1024] * 8,
}
GQA_FACTORS = (2, 4, 8)
DTYPES = ("fp16", "bf16")
CAUSALS = (False, True)
TILES = {
    "default": {},
    "candidate_bq32_bk32_wm2": {
        "MFA_V6_NAX_BQ": "32", "MFA_V6_NAX_BK": "32", "MFA_V6_NAX_WM": "2",
    },
}
SESSIONS = 5
SAMPLES_PER_SESSION = 3
WARMUP_SAMPLES = 2


def _prefix(lengths: list[int]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def _tiles(lengths: list[int], block_q: int) -> mx.array:
    return mx.array(_prefix([math.ceil(value / block_q) for value in lengths]), dtype=mx.int32)


@contextmanager
def _env(values: dict[str, str]):
    names = ("MFA_V6_NAX_BQ", "MFA_V6_NAX_BK", "MFA_V6_NAX_WM")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            if name in values:
                os.environ[name] = values[name]
            else:
                os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _eval(value) -> None:
    mx.eval(value)
    mx.synchronize()


def _cosine(a, b) -> float:
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    value = mx.sum(a * b) / mx.sqrt(mx.sum(a * a) * mx.sum(b * b))
    mx.eval(value)
    return float(value.item())


def _stats(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)],
        "min_ms": min(values), "max_ms": max(values),
        "samples_ms": values, "n": len(values),
    }


def _bench(fn) -> dict[str, object]:
    all_samples: list[float] = []
    sessions = []
    for _ in range(SESSIONS):
        for _ in range(WARMUP_SAMPLES):
            _eval(fn())
        samples = []
        for _ in range(SAMPLES_PER_SESSION):
            mx.synchronize()
            start = time.perf_counter()
            _eval(fn())
            samples.append((time.perf_counter() - start) * 1000.0)
        all_samples.extend(samples)
        sessions.append(_stats(samples))
    result = _stats(all_samples)
    result["sessions"] = sessions
    return result


def _split_sdpa(q, k, v, prefix, scale, causal):
    return mx.concatenate([
        mx.fast.scaled_dot_product_attention(
            q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop],
            scale=scale, mask="causal" if causal else None,
        )
        for start, stop in zip(prefix[:-1], prefix[1:])
    ], axis=2)


def _run_cell(geometry: str, lengths: list[int], gqa: int, dtype_name: str,
              causal: bool, tile_name: str, settings: dict[str, str], order: list[str]):
    D, H_Q = 128, 16
    H_KV = H_Q // gqa
    total = sum(lengths)
    dtype = mx.float16 if dtype_name == "fp16" else mx.bfloat16
    seed = (721000 + total + gqa * 23 + int(causal) * 17
            + (1 if dtype_name == "bf16" else 0) + (0 if tile_name == "default" else 101))
    mx.random.seed(seed)
    q = (mx.random.normal((1, H_Q, total, D)) * .05).astype(dtype)
    k = (mx.random.normal((1, H_KV, total, D)) * .05).astype(dtype)
    v = (mx.random.normal((1, H_KV, total, D)) * .05).astype(dtype)
    cu = mx.array(_prefix(lengths), dtype=mx.int32)
    prefix = _prefix(lengths)
    scale = 1.0 / math.sqrt(D)
    # Candidate BQ changes its own tile offsets; STEEL remains fixed BQ=32.
    nax_bq = int(settings.get("MFA_V6_NAX_BQ", "64"))
    nax_tiles = _tiles(lengths, nax_bq)
    steel_tiles = _tiles(lengths, 32)
    _eval((q, k, v, cu, nax_tiles, steel_tiles))

    with _env(settings):
        arms = {
            "nax": lambda: _ext.v6_nax_varlen_forward(q, k, v, cu, cu, nax_tiles, scale, causal)[0],
            "steel": lambda: _ext.mfa_attention_varlen_forward(q, k, v, cu, cu, steel_tiles, scale, causal)[0],
            "sdpa": lambda: _split_sdpa(q, k, v, prefix, scale, causal),
        }
        outputs = {name: call() for name, call in arms.items()}
        _eval(tuple(outputs.values()))
        segment_cos = [_cosine(outputs["nax"][:, :, start:stop], outputs["sdpa"][:, :, start:stop])
                       for start, stop in zip(prefix[:-1], prefix[1:])]
        correction = {
            "global_cos_nax_sdpa": _cosine(outputs["nax"], outputs["sdpa"]),
            "min_segment_cos_nax_sdpa": min(segment_cos),
            "finite": bool(mx.all(mx.isfinite(outputs["nax"])).item()),
        }
        deltas = {
            "nax_vs_steel_max_abs": float(mx.max(mx.abs(outputs["nax"].astype(mx.float32) - outputs["steel"].astype(mx.float32))).item()),
            "nax_vs_sdpa_max_abs": float(mx.max(mx.abs(outputs["nax"].astype(mx.float32) - outputs["sdpa"].astype(mx.float32))).item()),
        }
        if correction["global_cos_nax_sdpa"] < .999 or correction["min_segment_cos_nax_sdpa"] < .999:
            raise RuntimeError(f"correction failed: {geometry}/{gqa}/{dtype_name}/{causal}/{tile_name}: {correction}")
        if not correction["finite"] or not all(value > 0.0 for value in deltas.values()):
            raise RuntimeError(f"which-binary delta failed: {deltas}")
        timing = {name: _bench(arms[name]) for name in order}

    ratios = {
        "steel_over_nax": timing["steel"]["median_ms"] / timing["nax"]["median_ms"],
        "sdpa_over_nax": timing["sdpa"]["median_ms"] / timing["nax"]["median_ms"],
    }
    print(f"{geometry:13s} gqa={gqa} {dtype_name} causal={int(causal)} {tile_name:24s} "
          f"SDPA/NAX={ratios['sdpa_over_nax']:.3f}x STEEL/NAX={ratios['steel_over_nax']:.3f}x")
    return {
        "geometry": geometry, "lengths": lengths, "total_tokens": total,
        "num_segments": len(lengths), "Hq": H_Q, "Hkv": H_KV, "gqa": gqa,
        "D": D, "dtype": dtype_name, "causal": causal,
        "tile": {"name": tile_name, "bq": nax_bq, "bk": int(settings.get("MFA_V6_NAX_BK", "32")),
                 "wm": int(settings.get("MFA_V6_NAX_WM", "4"))},
        "correction": correction,
        "which_binary": {
            "nax": "_ext.v6_nax_varlen_forward", "steel": "_ext.mfa_attention_varlen_forward",
            "sdpa": "per-segment mx.fast.scaled_dot_product_attention", **deltas,
        },
        "timing": timing, "ratios": ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("nax,steel,sdpa", "sdpa,steel,nax"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    order = args.order.split(",")
    rows = [
        _run_cell(geometry, lengths, gqa, dtype, causal, tile_name, settings, order)
        for geometry, lengths in GEOMETRIES.items()
        for gqa in GQA_FACTORS
        for dtype in DTYPES
        for causal in CAUSALS
        for tile_name, settings in TILES.items()
    ]
    payload = {
        "schema": "mlx-mfa.varlen-packed-consolidation.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.executable, "mlx": getattr(mx, "__version__", importlib.metadata.version("mlx")),
        "platform": platform.platform(), "order": order,
        "method": {"sessions": SESSIONS, "samples_per_session": SAMPLES_PER_SESSION,
                   "warmup_samples": WARMUP_SAMPLES, "dispatches_per_sample": 1,
                   "sampling_asymmetry": "none"},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
