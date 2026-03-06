#!/usr/bin/env python3
"""bench_compile.py — mx.compile overhead benchmark (v0.9.2 Track DC).

Measures wall-clock time for:
  1. _softcap_sdpa_ref   (compiled, Track CB)
  2. _alibi_sdpa_ref     (compiled, Track CB)
  3. _apply_rope_mlx     (compiled, Track DC)

Each function is timed compiled vs raw-Python equivalent (no mx.compile),
reporting median latency and speedup over 50 timed iterations.

Usage:
    python benchmarks/bench_compile.py
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import mlx.core as mx

from mlx_mfa.attention import (
    _softcap_sdpa_ref,
    _alibi_sdpa_ref,
    _apply_rope_mlx,
)

_flush = getattr(mx, "eval")

WARMUP = 5
TIMED = 50


# ── Timing harness ──────────────────────────────────────────────────────────

def timed_ms(fn, warmup=WARMUP, n=TIMED) -> float:
    for _ in range(warmup):
        fn()
        _flush()
    mx.synchronize()
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        _flush()
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ── Raw (un-compiled) reference implementations ─────────────────────────────

def _softcap_raw(q, k, v, scale, causal, softcap):
    S = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * scale
    S = mx.tanh(S / softcap) * softcap
    if causal:
        N, Sk = q.shape[2], k.shape[2]
        mask = mx.triu(mx.full((N, Sk), float("-inf"), dtype=q.dtype), k=Sk - N + 1)
        S = S + mask
    A = mx.softmax(S.astype(mx.float32), axis=-1).astype(q.dtype)
    return mx.matmul(A, v)


def _alibi_raw(q, k, v, slopes, scale, causal):
    B, H, N, D = q.shape
    S = k.shape[2]
    scores = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * scale
    q_pos = mx.arange(N, dtype=mx.float32)[None, None, :, None]
    k_pos = mx.arange(S, dtype=mx.float32)[None, None, None, :]
    bias = slopes[:, :, None, None] * (k_pos - q_pos)
    scores = scores + bias.astype(scores.dtype)
    if causal:
        mask = mx.triu(mx.full((N, S), float("-inf"), dtype=q.dtype), k=S - N + 1)
        scores = scores + mask
    A = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return mx.matmul(A, v)


def _rope_raw(x, cos, sin, offset=0, interleaved=True):
    B, H, N, D = x.shape
    half_D = D // 2
    cos_n = cos[offset : offset + N, :]
    sin_n = sin[offset : offset + N, :]
    cos_bc = cos_n[None, None, :, :].astype(x.dtype)
    sin_bc = sin_n[None, None, :, :].astype(x.dtype)
    if interleaved:
        x_pairs = x.reshape(B, H, N, half_D, 2)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]
        x0_rot = x0 * cos_bc - x1 * sin_bc
        x1_rot = x0 * sin_bc + x1 * cos_bc
        return mx.stack([x0_rot, x1_rot], axis=-1).reshape(B, H, N, D)
    else:
        x0 = x[..., :half_D]
        x1 = x[..., half_D:]
        return mx.concatenate([x0 * cos_bc - x1 * sin_bc, x0 * sin_bc + x1 * cos_bc], axis=-1)


# ── Benchmark configurations ─────────────────────────────────────────────────

def bench_softcap():
    B, H, N, D = 1, 8, 2048, 128
    dtype = mx.float16
    scale, softcap = 1.0 / math.sqrt(D), 50.0
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    _flush(q, k, v)

    # Warm up compile cache
    _softcap_sdpa_ref(q, k, v, scale, True, softcap)

    compiled_ms = timed_ms(lambda: _softcap_sdpa_ref(q, k, v, scale, True, softcap))
    raw_ms = timed_ms(lambda: _softcap_raw(q, k, v, scale, True, softcap))
    return "softcap_sdpa N=2048 D=128 f16", compiled_ms, raw_ms


def bench_alibi():
    B, H, N, D = 1, 8, 2048, 128
    dtype = mx.float16
    scale = 1.0 / math.sqrt(D)
    slopes = mx.array([2 ** (-8 * (h + 1) / H) for h in range(H)], dtype=mx.float32)
    slopes = slopes.reshape(1, H, 1, 1)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    _flush(q, k, v, slopes)

    _alibi_sdpa_ref(q, k, v, slopes, scale, False)

    compiled_ms = timed_ms(lambda: _alibi_sdpa_ref(q, k, v, slopes, scale, False))
    raw_ms = timed_ms(lambda: _alibi_raw(q, k, v, slopes, scale, False))
    return "alibi_sdpa N=2048 D=128 f16", compiled_ms, raw_ms


def bench_rope():
    B, H, N, D = 1, 8, 2048, 128
    dtype = mx.float16
    x = mx.random.normal((B, H, N, D)).astype(dtype)
    cos = mx.random.normal((N, D // 2))
    sin = mx.random.normal((N, D // 2))
    _flush(x, cos, sin)

    _apply_rope_mlx(x, cos, sin, offset=0, interleaved=True)

    compiled_ms = timed_ms(lambda: _apply_rope_mlx(x, cos, sin, offset=0, interleaved=True))
    raw_ms = timed_ms(lambda: _rope_raw(x, cos, sin, offset=0, interleaved=True))
    return "rope_mlx N=2048 D=128 f16", compiled_ms, raw_ms


def bench_rope_ni():
    """Non-interleaved (GPT-NeoX) RoPE."""
    B, H, N, D = 1, 8, 2048, 128
    dtype = mx.float16
    x = mx.random.normal((B, H, N, D)).astype(dtype)
    cos = mx.random.normal((N, D // 2))
    sin = mx.random.normal((N, D // 2))
    _flush(x, cos, sin)

    _apply_rope_mlx(x, cos, sin, offset=0, interleaved=False)

    compiled_ms = timed_ms(lambda: _apply_rope_mlx(x, cos, sin, offset=0, interleaved=False))
    raw_ms = timed_ms(lambda: _rope_raw(x, cos, sin, offset=0, interleaved=False))
    return "rope_mlx_ni N=2048 D=128 f16", compiled_ms, raw_ms


# ── Print ────────────────────────────────────────────────────────────────────

HDR = f"{'Function':<35} {'Compiled (ms)':>14} {'Raw (ms)':>10} {'Speedup':>9}"
SEP = "-" * 74


def _row(label, compiled, raw):
    spd = raw / compiled if compiled > 0 else float("nan")
    tag = " ★" if spd >= 1.3 else "  "
    return f"{label:<35} {compiled:>13.3f}ms {raw:>9.3f}ms {spd:>8.2f}x{tag}"


def main():
    print(f"\nmlx-mfa v0.9.2 — bench_compile  (warmup={WARMUP}, timed={TIMED})")
    print(HDR)
    print(SEP)

    for fn in [bench_softcap, bench_alibi, bench_rope, bench_rope_ni]:
        label, compiled, raw = fn()
        print(_row(label, compiled, raw))

    print()


if __name__ == "__main__":
    main()
