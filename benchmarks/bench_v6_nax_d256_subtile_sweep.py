#!/usr/bin/env python3
"""Autoresearch sweep for the D=256 V6NAX head-subtile prototype."""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_mfa import _ext, flash_attention


REPO = Path(__file__).resolve().parents[1]
SESSIONS, SAMPLES, WARMUP = 5, 5, 2
VARIANTS = {
    "d128_bq64_bk32_wm4": {"MFA_V6_NAX_D_SUBTILE": "128", "MFA_V6_NAX_BQ": "64", "MFA_V6_NAX_BK": "32", "MFA_V6_NAX_WM": "4"},
    "d128_bq32_bk32_wm2": {"MFA_V6_NAX_D_SUBTILE": "128", "MFA_V6_NAX_BQ": "32", "MFA_V6_NAX_BK": "32", "MFA_V6_NAX_WM": "2"},
    "d128_bq64_bk32_wm2": {"MFA_V6_NAX_D_SUBTILE": "128", "MFA_V6_NAX_BQ": "64", "MFA_V6_NAX_BK": "32", "MFA_V6_NAX_WM": "2"},
    "d128_bq64_bk64_wm4": {"MFA_V6_NAX_D_SUBTILE": "128", "MFA_V6_NAX_BQ": "64", "MFA_V6_NAX_BK": "64", "MFA_V6_NAX_WM": "4"},
    "d128_bq32_bk64_wm2": {"MFA_V6_NAX_D_SUBTILE": "128", "MFA_V6_NAX_BQ": "32", "MFA_V6_NAX_BK": "64", "MFA_V6_NAX_WM": "2"},
    "d64_bq32_bk32_wm2": {"MFA_V6_NAX_D_SUBTILE": "64", "MFA_V6_NAX_BQ": "32", "MFA_V6_NAX_BK": "32", "MFA_V6_NAX_WM": "2"},
}
CELLS = ((4096, False), (4096, True), (8192, False))


@contextmanager
def _env(settings: dict[str, str]):
    prior = {name: os.environ.get(name) for name in settings}
    try:
        os.environ.update(settings)
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _cosine(a, b) -> float:
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    x = mx.sum(a * b) / mx.sqrt(mx.sum(a * a) * mx.sum(b * b))
    mx.eval(x)
    return float(x.item())


def _time(call):
    values = []
    for _ in range(SESSIONS):
        for _ in range(WARMUP):
            mx.eval(call())
        for _ in range(SAMPLES):
            mx.synchronize(); start = time.perf_counter()
            mx.eval(call()); mx.synchronize()
            values.append((time.perf_counter() - start) * 1000.0)
    return {"median_ms": statistics.median(values), "samples_ms": values}


def _run_variant(name, settings, N, causal):
    D = 256
    mx.random.seed(260000 + N + int(causal))
    q = (mx.random.normal((1, 1, N, D)) * 0.05).astype(mx.float16)
    k = (mx.random.normal((1, 1, N, D)) * 0.05).astype(mx.float16)
    v = (mx.random.normal((1, 1, N, D)) * 0.05).astype(mx.float16)
    scale = 1.0 / math.sqrt(D)
    sdpa = lambda: flash_attention(q, k, v, causal=causal, scale=scale, backend="sdpa")
    with _env(settings):
        nax = lambda: _ext.v6_nax_forward(q, k, v, causal, True)[0]
        out = nax()
        ref = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=scale, mask="causal" if causal else None,
        )
        mx.eval(out, ref)
        cos = _cosine(out, ref)
        if cos < 0.999:
            raise RuntimeError(f"{name} N={N} causal={causal}: cosine={cos}")
        timings = {}
        for order in ("sdpa,nax", "nax,sdpa"):
            arms = {"sdpa": sdpa, "nax": nax}
            timings[order] = {arm: _time(arms[arm]) for arm in order.split(",")}
    ratios = {order: value["sdpa"]["median_ms"] / value["nax"]["median_ms"]
              for order, value in timings.items()}
    return {"variant": name, "settings": settings, "N": N, "causal": causal,
            "cos_fp32": cos, "timing": timings, "sdpa_over_nax": ratios}


def main():
    rows = []
    for name, settings in VARIANTS.items():
        for N, causal in CELLS:
            try:
                row = _run_variant(name, settings, N, causal)
                rows.append(row)
                print(name, N, causal, row["sdpa_over_nax"])
            except Exception as exc:  # exploratory invalid tiles are recorded loudly.
                row = {"variant": name, "settings": settings, "N": N,
                       "causal": causal, "error": f"{type(exc).__name__}: {exc}"}
                rows.append(row)
                print(name, N, causal, "ERROR", row["error"])
    path = REPO / "benchmarks/results/headdim_stage1_sweep.json"
    path.write_text(json.dumps({"sessions": SESSIONS, "samples": SAMPLES,
                                "warmup": WARMUP, "rows": rows}, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
