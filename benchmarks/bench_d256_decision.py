"""D=256 decision benchmark: SDPA vs MFA routes.

This benchmark is intentionally narrow and production-oriented:
  - D=256 only
  - N in {4096, 8192, 16384}
  - causal on/off
  - f16, B=2, H=8

Compared paths:
  - SDPA baseline (`_fallback_sdpa`)
  - MFA V1 (`MFA_DISABLE_V2=1`)
  - MFA V2 D-split (`MFA_DISABLE_V2` unset)
  - `flash_attention(..., backend="auto")` effective route

Split-K is not applicable to D=256 (V2 split-K dispatch only supports D=64/128).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager

import mlx.core as mx

from mlx_mfa import flash_attention, get_device_info, __version__
from mlx_mfa.attention import _fallback_sdpa, _mfa_forward


@contextmanager
def _env(name: str, value: str | None):
    prev = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def _measure(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return float(times[len(times) // 2])


def run_case(B: int, H: int, N: int, causal: bool, warmup: int, iters: int) -> dict:
    D = 256
    scale = 1.0 / math.sqrt(D)
    q = mx.random.normal([B, H, N, D]).astype(mx.float16)
    k = mx.random.normal([B, H, N, D]).astype(mx.float16)
    v = mx.random.normal([B, H, N, D]).astype(mx.float16)
    mx.eval(q, k, v)

    sdpa_ms = _measure(lambda: _fallback_sdpa(q, k, v, scale, causal), warmup, iters)

    with _env("MFA_DISABLE_V2", "1"):
        v1_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), warmup, iters)

    with _env("MFA_DISABLE_V2", None):
        v2_ms = _measure(lambda: _mfa_forward(q, k, v, scale, causal), warmup, iters)

    auto_ms = _measure(
        lambda: flash_attention(q, k, v, scale=scale, causal=causal, backend="auto"),
        warmup,
        iters,
    )

    return {
        "B": B,
        "H": H,
        "N": N,
        "D": D,
        "causal": causal,
        "sdpa_ms": sdpa_ms,
        "v1_ms": v1_ms,
        "v2_dsplit_ms": v2_ms,
        "auto_ms": auto_ms,
        "v1_vs_sdpa": sdpa_ms / v1_ms if v1_ms > 0 else 0.0,
        "v2_vs_sdpa": sdpa_ms / v2_ms if v2_ms > 0 else 0.0,
        "auto_vs_sdpa": sdpa_ms / auto_ms if auto_ms > 0 else 0.0,
        "splitk_applicable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D=256 production decision benchmark")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--output",
        type=str,
        default="devnotes/d256_decision_latest.json",
        help="Where to write JSON results",
    )
    args = parser.parse_args()

    device = get_device_info()
    rows = []
    for causal in (True, False):
        for N in (4096, 8192, 16384):
            rows.append(run_case(args.batch, args.heads, N, causal, args.warmup, args.iters))

    print(
        f"Device: {device['device_name']} (gen {device['gpu_family_gen']}, "
        f"M3+={device['is_m3_plus']})"
    )
    print(f"mlx-mfa: {__version__}")
    print(
        f"{'Config':<30} {'SDPA':>8} {'V1':>8} {'V2-ds':>8} {'AUTO':>8} "
        f"{'V2/SDPA':>9} {'AUTO/SDPA':>10}"
    )
    print("-" * 90)
    for r in rows:
        cfg = f"N={r['N']} causal={r['causal']}"
        print(
            f"{cfg:<30} {r['sdpa_ms']:>8.2f} {r['v1_ms']:>8.2f} {r['v2_dsplit_ms']:>8.2f} "
            f"{r['auto_ms']:>8.2f} {r['v2_vs_sdpa']:>8.2f}x {r['auto_vs_sdpa']:>9.2f}x"
        )

    out = {
        "device": device,
        "mlx_mfa_version": __version__,
        "batch": args.batch,
        "heads": args.heads,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": rows,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {args.output}")
    print("Note: split-K is not applicable to D=256 (D=64/128 only).")


if __name__ == "__main__":
    main()
