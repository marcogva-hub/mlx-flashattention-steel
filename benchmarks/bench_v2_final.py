"""bench_v2_final.py — Comprehensive STEEL V2 benchmark for RESULTS.md generation.

Covers:
  - Dense causal / non-causal: D=64/128/256, N=2048/4096/8192/16384, f16/bf16
  - Window masking: D=64/128, N=4096/8192, window_size=512/256
  - V2 split-K: small-grid scenarios (B=1 H=1/2/4)
  - V2 vs V1 vs SDPA comparison

Usage:
    python benchmarks/bench_v2_final.py
    python benchmarks/bench_v2_final.py --section dense
    python benchmarks/bench_v2_final.py --section window
    python benchmarks/bench_v2_final.py --section splitk
"""

import argparse
import math
import os
import time

import mlx.core as mx

from mlx_mfa.attention import _fallback_sdpa, _mfa_forward
from mlx_mfa import flash_attention


def _measure(fn, n_warmup=8, n_iter=20):
    for _ in range(n_warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n_iter):
        mx.eval(fn())
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


def _speedup(ref_ms, ms):
    return ref_ms / ms if ms > 0 else float("inf")


# ------------------------------------------------------------------
# Dense causal / non-causal
# ------------------------------------------------------------------

DENSE_CONFIGS = [
    # (D, N, causal, dtype_str)
    (64,   2048, True,  "f16"),
    (64,   4096, True,  "f16"),
    (64,   8192, True,  "f16"),
    (64,   8192, False, "f16"),
    (128,  2048, True,  "f16"),
    (128,  4096, True,  "f16"),
    (128,  8192, True,  "f16"),
    (128,  16384, True, "f16"),
    (128,  4096, True,  "bf16"),
    (128,  8192, False, "f16"),
    (256,  4096, True,  "f16"),
    (256,  8192, True,  "f16"),
    (256,  4096, False, "f16"),
]

DTYPE_MAP = {"f16": mx.float16, "bf16": mx.bfloat16}


def bench_dense(B=2, H=8, n_warmup=8, n_iter=20):
    print(f"\n## Dense forward pass (B={B} H={H}, warmup={n_warmup}, iters={n_iter})\n")
    hdr = f"{'Config':<42} {'SDPA ms':>9} {'V1 ms':>8} {'V2 ms':>8}  {'V1/SDPA':>8} {'V2/SDPA':>8} {'V2/V1':>6}"
    print(hdr)
    print("-" * len(hdr))

    for D, N, causal, dtype_str in DENSE_CONFIGS:
        dtype = DTYPE_MAP[dtype_str]
        mx.random.seed(42)
        q = mx.random.normal([B, H, N, D]).astype(dtype)
        k = mx.random.normal([B, H, N, D]).astype(dtype)
        v = mx.random.normal([B, H, N, D]).astype(dtype)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        sdpa_ms = _measure(lambda: _fallback_sdpa(q, k, v, scale, causal), n_warmup, n_iter)
        os.environ["MFA_DISABLE_V2"] = "1"
        v1_ms   = _measure(lambda: _mfa_forward(q, k, v, scale, causal), n_warmup, n_iter)
        del os.environ["MFA_DISABLE_V2"]
        v2_ms   = _measure(lambda: _mfa_forward(q, k, v, scale, causal), n_warmup, n_iter)

        r1 = _speedup(sdpa_ms, v1_ms)
        r2 = _speedup(sdpa_ms, v2_ms)
        rv = _speedup(v1_ms, v2_ms)
        cstr = "causal" if causal else "non-causal"
        lbl  = f"D={D} N={N} {dtype_str} {cstr}"
        s1 = " *" if r1 >= 1.5 else "  "
        s2 = " *" if r2 >= 1.5 else "  "
        print(f"{lbl:<42} {sdpa_ms:>9.2f} {v1_ms:>8.2f} {v2_ms:>8.2f}  "
              f"{r1:>7.2f}x{s1} {r2:>7.2f}x{s2} {rv:>5.2f}x")


# ------------------------------------------------------------------
# Window masking
# ------------------------------------------------------------------

WINDOW_CONFIGS = [
    # (D, N, window_size)
    (64,  4096, 512),
    (64,  8192, 512),
    (128, 4096, 512),
    (128, 8192, 512),
    (128, 4096, 256),
    (128, 8192, 256),
]


def bench_window(B=2, H=8, n_warmup=8, n_iter=20):
    print(f"\n## Window masking f16 causal (B={B} H={H}, warmup={n_warmup}, iters={n_iter})\n")
    hdr = f"{'Config':<42} {'SDPA ms':>9} {'MFA ms':>8}  {'MFA/SDPA':>9}"
    print(hdr)
    print("-" * len(hdr))

    for D, N, win in WINDOW_CONFIGS:
        mx.random.seed(42)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        sdpa_ms = _measure(lambda: _fallback_sdpa(q, k, v, scale, True), n_warmup, n_iter)
        mfa_ms  = _measure(
            lambda: flash_attention(q, k, v, scale=scale, causal=True, window_size=(win, win)),
            n_warmup, n_iter)

        r   = _speedup(sdpa_ms, mfa_ms)
        lbl = f"D={D} N={N} win={win} f16 causal"
        s   = " *" if r >= 1.5 else "  "
        print(f"{lbl:<42} {sdpa_ms:>9.2f} {mfa_ms:>8.2f}  {r:>8.2f}x{s}")


# ------------------------------------------------------------------
# V2 split-K (small-grid)
# ------------------------------------------------------------------

SPLITK_CONFIGS = [
    (1, 1,  512, 64),
    (1, 1, 1024, 64),
    (1, 1,  512, 128),
    (1, 1, 1024, 128),
    (1, 2,  512, 128),
    (1, 4,  512, 128),
]


def bench_splitk(n_warmup=8, n_iter=20):
    print(f"\n## V2 split-K (small-grid) f16 causal (warmup={n_warmup}, iters={n_iter})\n")
    hdr = f"{'Config':<42} {'SDPA ms':>9} {'V2 ms':>8}  {'V2/SDPA':>8}"
    print(hdr)
    print("-" * len(hdr))

    for B, H, N, D in SPLITK_CONFIGS:
        mx.random.seed(42)
        q = mx.random.normal([B, H, N, D]).astype(mx.float16)
        k = mx.random.normal([B, H, N, D]).astype(mx.float16)
        v = mx.random.normal([B, H, N, D]).astype(mx.float16)
        mx.eval(q, k, v)
        scale = 1.0 / math.sqrt(D)

        sdpa_ms = _measure(lambda: _fallback_sdpa(q, k, v, scale, True), n_warmup, n_iter)
        v2_ms   = _measure(lambda: _mfa_forward(q, k, v, scale, True),  n_warmup, n_iter)

        r   = _speedup(sdpa_ms, v2_ms)
        lbl = f"B={B} H={H} N={N} D={D} f16 causal"
        s   = " *" if r >= 1.5 else "  "
        print(f"{lbl:<42} {sdpa_ms:>9.2f} {v2_ms:>8.2f}  {r:>8.2f}x{s}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="STEEL V2 comprehensive benchmark")
    parser.add_argument("--section", choices=["dense", "window", "splitk", "all"],
                        default="all")
    parser.add_argument("--batch",  type=int, default=2)
    parser.add_argument("--heads",  type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters",  type=int, default=20)
    args = parser.parse_args()

    import mlx_mfa
    info = mlx_mfa.get_device_info()
    print(f"Device: {info['device_name']}  gen={info['gpu_family_gen']}  M3+={info['is_m3_plus']}")
    print(f"mlx-mfa version: {mlx_mfa.__version__}")

    if args.section in ("dense", "all"):
        bench_dense(args.batch, args.heads, args.warmup, args.iters)
    if args.section in ("window", "all"):
        bench_window(args.batch, args.heads, args.warmup, args.iters)
    if args.section in ("splitk", "all"):
        bench_splitk(args.warmup, args.iters)


if __name__ == "__main__":
    main()
