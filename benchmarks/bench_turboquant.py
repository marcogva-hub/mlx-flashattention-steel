#!/usr/bin/env python3
"""Benchmark TurboQuant compression ratio, quality, and overhead.

Reports: compression time, decompression time, memory ratio,
inner product correlation, and attention output max diff.
"""
import math
import sys
import time

import mlx.core as mx

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))  # repo review 2026-05: allow `python benchmarks/<f>.py` from repo root
from bench_utils import med


def _pearson_corr(a, b):
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    a_m = a_f - a_f.mean()
    b_m = b_f - b_f.mean()
    num = (a_m * b_m).mean()
    den = mx.sqrt((a_m * a_m).mean()) * mx.sqrt((b_m * b_m).mean())
    return (num / (den + 1e-12)).item()


def bench_compress_decompress():
    from mlx_mfa.turboquant import turboquant_compress, turboquant_decompress

    B, H, D = 1, 8, 128
    configs = [
        # (S, bits, use_qjl, rotation)
        (1024,  2, False, "wht"),
        (1024,  2, True,  "wht"),
        (1024,  3, False, "wht"),
        (1024,  3, True,  "wht"),
        (1024,  3, False, "qr"),
        (1024,  3, True,  "qr"),
        (1024,  4, False, "wht"),
        (1024,  4, True,  "wht"),
        (4096,  3, True,  "wht"),
        (16384, 3, True,  "wht"),
    ]

    print(f"{'Config':48s}  {'Comp ms':>8}  {'Dec ms':>8}  {'Ratio':>6}  {'Corr':>6}  {'Attn Diff':>9}")
    print("-" * 100)

    mx.random.seed(42)
    for S, bits, use_qjl, rotation in configs:
        Q = mx.random.normal((B, H, 16, D)).astype(mx.float16)
        K = mx.random.normal((B, H, S, D)).astype(mx.float16)
        V = mx.random.normal((B, H, S, D)).astype(mx.float16)
        mx.synchronize()

        # Compress timing
        comp_ms = med(
            lambda: turboquant_compress(K, bits=bits, use_qjl=use_qjl, rotation=rotation),
            warmup=2, iters=5,
        )

        # Decompress timing
        c = turboquant_compress(K, bits=bits, use_qjl=use_qjl, rotation=rotation)
        dec_ms = med(lambda: turboquant_decompress(c), warmup=2, iters=5)

        # Memory ratio
        nbytes_c = c["x_q_packed"].nbytes + c["scales"].nbytes
        if "qjl_signs_packed" in c:
            nbytes_c += c["qjl_signs_packed"].nbytes + c["qjl_norms"].nbytes
        nbytes_fp16 = K.nbytes
        ratio = nbytes_fp16 / nbytes_c

        # Score correlation
        K_dec = turboquant_decompress(c)
        scores_true = Q.astype(mx.float32) @ K.astype(mx.float32).swapaxes(-1, -2)
        scores_approx = Q.astype(mx.float32) @ K_dec.astype(mx.float32).swapaxes(-1, -2)
        corr = _pearson_corr(scores_true, scores_approx)

        # Attention output diff
        scale = 1.0 / math.sqrt(D)
        ref = mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale)
        approx = mx.fast.scaled_dot_product_attention(Q, K_dec, V, scale=scale)
        attn_diff = (ref.astype(mx.float32) - approx.astype(mx.float32)).abs().max().item()

        qjl_str = "+QJL" if use_qjl else "    "
        lbl = f"S={S:5d} {bits}bit {qjl_str} {rotation:3s}  B={B} H={H} D={D}"
        print(f"{lbl:48s}  {comp_ms:8.2f}  {dec_ms:8.2f}  {ratio:5.2f}x  {corr:6.4f}  {attn_diff:9.4f}")

    print()


def bench_kvcache_memory():
    from mlx_mfa.turboquant import TurboQuantKVCache

    print("KV Cache Memory Report")
    print("-" * 70)
    print(f"{'Config':36s}  {'Compressed':>12}  {'FP16':>12}  {'Ratio':>7}")
    print("-" * 70)

    B, H, D, S = 1, 32, 128, 4096  # Llama-7B-like
    for bits in [2, 3, 4]:
        for compress_v in [False, True]:
            cache = TurboQuantKVCache(
                bits=bits, use_qjl=True, compress_v=compress_v
            )
            k = mx.random.normal((B, H, S, D)).astype(mx.float16)
            v = mx.random.normal((B, H, S, D)).astype(mx.float16)
            cache.append(k, v)

            cv_str = "K+V" if compress_v else "K  "
            lbl = f"{bits}bit {cv_str}  B={B} H={H} S={S} D={D}"
            print(
                f"{lbl:36s}  {cache.memory_bytes:>10,} B  "
                f"{cache.memory_bytes_fp16:>10,} B  "
                f"{cache.compression_ratio:6.2f}x"
            )
    print()


def main():
    print("=" * 100)
    print("TurboQuant Benchmark — Phase 1 (non-fused)")
    print("=" * 100)
    print()
    bench_compress_decompress()
    bench_kvcache_memory()


if __name__ == "__main__":
    main()
