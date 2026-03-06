"""bench_kvcache.py -- KV-cache append performance comparison.

Compares two strategies for autoregressive decode with RoPE:

  A (naive):   Store K unrotated in cache. Each decode step re-applies RoPE
               to the full cache (O(past_len) rotation cost per step).

  B (FC):      Store K pre-rotated in cache using flash_attention_kvcache_rope_append.
               Each decode step only rotates k_new (O(1) rotation cost per step).

Usage::

    python benchmarks/bench_kvcache.py               # default params
    python benchmarks/bench_kvcache.py --help        # all options
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_mfa import flash_attention_rope
from mlx_mfa import flash_attention_kvcache_rope_append
from mlx_mfa.attention import _apply_rope_mlx


def bench_naive_rope_decode(q, k_new, v_new, k_cache_unrot, v_cache, cos, sin,
                             past_len, warmup=5, iters=20):
    """Strategy A: rotate full K (cache + new) at every step."""
    times = []
    for i in range(warmup + iters):
        mx.synchronize()
        t0 = time.perf_counter()
        k_full_unrot = mx.concatenate([k_cache_unrot, k_new], axis=2)
        v_full = mx.concatenate([v_cache, v_new], axis=2)
        out = flash_attention_rope(q, k_full_unrot, v_full, rotary_cos=cos,
                                   rotary_sin=sin, causal=True,
                                   cache_seqlens=past_len)
        mx.eval(out)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)
    return times


def bench_fc_rope_decode(q, k_new, v_new, k_cache_rot, v_cache, cos, sin,
                          past_len, warmup=5, iters=20):
    """Strategy B (FC): cache is pre-rotated; only rotate k_new."""
    times = []
    for i in range(warmup + iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out, _, _ = flash_attention_kvcache_rope_append(
            q, k_new, v_new, k_cache_rot, v_cache, cos, sin,
            cache_seqlens=past_len, causal=True,
        )
        mx.eval(out)
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)
    return times


def _stats(ts):
    avg = sum(ts) / len(ts)
    mn  = min(ts)
    mx_ = max(ts)
    return avg, mn, mx_


def main():
    parser = argparse.ArgumentParser(description="KV-cache RoPE append benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--past_len", type=int, default=512,
                        help="KV cache length (past tokens)")
    parser.add_argument("--new_tokens", type=int, default=1,
                        help="New tokens per decode step")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--dtype", choices=["f16", "bf16"], default="f16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = mx.float16 if args.dtype == "f16" else mx.bfloat16
    B, H, D = args.batch, args.heads, args.head_dim
    S = args.past_len
    N = args.new_tokens

    mx.random.seed(42)
    q          = mx.random.normal((B, H, N, D)).astype(dtype)
    k_new      = mx.random.normal((B, H, N, D)).astype(dtype)
    v_new      = mx.random.normal((B, H, N, D)).astype(dtype)
    k_cache_un = mx.random.normal((B, H, S, D)).astype(dtype)
    v_cache    = mx.random.normal((B, H, S, D)).astype(dtype)
    cos        = mx.random.normal((args.max_len, D // 2)).astype(mx.float32)
    sin        = mx.random.normal((args.max_len, D // 2)).astype(mx.float32)

    # Pre-rotate cache for strategy B.
    k_cache_rot = _apply_rope_mlx(k_cache_un, cos, sin, offset=0)
    mx.eval(q, k_new, v_new, k_cache_un, k_cache_rot, v_cache, cos, sin)

    print(f"\nKV-cache RoPE append benchmark")
    print(f"  B={B} H={H} D={D} past_len={S} new_tokens={N} dtype={args.dtype}")
    print(f"  warmup={args.warmup} iters={args.iters}")
    print()

    t_naive = bench_naive_rope_decode(
        q, k_new, v_new, k_cache_un, v_cache, cos, sin, S,
        warmup=args.warmup, iters=args.iters,
    )
    t_fc = bench_fc_rope_decode(
        q, k_new, v_new, k_cache_rot, v_cache, cos, sin, S,
        warmup=args.warmup, iters=args.iters,
    )

    a_avg, a_min, a_max = _stats(t_naive)
    b_avg, b_min, b_max = _stats(t_fc)

    print(f"Strategy A (naive re-rotate full K):  avg={a_avg:.2f}ms  "
          f"min={a_min:.2f}ms  max={a_max:.2f}ms")
    print(f"Strategy B (FC pre-rotated cache):    avg={b_avg:.2f}ms  "
          f"min={b_min:.2f}ms  max={b_max:.2f}ms")
    speedup = a_avg / b_avg if b_avg > 0 else float("inf")
    print(f"Speedup B/A (avg): {speedup:.2f}x")
    print()

    if speedup >= 1.05:
        print("FC strategy is faster as expected (rotation O(1) vs O(S)).")
    else:
        print("No significant speedup -- attention dominates over rotation cost.")


if __name__ == "__main__":
    main()
