"""bench_paged_kv.py — Paged KV attention performance comparison.

Compares three strategies for paged-KV autoregressive decode:

  A (gather+attend):  Metal gather kernel materialises contiguous K/V,
                      then runs flash_attention on the result.
  B (paged-STEEL):    Kernel-level paged KV — K/V tiles read directly from
                      the page pool in the STEEL forward kernel (Track FD).
  C (flash-decode):   Same gather, but N_q <= 4 activates Flash Decoding
                      (split-KV two-phase) for better SM parallelism.

Usage::

    python benchmarks/bench_paged_kv.py
    python benchmarks/bench_paged_kv.py --seq-len 16384 --D 128
    python benchmarks/bench_paged_kv.py --help
"""

from __future__ import annotations

import argparse
import time
import math

import mlx.core as mx


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_pool(k, v, block_size: int):
    """Pack contiguous [B,H,S,D] K/V into paged pool + metadata."""
    B, H, S, D = k.shape
    n_blk = (S + block_size - 1) // block_size
    pad_len = n_blk * block_size - S
    k_pad = mx.pad(k, [(0,0),(0,0),(0,pad_len),(0,0)]) if pad_len else k
    v_pad = mx.pad(v, [(0,0),(0,0),(0,pad_len),(0,0)]) if pad_len else v
    k_blk = k_pad.reshape(B, H, n_blk, block_size, D).transpose(0, 2, 3, 1, 4)
    v_blk = v_pad.reshape(B, H, n_blk, block_size, D).transpose(0, 2, 3, 1, 4)
    pool_k = k_blk.reshape(B * n_blk, block_size, H, D)
    pool_v = v_blk.reshape(B * n_blk, block_size, H, D)
    table = mx.array([[b * n_blk + i for i in range(n_blk)] for b in range(B)],
                     dtype=mx.int32)
    lens = mx.array([S] * B, dtype=mx.int32)
    return pool_k, pool_v, table, lens


def bench_fn(fn, warmup: int = 5, iters: int = 30) -> list:
    times = []
    for i in range(warmup + iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)
    return times


def stats(ms: list) -> tuple:
    mean = sum(ms) / len(ms)
    s = sorted(ms)
    med = s[len(s) // 2]
    return mean, med


# ── benchmark body ────────────────────────────────────────────────────────────

def run(B: int, H: int, N_q: int, S: int, D: int,
        block_size: int, dtype,
        warmup: int, iters: int) -> None:
    from mlx_mfa import flash_attention
    from mlx_mfa._ext import mfa_paged_steel_forward, mfa_paged_kv_gather

    mx.random.seed(42)
    scale = 1.0 / math.sqrt(D)

    q       = mx.random.normal((B, H, N_q, D)).astype(dtype)
    k_dense = mx.random.normal((B, H, S, D)).astype(dtype)
    v_dense = mx.random.normal((B, H, S, D)).astype(dtype)
    mx.synchronize()

    pool_k, pool_v, table, lens = _build_pool(k_dense, v_dense, block_size)
    mx.synchronize()

    # Pre-gather for strategy C (avoid counting gather time in flash decode bench)
    K_gather = mfa_paged_kv_gather(pool_k, table, lens, S)
    V_gather = mfa_paged_kv_gather(pool_v, table, lens, S)
    mx.synchronize()

    # Strategy A: gather + attend (standard paged path)
    def fn_gather_attend():
        K = mfa_paged_kv_gather(pool_k, table, lens, S)
        V = mfa_paged_kv_gather(pool_v, table, lens, S)
        return flash_attention(q, K, V, scale=scale, causal=False)

    # Strategy B: kernel-level paged STEEL (no separate gather)
    def fn_paged_steel():
        O, _L = mfa_paged_steel_forward(
            q, pool_k, pool_v, table, lens,
            scale=scale, causal=False, block_size=block_size)
        return O

    # Strategy C: pre-gathered + flash_attention (activates Flash Decode when N_q<=4)
    def fn_flash_decode():
        return flash_attention(q, K_gather, V_gather, scale=scale, causal=False)

    a_ms  = bench_fn(fn_gather_attend, warmup, iters)
    b_ms  = bench_fn(fn_paged_steel,   warmup, iters)
    c_ms  = bench_fn(fn_flash_decode,  warmup, iters)

    a_mean, a_med = stats(a_ms)
    b_mean, b_med = stats(b_ms)
    c_mean, c_med = stats(c_ms)

    print(f"\n{'='*68}")
    dtype_str = 'f16' if dtype == mx.float16 else 'bf16'
    print(f"  B={B} H={H} N_q={N_q} S={S} D={D} bs={block_size} {dtype_str}")
    print(f"{'='*68}")
    print(f"  {'Strategy':<32} {'mean ms':>8} {'median ms':>10} {'vs A':>7}")
    print(f"  {'-'*58}")
    print(f"  {'A  gather + attend':<32} {a_mean:>8.3f} {a_med:>10.3f} {'—':>7}")
    print(f"  {'B  paged STEEL kernel':<32} {b_mean:>8.3f} {b_med:>10.3f} "
          f"{a_med/b_med:>6.2f}x")
    print(f"  {'C  pre-gathered flash decode':<32} {c_mean:>8.3f} {c_med:>10.3f} "
          f"{a_med/c_med:>6.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paged KV attention benchmark")
    ap.add_argument("--B",          type=int,  default=1)
    ap.add_argument("--H",          type=int,  default=8)
    ap.add_argument("--N-q",        type=int,  default=1,
                    help="Query length (decode step size)")
    ap.add_argument("--seq-lens",   type=int,  nargs="+",
                    default=[1024, 4096, 16384],
                    help="KV sequence lengths to sweep")
    ap.add_argument("--D",          type=int,  default=128)
    ap.add_argument("--block-size", type=int,  default=64)
    ap.add_argument("--bf16",       action="store_true")
    ap.add_argument("--warmup",     type=int,  default=5)
    ap.add_argument("--iters",      type=int,  default=30)
    args = ap.parse_args()

    dtype = mx.bfloat16 if args.bf16 else mx.float16
    print(f"\nPaged KV attention benchmark")
    print(f"Device: {mx.default_device()}")

    N_q = getattr(args, 'N_q', None) or 1
    for S in args.seq_lens:
        run(B=args.B, H=args.H, N_q=N_q, S=S, D=args.D,
            block_size=args.block_size, dtype=dtype,
            warmup=args.warmup, iters=args.iters)

    print()


if __name__ == "__main__":
    main()
