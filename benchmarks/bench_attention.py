"""Benchmark: mlx-mfa vs MLX SDPA (dense and block-sparse)."""

import argparse
import math
import time

import mlx.core as mx

from mlx_mfa import flash_attention, flash_attention_sparse
from mlx_mfa import make_causal_block_mask, make_sliding_window_mask
from mlx_mfa.attention import _fallback_sdpa


def benchmark_one(B, H, N, D, causal, dtype, n_warmup=10, n_iter=20):
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)

    results = {}
    for name, fn in [
        ("mlx_sdpa", lambda: _fallback_sdpa(q, k, v, scale, causal)),
        ("mlx_mfa", lambda: flash_attention(q, k, v, scale=scale, causal=causal)),
    ]:
        # Warmup: first call compiles Metal shaders and MLX compute graphs.
        for _ in range(n_warmup):
            out = fn()
            mx.eval(out)
        mx.synchronize()

        times = []
        for _ in range(n_iter):
            mx.synchronize()
            t0 = time.perf_counter()
            out = fn()
            mx.eval(out)
            mx.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        median_ms = sorted(times)[len(times) // 2] * 1000
        results[name] = median_ms

    return results


def benchmark_sparse_one(B, H, N, D, block_mask, dtype, n_warmup=10, n_iter=20):
    """Compare flash_attention_sparse vs SDPA with the SAME block mask.

    Methodology fix (audit H-09): the SDPA arm previously ran UNMASKED dense
    attention while the sparse arm applied the block mask — different math, so the
    "speedup" mixed a semantic difference into the timing.  Now SDPA gets the
    block mask expanded to an additive bias (equal semantics), and the sparse
    output is checked against an independent fp32 block-masked oracle BEFORE any
    timing (a wrong arm makes the ratio meaningless — Lesson #11).
    """
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)

    # Expand the TILE-level block mask to an element bias so SDPA matches the
    # sparse kernel's semantics (tile on/off, no element-causal within a tile).
    NQ, NK = block_mask.shape[-2], block_mask.shape[-1]
    em = mx.repeat(mx.repeat(block_mask.astype(mx.float32), N // NQ, -2), N // NK, -1)
    while em.ndim < 4:
        em = em[None]
    bias = mx.where(em > 0, mx.array(0.0), mx.array(-1e9, mx.float32)).astype(dtype)
    mx.eval(q, k, v, block_mask, bias)

    def _sparse():
        return flash_attention_sparse(q, k, v, block_mask, scale=scale, causal=False)

    def _sdpa_masked():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)

    # fp32-oracle output check BEFORE timing (independent of both kernels).
    o_sp, o_sd = _sparse(), _sdpa_masked()
    mx.eval(o_sp, o_sd)
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    s = (qf @ kf.swapaxes(-1, -2)) * scale
    s = mx.where(em > 0, s, mx.array(-1e30, mx.float32))
    ref = mx.softmax(s, -1) @ vf
    rel = float(mx.max(mx.abs(o_sp.astype(mx.float32) - ref)) / (mx.max(mx.abs(ref)) + 1e-6))
    if not math.isfinite(rel) or rel > 5e-2:
        raise RuntimeError(
            f"sparse bench oracle check FAILED (D={D} N={N}): rel_err {rel:.3e} vs fp32 "
            f"block-masked oracle — the sparse output is wrong, the timing ratio is meaningless.")

    results = {}
    for name, fn in [
        ("mlx_sdpa_masked", _sdpa_masked),
        ("mlx_mfa_sparse", _sparse),
    ]:
        for _ in range(n_warmup):
            out = fn()
            mx.eval(out)
        mx.synchronize()

        times = []
        for _ in range(n_iter):
            mx.synchronize()
            t0 = time.perf_counter()
            out = fn()
            mx.eval(out)
            mx.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        median_ms = sorted(times)[len(times) // 2] * 1000
        results[name] = median_ms

    return results


def main():
    # Phantom-bench gate (audit H7/H-09): if mlx-mfa acceleration isn't live, the
    # MFA arm would silently be SDPA → fake parity. Fail loud before timing.
    from _bench_guard import require_accel_or_die
    require_accel_or_die("bench_attention.py")

    parser = argparse.ArgumentParser(description="mlx-mfa benchmark")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--seq-len", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=["f16", "bf16", "f32"], default="f16")
    # Block-sparse modes (mutually exclusive)
    parser.add_argument("--sparse-causal", action="store_true",
                        help="Benchmark block-sparse causal mask vs dense SDPA")
    parser.add_argument("--sliding-window", type=int, default=0,
                        help="Benchmark sliding-window block mask of given size vs dense SDPA")
    args = parser.parse_args()

    dtype_map = {"f16": mx.float16, "bf16": mx.bfloat16, "f32": mx.float32}
    dtype = dtype_map[args.dtype]

    if args.sparse_causal or args.sliding_window > 0:
        # Sparse benchmark
        mode = "sparse-causal" if args.sparse_causal else f"sliding-window={args.sliding_window}"
        print(f"B={args.batch} H={args.heads} mode={mode} dtype={args.dtype}")
        print(f"{'D':>5} {'N':>6} {'SDPA ms':>9} {'Sparse ms':>11} {'Speedup':>8} {'Active%':>8}")
        print("-" * 55)
        for D in args.head_dim:
            for N in args.seq_len:
                if args.sparse_causal:
                    mask = make_causal_block_mask(N, head_dim=D)
                else:
                    mask = make_sliding_window_mask(N, args.sliding_window, head_dim=D)
                # Compute sparsity
                mx.eval(mask)
                active = mx.sum(mask.astype(mx.int32)).item()
                total = mask.shape[0] * mask.shape[1]
                active_pct = 100.0 * active / total

                r = benchmark_sparse_one(args.batch, args.heads, N, D, mask, dtype)
                sdpa = r["mlx_sdpa_masked"]  # H-09: now mask-equal to the sparse arm
                sparse = r["mlx_mfa_sparse"]
                speedup = sdpa / sparse if sparse > 0 else float("inf")
                print(f"{D:>5} {N:>6} {sdpa:>9.2f} {sparse:>11.2f} {speedup:>7.2f}x {active_pct:>7.1f}%")
    else:
        # Dense benchmark
        print(f"B={args.batch} H={args.heads} causal={args.causal} dtype={args.dtype}")
        print(f"{'D':>5} {'N':>6} {'SDPA ms':>9} {'MFA ms':>9} {'Speedup':>8}")
        print("-" * 45)
        for D in args.head_dim:
            for N in args.seq_len:
                r = benchmark_one(args.batch, args.heads, N, D,
                                  args.causal, dtype)
                sdpa = r["mlx_sdpa"]
                mfa = r["mlx_mfa"]
                speedup = sdpa / mfa if mfa > 0 else float("inf")
                print(f"{D:>5} {N:>6} {sdpa:>9.2f} {mfa:>9.2f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
