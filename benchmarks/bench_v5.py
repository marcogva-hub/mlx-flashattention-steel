"""Benchmark: STEEL V5 vs V2 vs SDPA.

Each kernel path runs in a SEPARATE subprocess to avoid ShaderCache
contamination between V2 and V5 measurements.

Usage:
    python benchmarks/bench_v5.py
    python benchmarks/bench_v5.py --dtype bf16
    python benchmarks/bench_v5.py --head-dim 128 --causal true
"""

import argparse
import os
import subprocess
import sys
import time


def _run_one_kernel(D, N, causal, dtype_str, n_warmup, n_iter):
    """Run single kernel timing; print median ms to stdout."""
    import mlx.core as mx
    from mlx_mfa.attention import _fallback_sdpa, _mfa_forward

    kernel = os.environ.get("BENCH_KERNEL", "sdpa")
    dtype = mx.bfloat16 if dtype_str == "bf16" else mx.float16

    mx.random.seed(42)
    B, H = 2, 8
    q = mx.random.normal([B, H, N, D]).astype(dtype)
    k = mx.random.normal([B, H, N, D]).astype(dtype)
    v = mx.random.normal([B, H, N, D]).astype(dtype)
    scale = D ** -0.5
    mx.synchronize()

    if kernel == "sdpa":
        fn = lambda: _fallback_sdpa(q, k, v, scale, causal)
    else:
        fn = lambda: _mfa_forward(q, k, v, scale, causal)

    import mlx.core as _mx_local
    for _ in range(n_warmup):
        _mx_local.eval(fn())
    _mx_local.synchronize()

    times = []
    for _ in range(n_iter):
        _mx_local.synchronize()
        t0 = time.perf_counter()
        _mx_local.eval(fn())
        _mx_local.synchronize()
        times.append(time.perf_counter() - t0)

    ms = sorted(times)[len(times) // 2] * 1000
    print(f"{ms:.3f}")


def _bench_subprocess(D, N, causal, dtype_str, kernel, n_warmup, n_iter):
    """Run one kernel in an isolated subprocess; return median ms."""
    env = os.environ.copy()
    env["BENCH_KERNEL"] = kernel
    if kernel == "sdpa":
        env["MFA_DISABLE_V2"] = "1"
    elif kernel == "v2":
        env.pop("MFA_ENABLE_V5", None)
        env.pop("MFA_DISABLE_V2", None)
    elif kernel == "v5":
        env["MFA_ENABLE_V5"] = "1"
        env.pop("MFA_DISABLE_V2", None)

    cmd = [
        sys.executable, __file__,
        "--subprocess-mode",
        "--head-dim", str(D),
        "--seq-len",  str(N),
        "--causal",   "1" if causal else "0",
        "--dtype",    dtype_str,
        "--warmup",   str(n_warmup),
        "--iters",    str(n_iter),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed kernel={kernel} D={D} N={N}:\n{r.stderr[-800:]}")
    return float(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--seq-len",  type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--causal",   default="both",
                    help="true | false | both")
    ap.add_argument("--dtype",    default="f16")
    ap.add_argument("--warmup",   type=int, default=5)
    ap.add_argument("--iters",    type=int, default=15)
    ap.add_argument("--subprocess-mode", action="store_true")
    args = ap.parse_args()

    if args.subprocess_mode:
        causal = args.causal == "1"
        _run_one_kernel(args.head_dim[0], args.seq_len[0], causal,
                        args.dtype, args.warmup, args.iters)
        return

    causal_map = {"true": [True], "false": [False], "both": [True, False]}
    causal_vals = causal_map.get(args.causal.lower(), [True, False])

    print(f"\nSTEEL V5 vs V2 vs SDPA  (B=2 H=8 dtype={args.dtype})")
    print(f"{'D':>4}  {'N':>6}  {'Mode':<8}  "
          f"{'SDPA':>9}  {'V2':>9}  {'V5':>9}  "
          f"{'V5/SDPA':>8}  {'V5/V2':>7}")
    print("-" * 75)

    for D in args.head_dim:
        for N in args.seq_len:
            for causal in causal_vals:
                mode = "causal" if causal else "dense"
                try:
                    ms_sdpa = _bench_subprocess(D, N, causal, args.dtype,
                                                "sdpa", args.warmup, args.iters)
                    ms_v2   = _bench_subprocess(D, N, causal, args.dtype,
                                                "v2",   args.warmup, args.iters)
                    ms_v5   = _bench_subprocess(D, N, causal, args.dtype,
                                                "v5",   args.warmup, args.iters)
                    v5_sdpa = ms_sdpa / ms_v5
                    v5_v2   = ms_v2   / ms_v5
                    print(f"{D:>4}  {N:>6}  {mode:<8}  "
                          f"{ms_sdpa:>8.2f}ms  {ms_v2:>8.2f}ms  "
                          f"{ms_v5:>8.2f}ms  "
                          f"{v5_sdpa:>7.2f}x  {v5_v2:>6.2f}x")
                except Exception as exc:
                    print(f"{D:>4}  {N:>6}  {mode:<8}  ERROR: {exc}")
    print()


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
