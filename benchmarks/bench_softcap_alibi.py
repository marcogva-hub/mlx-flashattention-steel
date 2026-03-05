"""Benchmark: softcap and ALiBi overhead vs baseline flash attention."""

import argparse
import math
import time

import mlx.core as mx

from mlx_mfa import flash_attention
from mlx_mfa.attention import _fallback_sdpa


def _bench(fns: dict, n_warmup: int = 10, n_iter: int = 20) -> dict:
    """Run a dictionary of {name: callable} and return {name: median_ms}."""
    results = {}
    for name, fn in fns.items():
        for _ in range(n_warmup):
            mx.eval(fn())
        mx.synchronize()

        times = []
        for _ in range(n_iter):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(fn())
            mx.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results[name] = sorted(times)[len(times) // 2] * 1000
    return results


def bench_softcap(B: int, H: int, N: int, D: int, causal: bool, dtype) -> dict:
    """Softcap overhead: sdpa_ref, sdpa_softcap_ref, mfa_plain, mfa_softcap."""
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)
    cap = 50.0
    mx.eval(q, k, v)

    return _bench({
        "sdpa_ref":       lambda: _fallback_sdpa(q, k, v, scale, causal),
        "sdpa_softcap":   lambda: _fallback_sdpa(q, k, v, scale, causal, softcap=cap),
        "mfa_plain":      lambda: flash_attention(q, k, v, scale=scale, causal=causal),
        "mfa_softcap":    lambda: flash_attention(q, k, v, scale=scale, causal=causal, softcap=cap),
    })


def bench_alibi(B: int, H: int, N: int, D: int, causal: bool, dtype) -> dict:
    """ALiBi overhead: sdpa_ref, sdpa_alibi_ref, mfa_plain, mfa_alibi."""
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    k = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    v = mx.random.normal(shape=(B, H, N, D)).astype(dtype)
    scale = 1.0 / math.sqrt(D)
    # Standard ALiBi slopes: m_h = 2^(-8*h/H) for h in [1..H]
    slopes = mx.array([2 ** (-8 * (h + 1) / H) for h in range(H)], dtype=mx.float32)
    mx.eval(q, k, v, slopes)

    try:
        from mlx_mfa.attention import flash_attention_alibi as _fa_alibi  # type: ignore
        has_alibi = True
    except ImportError:
        has_alibi = False

    def _sdpa_alibi_ref():
        """ALiBi reference: add linear bias in log domain before softmax."""
        S = k.shape[2]
        pos = mx.arange(S, dtype=mx.float32)
        # bias[h, q, k] = slope_h * (k_pos - q_pos)
        q_pos = mx.arange(N, dtype=mx.float32)[:, None]
        k_pos = mx.arange(S, dtype=mx.float32)[None, :]
        # shape [H, N, S] → [1, H, N, S]
        bias = (slopes[:, None, None] * (k_pos[None] - q_pos[None])).astype(q.dtype)
        bias = bias[None]  # [1, H, N, S]
        scores = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * scale + bias
        return mx.matmul(mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype), v)

    fns = {
        "sdpa_ref":       lambda: _fallback_sdpa(q, k, v, scale, causal),
        "sdpa_alibi_ref": _sdpa_alibi_ref,
        "mfa_plain":      lambda: flash_attention(q, k, v, scale=scale, causal=causal),
    }
    if has_alibi:
        from mlx_mfa.attention import flash_attention_alibi as _fa_alibi  # noqa: F811
        fns["mfa_alibi"] = lambda: _fa_alibi(q, k, v, slopes, scale=scale, causal=causal)

    return _bench(fns)


def print_table(title: str, results: dict, baseline: str = "sdpa_ref") -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    base = results.get(baseline, None)
    col_w = max(len(n) for n in results) + 2
    print(f"  {'name':<{col_w}}  {'ms':>8}  {'vs ' + baseline:>14}")
    print(f"  {'-' * col_w}  {'-' * 8}  {'-' * 14}")
    for name, ms in results.items():
        ratio = f"{ms / base:.2f}x" if base else "—"
        print(f"  {name:<{col_w}}  {ms:>8.3f}  {ratio:>14}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark softcap and ALiBi features")
    parser.add_argument("--B", type=int, default=2, help="batch size")
    parser.add_argument("--H", type=int, default=8, help="num heads")
    parser.add_argument("--N", type=int, default=4096, help="sequence length")
    parser.add_argument("--D", type=int, default=128, help="head dim")
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--dtype", choices=["f16", "bf16", "f32"], default="f16")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=20)
    args = parser.parse_args()

    dtype_map = {"f16": mx.float16, "bf16": mx.bfloat16, "f32": mx.float32}
    dtype = dtype_map[args.dtype]

    header = (f"B={args.B} H={args.H} N={args.N} D={args.D} "
              f"{'causal' if args.causal else 'non-causal'} {args.dtype}")
    print(f"\nConfig: {header}")

    sc = bench_softcap(args.B, args.H, args.N, args.D, args.causal, dtype)
    print_table(f"Softcap  | {header}", sc)

    al = bench_alibi(args.B, args.H, args.N, args.D, args.causal, dtype)
    print_table(f"ALiBi    | {header}", al)


if __name__ == "__main__":
    main()
