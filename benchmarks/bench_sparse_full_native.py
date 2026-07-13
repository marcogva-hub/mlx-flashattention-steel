"""Fresh-process sparse full-native backward benchmark.

The native arm is public and opt-in gated.  The baseline is an explicit
SDPA-vjp implementation with the same BT64 mask, so the two arms are not
vacuous.  Run each arm in a separate foreground process and repeat with the
opposite arm order.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import mlx.core as mx
import numpy as np

from mlx_mfa import _dispatch_trace, flash_attention_sparse
from mlx_mfa.attention import _block_mask_to_float_bias


def _cosine(a, b) -> float:
    x = np.array(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    y = np.array(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("native", "sdpa"), required=True)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--density", type=float, default=0.1)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    B, H, N, D, BT = 1, 2, args.n, args.d, 64
    mx.random.seed(20260713 + D + int(args.causal))
    q = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    dO = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    rng = np.random.default_rng(2026 + D + int(args.causal))
    mask_np = rng.random((N // BT, N // BT)) < args.density
    np.fill_diagonal(mask_np, True)
    block_mask = mx.array(mask_np)
    scale = 1.0 / math.sqrt(D)
    mx.eval(q, k, v, dO, block_mask)

    bias = _block_mask_to_float_bias(
        block_mask, N, N, scale_q_dtype=q.dtype, tile_q=BT, tile_k=BT
    )
    if args.causal:
        bias = bias + mx.triu(
            mx.full((N, N), -float("inf"), dtype=q.dtype), k=1
        )

    def native_loss(qi, ki, vi):
        return (flash_attention_sparse(
            qi, ki, vi, block_mask, scale=scale, causal=args.causal
        ) * dO).sum()

    def sdpa_loss(qi, ki, vi):
        out = mx.fast.scaled_dot_product_attention(
            qi, ki, vi, scale=scale, mask=bias
        )
        return (out * dO).sum()

    def call():
        if args.arm == "native":
            return mx.grad(native_loss, argnums=(0, 1, 2))(q, k, v)
        return mx.grad(sdpa_loss, argnums=(0, 1, 2))(q, k, v)

    with _dispatch_trace.capture() as trace:
        probe = call()
        mx.eval(*probe)
        mx.synchronize()
    if args.arm == "native":
        if not any(t[0] == "v6nax_sparse_lse" for t in trace):
            raise RuntimeError(f"native arm did not engage sparse V6NAX LSE: {trace}")
    elif any(t[0] == "v6nax_sparse_lse" for t in trace):
        raise RuntimeError(f"SDPA arm unexpectedly engaged sparse V6NAX: {trace}")

    # The explicit SDPA arm is the correction oracle for the comparative
    # benchmark; the LSE correctness grid separately checks a fp32 oracle.
    oracle = mx.grad(sdpa_loss, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*oracle)
    correction = {
        name: _cosine(a, b)
        for name, a, b in zip(("dQ", "dK", "dV"), probe, oracle)
    }
    if min(correction.values()) < 0.999:
        raise RuntimeError(f"gradient correction failed: {correction}")
    for _ in range(2):
        mx.eval(*call()); mx.synchronize()
    samples = []
    for _ in range(args.runs):
        start = time.perf_counter()
        mx.eval(*call()); mx.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    result = {
        "arm": args.arm,
        "shape": {"B": B, "H": H, "N": N, "D": D, "BT": BT,
                  "density": args.density, "causal": args.causal},
        "trace": trace,
        "correction_cos_vs_sdpa_vjp": correction,
        "samples_ms": samples,
        "median_ms": float(np.median(samples)),
        "runs": args.runs,
        "method": "fresh process, two-arm external order; native uses public full-native opt-in",
    }
    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
