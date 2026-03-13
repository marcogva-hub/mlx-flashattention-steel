#!/usr/bin/env python3
"""Paged/shared-prefix/splitfuse runtime decision matrix.

Benchmarks are executed in isolated subprocesses per route to avoid cache/env
cross-talk and to satisfy separate-process benchmarking constraints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from mlx_mfa import __version__, get_device_info


@dataclass(frozen=True)
class GQAProfile:
    name: str
    B: int
    H_q: int
    H_kv: int


def _median(values: list[float]) -> float:
    values = sorted(values)
    return float(values[len(values) // 2])


def _measure(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    mx.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return _median(times)


def _build_paged_pool(k: mx.array, v: mx.array, block_size: int):
    """Convert contiguous [B,H,S,D] KV to paged pool + metadata."""
    B, H, S, D = k.shape
    n_blocks = (S + block_size - 1) // block_size
    padded = n_blocks * block_size
    pad_len = padded - S

    if pad_len > 0:
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])

    k_blk = k.reshape(B, H, n_blocks, block_size, D).transpose(0, 2, 3, 1, 4)
    v_blk = v.reshape(B, H, n_blocks, block_size, D).transpose(0, 2, 3, 1, 4)
    k_pool = k_blk.reshape(B * n_blocks, block_size, H, D)
    v_pool = v_blk.reshape(B * n_blocks, block_size, H, D)

    table = mx.array(
        [[b * n_blocks + i for i in range(n_blocks)] for b in range(B)],
        dtype=mx.int32,
    )
    seq_lens = mx.array([S] * B, dtype=mx.int32)
    return k_pool, v_pool, table, seq_lens


def _paged_num_blocks(*, B: int, max_seq_len: int, block_size: int) -> int:
    return max(1, (B * max_seq_len + block_size - 1) // block_size)


def _classify_ratio(ratio: float) -> str:
    if ratio >= 1.05:
        return "clear_win"
    if ratio >= 1.01:
        return "maybe_win"
    if ratio >= 0.95:
        return "no_win"
    return "losing"


def _run_subprocess_mode(args: argparse.Namespace) -> None:
    from mlx_mfa import (
        create_decode_runtime,
        flash_attention,
        flash_attention_kvcache,
        flash_attention_paged,
        flash_attention_splitfuse,
        make_shared_prefix_cache,
    )

    mx.random.seed(args.seed)
    D = args.head_dim
    scale = 1.0 / math.sqrt(D)

    if args.family == "paged_step":
        B, H_q, H_kv = args.batch, args.heads_q, args.heads_kv
        N_q, N_cache = args.n_q, args.n_cache

        q = mx.random.normal((B, H_q, N_q, D)).astype(mx.float16)
        k_cache = mx.random.normal((B, H_kv, N_cache, D)).astype(mx.float16)
        v_cache = mx.random.normal((B, H_kv, N_cache, D)).astype(mx.float16)
        mx.eval(q, k_cache, v_cache)

        if args.route == "dense":
            fn = lambda: flash_attention_kvcache(
                q,
                k_cache,
                v_cache,
                scale=scale,
                causal=True,
            )
            path = "dense_kvcache"
        elif args.route == "paged":
            k_pool, v_pool, table, seq_lens = _build_paged_pool(
                k_cache,
                v_cache,
                block_size=args.block_size,
            )
            mx.eval(k_pool, v_pool, table, seq_lens)
            fn = lambda: flash_attention_paged(
                q,
                k_pool,
                v_pool,
                table,
                seq_lens,
                scale=scale,
                causal=True,
                block_size=args.block_size,
            )
            path = "paged_pool"
        else:
            raise ValueError(f"Unsupported route for paged_step: {args.route}")

        ms = _measure(fn, warmup=args.warmup, iters=args.iters)
        print(json.dumps({"ms": ms, "path": path}))
        return

    if args.family == "paged_setup":
        B, H_q, H_kv = args.batch, args.heads_q, args.heads_kv
        N_cache = args.n_cache

        q_pre = mx.random.normal((B, H_q, N_cache, D)).astype(mx.float16)
        k_pre = mx.random.normal((B, H_kv, N_cache, D)).astype(mx.float16)
        v_pre = mx.random.normal((B, H_kv, N_cache, D)).astype(mx.float16)
        mx.eval(q_pre, k_pre, v_pre)

        def _dense_setup():
            rt = create_decode_runtime(
                backend="dense",
                quantized_kv=False,
                B=B,
                H_q=H_q,
                H_kv=H_kv,
                D=D,
                max_seq_len=N_cache + 64,
            )
            return rt.prefill(q_pre, k_pre, v_pre, scale=scale)

        def _paged_setup():
            rt = create_decode_runtime(
                backend="paged",
                paged=True,
                quantized_kv=False,
                B=B,
                H_q=H_q,
                H_kv=H_kv,
                D=D,
                max_seq_len=N_cache + 64,
                num_blocks=_paged_num_blocks(
                    B=B,
                    max_seq_len=N_cache + 64,
                    block_size=args.block_size,
                ),
                block_size=args.block_size,
            )
            return rt.prefill(q_pre, k_pre, v_pre, scale=scale)

        if args.route == "dense":
            fn = _dense_setup
            path = "runtime_dense_prefill"
        elif args.route == "paged":
            fn = _paged_setup
            path = "runtime_paged_prefill"
        else:
            raise ValueError(f"Unsupported route for paged_setup: {args.route}")

        ms = _measure(fn, warmup=args.warmup, iters=args.iters)
        print(json.dumps({"ms": ms, "path": path}))
        return

    if args.family == "shared_prefix":
        B, H_q, H_kv = args.batch, args.heads_q, args.heads_kv
        N_prefix, N_suffix = args.n_prefix, args.n_suffix
        reuse = args.reuse

        prefix_q = mx.random.normal((B, H_q, N_prefix, D)).astype(mx.float16)
        prefix_k = mx.random.normal((B, H_kv, N_prefix, D)).astype(mx.float16)
        prefix_v = mx.random.normal((B, H_kv, N_prefix, D)).astype(mx.float16)

        suffix_q = [
            mx.random.normal((B, H_q, N_suffix, D)).astype(mx.float16)
            for _ in range(reuse)
        ]
        suffix_k = [
            mx.random.normal((B, H_kv, N_suffix, D)).astype(mx.float16)
            for _ in range(reuse)
        ]
        suffix_v = [
            mx.random.normal((B, H_kv, N_suffix, D)).astype(mx.float16)
            for _ in range(reuse)
        ]
        mx.eval(prefix_q, prefix_k, prefix_v, *suffix_q, *suffix_k, *suffix_v)

        def _baseline():
            out_last = None
            for i in range(reuse):
                flash_attention(
                    prefix_q,
                    prefix_k,
                    prefix_v,
                    scale=scale,
                    causal=True,
                )
                k_full = mx.concatenate([prefix_k, suffix_k[i]], axis=2)
                v_full = mx.concatenate([prefix_v, suffix_v[i]], axis=2)
                out_last = flash_attention(
                    suffix_q[i],
                    k_full,
                    v_full,
                    scale=scale,
                    causal=True,
                )
            return out_last

        def _reuse():
            _, kp, vp = make_shared_prefix_cache(
                prefix_q,
                prefix_k,
                prefix_v,
                scale=scale,
            )
            out_last = None
            for i in range(reuse):
                k_full = mx.concatenate([kp, suffix_k[i]], axis=2)
                v_full = mx.concatenate([vp, suffix_v[i]], axis=2)
                out_last = flash_attention(
                    suffix_q[i],
                    k_full,
                    v_full,
                    scale=scale,
                    causal=True,
                )
            return out_last

        if args.route == "baseline":
            fn = _baseline
            path = "repeat_full_prefill"
        elif args.route == "shared":
            fn = _reuse
            path = "shared_prefix_reuse"
        else:
            raise ValueError(f"Unsupported route for shared_prefix: {args.route}")

        ms = _measure(fn, warmup=args.warmup, iters=args.iters)
        print(json.dumps({"ms": ms, "path": path}))
        return

    if args.family == "splitfuse":
        Bp, Bd, H_q, H_kv = args.batch_prefill, args.batch_decode, args.heads_q, args.heads_kv
        Np, Nq, Ncache = args.n_prefill, args.n_q, args.n_cache

        q_prefill = mx.random.normal((Bp, H_q, Np, D)).astype(mx.float16)
        k_prefill = mx.random.normal((Bp, H_kv, Np, D)).astype(mx.float16)
        v_prefill = mx.random.normal((Bp, H_kv, Np, D)).astype(mx.float16)

        q_decode = mx.random.normal((Bd, H_q, Nq, D)).astype(mx.float16)
        k_decode = mx.random.normal((Bd, H_kv, Ncache, D)).astype(mx.float16)
        v_decode = mx.random.normal((Bd, H_kv, Ncache, D)).astype(mx.float16)
        mx.eval(q_prefill, k_prefill, v_prefill, q_decode, k_decode, v_decode)

        def _baseline():
            out_p = flash_attention(
                q_prefill,
                k_prefill,
                v_prefill,
                scale=scale,
                causal=True,
            )
            out_d = flash_attention(
                q_decode,
                k_decode,
                v_decode,
                scale=scale,
                causal=True,
            )
            return out_p, out_d

        def _splitfuse():
            return flash_attention_splitfuse(
                q_prefill,
                k_prefill,
                v_prefill,
                q_decode,
                k_decode,
                v_decode,
                scale=scale,
                causal=True,
            )

        if args.route == "baseline":
            fn = _baseline
            path = "separate_prefill_decode"
        elif args.route == "splitfuse":
            fn = _splitfuse
            path = "splitfuse_helper"
        else:
            raise ValueError(f"Unsupported route for splitfuse: {args.route}")

        ms = _measure(fn, warmup=args.warmup, iters=args.iters)
        print(json.dumps({"ms": ms, "path": path}))
        return

    raise ValueError(f"Unknown family: {args.family}")


def _run_case_subprocess(args: argparse.Namespace, **kwargs: Any) -> dict[str, Any]:
    cmd = [
        sys.executable,
        __file__,
        "--subprocess-mode",
    ]
    for k, v in kwargs.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed ({kwargs}):\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    payload = json.loads(proc.stdout.strip())
    return payload


def _paged_matrix(args: argparse.Namespace, profile: GQAProfile) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_rows: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []

    for D in (64, 128):
        for N_cache in (1024, 2048, 4096, 8192, 16384):
            dense_setup = _run_case_subprocess(
                args,
                family="paged_setup",
                route="dense",
                seed=args.seed,
                warmup=args.warmup,
                iters=args.iters,
                head_dim=D,
                batch=profile.B,
                heads_q=profile.H_q,
                heads_kv=profile.H_kv,
                n_cache=N_cache,
                block_size=args.block_size,
            )
            paged_setup = _run_case_subprocess(
                args,
                family="paged_setup",
                route="paged",
                seed=args.seed,
                warmup=args.warmup,
                iters=args.iters,
                head_dim=D,
                batch=profile.B,
                heads_q=profile.H_q,
                heads_kv=profile.H_kv,
                n_cache=N_cache,
                block_size=args.block_size,
            )
            setup_ratio = dense_setup["ms"] / paged_setup["ms"] if paged_setup["ms"] > 0 else 0.0
            setup_row = {
                "family": "paged_setup",
                "profile": profile.name,
                "D": D,
                "N_cache": N_cache,
                "dense_setup_ms": dense_setup["ms"],
                "paged_setup_ms": paged_setup["ms"],
                "paged_vs_dense": setup_ratio,
                "classification": _classify_ratio(setup_ratio),
            }
            setup_rows.append(setup_row)
            print(
                f"paged_setup D={D:<3} N_cache={N_cache:<5} "
                f"ratio={setup_ratio:.2f}x class={setup_row['classification']}"
            )

            for N_q in (1, 2, 4):
                dense = _run_case_subprocess(
                    args,
                    family="paged_step",
                    route="dense",
                    seed=args.seed,
                    warmup=args.warmup,
                    iters=args.iters,
                    head_dim=D,
                    batch=profile.B,
                    heads_q=profile.H_q,
                    heads_kv=profile.H_kv,
                    n_q=N_q,
                    n_cache=N_cache,
                    block_size=args.block_size,
                )
                paged = _run_case_subprocess(
                    args,
                    family="paged_step",
                    route="paged",
                    seed=args.seed,
                    warmup=args.warmup,
                    iters=args.iters,
                    head_dim=D,
                    batch=profile.B,
                    heads_q=profile.H_q,
                    heads_kv=profile.H_kv,
                    n_q=N_q,
                    n_cache=N_cache,
                    block_size=args.block_size,
                )
                ratio = dense["ms"] / paged["ms"] if paged["ms"] > 0 else 0.0
                row = {
                    "family": "paged_step",
                    "profile": profile.name,
                    "D": D,
                    "N_q": N_q,
                    "N_cache": N_cache,
                    "dense_ms": dense["ms"],
                    "paged_ms": paged["ms"],
                    "paged_vs_dense": ratio,
                    "dense_path": dense["path"],
                    "paged_path": paged["path"],
                    "classification": _classify_ratio(ratio),
                }
                step_rows.append(row)
                print(
                    f"paged_step  D={D:<3} N_q={N_q} N_cache={N_cache:<5} "
                    f"ratio={ratio:.2f}x class={row['classification']}"
                )

    return step_rows, setup_rows


def _shared_prefix_matrix(args: argparse.Namespace, profile: GQAProfile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for D in (64, 128):
        for N_prefix in (1024, 2048):
            for reuse in (2, 4):
                baseline = _run_case_subprocess(
                    args,
                    family="shared_prefix",
                    route="baseline",
                    seed=args.seed,
                    warmup=args.warmup,
                    iters=args.iters,
                    head_dim=D,
                    batch=profile.B,
                    heads_q=profile.H_q,
                    heads_kv=profile.H_kv,
                    n_prefix=N_prefix,
                    n_suffix=args.shared_suffix,
                    reuse=reuse,
                )
                shared = _run_case_subprocess(
                    args,
                    family="shared_prefix",
                    route="shared",
                    seed=args.seed,
                    warmup=args.warmup,
                    iters=args.iters,
                    head_dim=D,
                    batch=profile.B,
                    heads_q=profile.H_q,
                    heads_kv=profile.H_kv,
                    n_prefix=N_prefix,
                    n_suffix=args.shared_suffix,
                    reuse=reuse,
                )
                ratio = baseline["ms"] / shared["ms"] if shared["ms"] > 0 else 0.0
                row = {
                    "family": "shared_prefix",
                    "profile": profile.name,
                    "D": D,
                    "N_prefix": N_prefix,
                    "N_suffix": args.shared_suffix,
                    "reuse": reuse,
                    "baseline_ms": baseline["ms"],
                    "shared_ms": shared["ms"],
                    "shared_vs_baseline": ratio,
                    "classification": _classify_ratio(ratio),
                }
                rows.append(row)
                print(
                    f"shared_prefix D={D:<3} N_prefix={N_prefix:<5} reuse={reuse} "
                    f"ratio={ratio:.2f}x class={row['classification']}"
                )
    return rows


def _splitfuse_matrix(args: argparse.Namespace, profile: GQAProfile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for D in (64, 128):
        for N_prefill in (256, 1024):
            for N_q in (1, 4):
                baseline = _run_case_subprocess(
                    args,
                    family="splitfuse",
                    route="baseline",
                    seed=args.seed,
                    warmup=args.warmup,
                    iters=args.iters,
                    head_dim=D,
                    batch_prefill=profile.B,
                    batch_decode=profile.B,
                    heads_q=profile.H_q,
                    heads_kv=profile.H_kv,
                    n_prefill=N_prefill,
                    n_q=N_q,
                    n_cache=args.splitfuse_cache,
                )
                splitfuse = _run_case_subprocess(
                    args,
                    family="splitfuse",
                    route="splitfuse",
                    seed=args.seed,
                    warmup=args.warmup,
                    iters=args.iters,
                    head_dim=D,
                    batch_prefill=profile.B,
                    batch_decode=profile.B,
                    heads_q=profile.H_q,
                    heads_kv=profile.H_kv,
                    n_prefill=N_prefill,
                    n_q=N_q,
                    n_cache=args.splitfuse_cache,
                )
                ratio = baseline["ms"] / splitfuse["ms"] if splitfuse["ms"] > 0 else 0.0
                row = {
                    "family": "splitfuse",
                    "profile": profile.name,
                    "D": D,
                    "N_prefill": N_prefill,
                    "N_q": N_q,
                    "N_cache": args.splitfuse_cache,
                    "baseline_ms": baseline["ms"],
                    "splitfuse_ms": splitfuse["ms"],
                    "splitfuse_vs_baseline": ratio,
                    "classification": _classify_ratio(ratio),
                }
                rows.append(row)
                print(
                    f"splitfuse    D={D:<3} N_prefill={N_prefill:<4} N_q={N_q} "
                    f"ratio={ratio:.2f}x class={row['classification']}"
                )
    return rows


def _family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "clear_win": sum(1 for r in rows if r["classification"] == "clear_win"),
        "maybe_win": sum(1 for r in rows if r["classification"] == "maybe_win"),
        "no_win": sum(1 for r in rows if r["classification"] == "no_win"),
        "losing": sum(1 for r in rows if r["classification"] == "losing"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paged/shared-prefix/splitfuse benchmark matrix")
    ap.add_argument("--subprocess-mode", action="store_true")
    ap.add_argument("--family", type=str, default="paged_step")
    ap.add_argument("--route", type=str, default="dense")

    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--batch-prefill", type=int, default=1)
    ap.add_argument("--batch-decode", type=int, default=1)
    ap.add_argument("--heads-q", type=int, default=8)
    ap.add_argument("--heads-kv", type=int, default=4)
    ap.add_argument("--n-q", type=int, default=1)
    ap.add_argument("--n-cache", type=int, default=4096)
    ap.add_argument("--n-prefix", type=int, default=1024)
    ap.add_argument("--n-suffix", type=int, default=64)
    ap.add_argument("--reuse", type=int, default=4)
    ap.add_argument("--n-prefill", type=int, default=256)

    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--shared-suffix", type=int, default=64)
    ap.add_argument("--splitfuse-cache", type=int, default=4096)

    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        type=str,
        default="devnotes/paged_sharedprefix_matrix_latest.json",
    )
    args = ap.parse_args()

    if args.subprocess_mode:
        _run_subprocess_mode(args)
        return 0

    profile = GQAProfile(name="prod_gqa_b1_hq8_hkv4", B=1, H_q=8, H_kv=4)

    paged_step_rows, paged_setup_rows = _paged_matrix(args, profile)
    shared_rows = _shared_prefix_matrix(args, profile)
    splitfuse_rows = _splitfuse_matrix(args, profile)

    payload = {
        "date": time.strftime("%Y-%m-%d"),
        "mlx_mfa_version": __version__,
        "device": get_device_info(),
        "benchmark": {
            "profile": profile.__dict__,
            "dtype": "float16",
            "warmup": args.warmup,
            "iters": args.iters,
            "block_size": args.block_size,
        },
        "paged_step": {
            "rows": paged_step_rows,
            "counts": _family_counts(paged_step_rows),
        },
        "paged_setup": {
            "rows": paged_setup_rows,
            "counts": _family_counts(paged_setup_rows),
        },
        "shared_prefix": {
            "rows": shared_rows,
            "counts": _family_counts(shared_rows),
        },
        "splitfuse": {
            "rows": splitfuse_rows,
            "counts": _family_counts(splitfuse_rows),
        },
    }

    print("\nSummary counts:")
    for family in ("paged_step", "paged_setup", "shared_prefix", "splitfuse"):
        c = payload[family]["counts"]
        print(
            f"  {family:<13} clear={c['clear_win']} maybe={c['maybe_win']} "
            f"no_win={c['no_win']} losing={c['losing']}"
        )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
