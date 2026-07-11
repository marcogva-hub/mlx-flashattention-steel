#!/usr/bin/env python3
"""Paged-varlen multi-sequence benchmark.

Measures the previously-blocked multi-sequence case with a corrected fixture:

  A) raw page-native PagedVarlenForward (`_ext.mfa_paged_varlen_forward`)
  B) materialize-then-flash (paged gather K/V -> padded batched SDPA)
  C) per-sequence SDPA oracle arm (contiguous K/V, one call per sequence)
  D) current public `flash_attention_paged_varlen` dispatch

Each config is oracle-checked before any ratio is reported.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

sys.path.insert(0, ".")

import mlx  # noqa: E402
from mlx_mfa import _ext, flash_attention_paged_varlen  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "benchmarks" / "results"


@dataclass(frozen=True)
class Scenario:
    name: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("decode4_hetero_kv", (1, 1, 1, 1), (512, 1024, 2048, 4096)),
    Scenario("decode8_hetero_kv", (1, 1, 1, 1, 1, 1, 1, 1), (256, 512, 768, 1024, 1536, 2048, 3072, 4096)),
    Scenario("prefill4_hetero_qkv", (8, 16, 32, 64), (512, 1024, 2048, 4096)),
    Scenario("mixed8_hetero_qkv", (1, 4, 1, 8, 1, 2, 1, 16), (256, 512, 768, 1024, 1280, 1536, 1792, 2048)),
)

QUICK_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("decode4_hetero_kv", (1, 1, 1, 1), (512, 1024, 2048, 4096)),
    Scenario("prefill4_hetero_qkv", (8, 16, 32, 64), (512, 1024, 2048, 4096)),
)


def _dtype_from_name(name: str):
    if name == "fp16":
        return mx.float16
    if name == "bf16":
        return mx.bfloat16
    raise ValueError(f"unknown dtype {name}")


def _dtype_name(dtype) -> str:
    return "bf16" if dtype == mx.bfloat16 else "fp16"


def _cos_np(a, b) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    den = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / den) if den else 1.0


def _stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    p95_idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_idx],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "mean_ms": statistics.fmean(samples),
        "n": len(samples),
        "samples_ms": samples,
    }


def _tile_offsets(q_lens: tuple[int, ...], bq: int = 32) -> mx.array:
    offsets = [0]
    for ql in q_lens:
        offsets.append(offsets[-1] + (ql + bq - 1) // bq)
    return mx.array(offsets, dtype=mx.int32)


def _build_paged_pool(k_seqs, v_seqs, block_size: int, dtype):
    B = len(k_seqs)
    H_kv = k_seqs[0].shape[1]
    D = k_seqs[0].shape[3]
    blocks_per_seq = [(int(k.shape[2]) + block_size - 1) // block_size for k in k_seqs]
    total_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)
    pool_k = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float32)
    pool_v = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float32)
    table = np.full((B, max_blocks), -1, dtype=np.int32)
    lens = np.zeros((B,), dtype=np.int32)
    base = 0
    for b, (k_b, v_b) in enumerate(zip(k_seqs, v_seqs)):
        k_np = np.array(k_b.astype(mx.float32))[0].transpose(1, 0, 2)
        v_np = np.array(v_b.astype(mx.float32))[0].transpose(1, 0, 2)
        S = k_np.shape[0]
        lens[b] = S
        for lb in range(blocks_per_seq[b]):
            s0 = lb * block_size
            s1 = min(S, s0 + block_size)
            table[b, lb] = base + lb
            pool_k[base + lb, : s1 - s0] = k_np[s0:s1]
            pool_v[base + lb, : s1 - s0] = v_np[s0:s1]
        base += blocks_per_seq[b]
    return (
        mx.array(pool_k).astype(dtype),
        mx.array(pool_v).astype(dtype),
        mx.array(table, dtype=mx.int32),
        mx.array(lens, dtype=mx.int32),
    )


def _causal_bias_np(q_len: int, kv_len: int) -> np.ndarray:
    q_pos = max(0, kv_len - q_len) + np.arange(q_len)[:, None]
    k_pos = np.arange(kv_len)[None, :]
    return np.where(k_pos <= q_pos, 0.0, -np.inf).astype(np.float32)


def _padded_bias(q_lens: tuple[int, ...], kv_lens: tuple[int, ...], causal: bool, dtype) -> mx.array:
    B = len(q_lens)
    max_q = max(q_lens)
    max_kv = max(kv_lens)
    bias = np.full((B, 1, max_q, max_kv), -np.inf, dtype=np.float32)
    for b, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        if causal:
            local = _causal_bias_np(ql, kl)
        else:
            local = np.zeros((ql, kl), dtype=np.float32)
        bias[b, 0, :ql, :kl] = local
        if ql < max_q:
            bias[b, 0, ql:, :kl] = 0.0
    return mx.array(bias).astype(dtype)


def _per_seq_sdpa(q_seqs, k_seqs, v_seqs, scale: float, causal: bool, fp32: bool = False):
    parts = []
    for q_i, k_i, v_i in zip(q_seqs, k_seqs, v_seqs):
        if fp32:
            q_i, k_i, v_i = q_i.astype(mx.float32), k_i.astype(mx.float32), v_i.astype(mx.float32)
        mask = None
        if causal:
            mask_dtype = mx.float32 if fp32 else q_i.dtype
            mask = mx.array(_causal_bias_np(q_i.shape[2], k_i.shape[2])).astype(mask_dtype)
        parts.append(mx.fast.scaled_dot_product_attention(q_i, k_i, v_i, scale=scale, mask=mask))
    return mx.concatenate(parts, axis=2)


def _repack_padded(out_pad: mx.array, q_lens: tuple[int, ...]) -> mx.array:
    return mx.concatenate([out_pad[i : i + 1, :, :ql, :] for i, ql in enumerate(q_lens)], axis=2)


@dataclass
class Fixture:
    scenario: Scenario
    D: int
    dtype_name: str
    dtype: object
    causal: bool
    H_q: int
    H_kv: int
    block_size: int
    scale: float
    q_seqs: list
    k_seqs: list
    v_seqs: list
    q_pack: mx.array
    q_pad: mx.array
    cu_q: mx.array
    tile_offsets: mx.array
    pool_k: mx.array
    pool_v: mx.array
    table: mx.array
    lens: mx.array
    mask_pad: mx.array
    ref32: mx.array


def make_fixture(scenario: Scenario, D: int, dtype, causal: bool, seed: int, block_size: int) -> Fixture:
    H_q, H_kv = 8, 4
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(seed)
    q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(dtype) for ql in scenario.q_lens]
    k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(dtype) for kl in scenario.kv_lens]
    v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(dtype) for kl in scenario.kv_lens]
    mx.eval(*q_seqs, *k_seqs, *v_seqs)
    q_pack = mx.concatenate(q_seqs, axis=2)
    max_q = max(scenario.q_lens)
    q_pad = mx.concatenate(
        [
            mx.pad(q_i, [(0, 0), (0, 0), (0, max_q - q_i.shape[2]), (0, 0)])
            for q_i in q_seqs
        ],
        axis=0,
    )
    cu_vals = [0]
    for ql in scenario.q_lens:
        cu_vals.append(cu_vals[-1] + ql)
    cu_q = mx.array(cu_vals, dtype=mx.int32)
    tile_offsets = _tile_offsets(scenario.q_lens)
    pool_k, pool_v, table, lens = _build_paged_pool(k_seqs, v_seqs, block_size, dtype)
    mask_pad = _padded_bias(scenario.q_lens, scenario.kv_lens, causal, dtype)
    ref32 = _per_seq_sdpa(q_seqs, k_seqs, v_seqs, scale, causal, fp32=True)
    mx.eval(q_pack, q_pad, cu_q, tile_offsets, pool_k, pool_v, table, lens, mask_pad, ref32)
    return Fixture(
        scenario=scenario,
        D=D,
        dtype_name=_dtype_name(dtype),
        dtype=dtype,
        causal=causal,
        H_q=H_q,
        H_kv=H_kv,
        block_size=block_size,
        scale=scale,
        q_seqs=q_seqs,
        k_seqs=k_seqs,
        v_seqs=v_seqs,
        q_pack=q_pack,
        q_pad=q_pad,
        cu_q=cu_q,
        tile_offsets=tile_offsets,
        pool_k=pool_k,
        pool_v=pool_v,
        table=table,
        lens=lens,
        mask_pad=mask_pad,
        ref32=ref32,
    )


def arm_page_native(fx: Fixture):
    out, _lse = _ext.mfa_paged_varlen_forward(
        fx.q_pack,
        fx.pool_k,
        fx.pool_v,
        fx.cu_q,
        fx.tile_offsets,
        fx.table,
        fx.lens,
        fx.scale,
        fx.causal,
        fx.block_size,
    )
    return out


def arm_materialize_padded_flash(fx: Fixture):
    max_kv = max(fx.scenario.kv_lens)
    K = _ext.mfa_paged_kv_gather(fx.pool_k, fx.table, fx.lens, max_kv)
    V = _ext.mfa_paged_kv_gather(fx.pool_v, fx.table, fx.lens, max_kv)
    out_pad = mx.fast.scaled_dot_product_attention(
        fx.q_pad, K, V, scale=fx.scale, mask=fx.mask_pad
    )
    return _repack_padded(out_pad, fx.scenario.q_lens)


def arm_sdpa_per_seq(fx: Fixture):
    return _per_seq_sdpa(fx.q_seqs, fx.k_seqs, fx.v_seqs, fx.scale, fx.causal, fp32=False)


def arm_public_current(fx: Fixture):
    return flash_attention_paged_varlen(
        fx.q_pack,
        fx.pool_k,
        fx.pool_v,
        fx.table,
        fx.lens,
        fx.cu_q,
        max_seqlen_q=max(fx.scenario.q_lens),
        scale=fx.scale,
        causal=fx.causal,
        block_size=fx.block_size,
    )


ARMS: dict[str, Callable[[Fixture], mx.array]] = {
    "page_native_raw": arm_page_native,
    "materialize_padded_flash": arm_materialize_padded_flash,
    "sdpa_per_sequence": arm_sdpa_per_seq,
    "public_current": arm_public_current,
}


def validate_arms(fx: Fixture) -> dict[str, dict]:
    ref = np.array(fx.ref32)
    out: dict[str, dict] = {}
    for name, fn in ARMS.items():
        arr = fn(fx)
        mx.eval(arr)
        got = np.array(arr.astype(mx.float32))
        cos = _cos_np(got, ref)
        max_abs = float(np.max(np.abs(got - ref)))
        finite = bool(np.isfinite(got).all())
        out[name] = {"cos": cos, "max_abs": max_abs, "finite": finite}
        if not finite or cos < 0.999:
            raise RuntimeError(
                f"{fx.scenario.name} D={fx.D} {fx.dtype_name} {name} failed "
                f"oracle validation: finite={finite} cos={cos:.6f} max_abs={max_abs:.6g}"
            )
    return out


def bench_arm(fx: Fixture, fn: Callable[[Fixture], mx.array], sessions: int, warmup: int, iters: int) -> dict:
    session_stats = []
    all_samples: list[float] = []
    for _session in range(sessions):
        for _ in range(warmup):
            arr = fn(fx)
            mx.eval(arr)
            mx.synchronize()
        samples = []
        for _ in range(iters):
            mx.synchronize()
            t0 = time.perf_counter()
            arr = fn(fx)
            mx.eval(arr)
            mx.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        session_stats.append(_stats(samples))
        all_samples.extend(samples)
    overall = _stats(all_samples)
    overall["sessions"] = session_stats
    return overall


def run_config(fx: Fixture, sessions: int, warmup: int, iters: int) -> dict:
    validation = validate_arms(fx)
    arms = {}
    for name, fn in ARMS.items():
        arms[name] = bench_arm(fx, fn, sessions, warmup, iters)
    native = arms["page_native_raw"]["median_ms"]
    materialized = arms["materialize_padded_flash"]["median_ms"]
    per_seq = arms["sdpa_per_sequence"]["median_ms"]
    public = arms["public_current"]["median_ms"]
    return {
        "scenario": fx.scenario.name,
        "q_lens": list(fx.scenario.q_lens),
        "kv_lens": list(fx.scenario.kv_lens),
        "num_seqs": len(fx.scenario.q_lens),
        "total_q": sum(fx.scenario.q_lens),
        "max_q": max(fx.scenario.q_lens),
        "max_kv": max(fx.scenario.kv_lens),
        "D": fx.D,
        "dtype": fx.dtype_name,
        "causal": fx.causal,
        "validation": validation,
        "arms": arms,
        "ratios": {
            "page_native_over_materialize": native / materialized,
            "page_native_over_sdpa_per_seq": native / per_seq,
            "public_over_materialize": public / materialized,
            "materialize_over_sdpa_per_seq": materialized / per_seq,
        },
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Paged Varlen Multi-Seq Benchmark",
        "",
        f"- commit: `{payload['commit']}`",
        f"- mlx: `{payload['mlx_version']}`",
        f"- device: `{payload['device']}`",
        f"- sessions/arm: `{payload['sessions']}`, warmup: `{payload['warmup']}`, iters/session: `{payload['iters']}`",
        "",
        "| scenario | causal | dtype | D | page-native ms | materialize ms | sdpa/seq ms | native/materialize | public/materialize |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        arms = row["arms"]
        ratios = row["ratios"]
        lines.append(
            f"| {row['scenario']} | {'yes' if row['causal'] else 'no'} | "
            f"{row['dtype']} | {row['D']} | "
            f"{arms['page_native_raw']['median_ms']:.4f} | "
            f"{arms['materialize_padded_flash']['median_ms']:.4f} | "
            f"{arms['sdpa_per_sequence']['median_ms']:.4f} | "
            f"{ratios['page_native_over_materialize']:.3f}x | "
            f"{ratios['public_over_materialize']:.3f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_head() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()


def _mlx_version() -> str:
    return getattr(mlx, "__version__", None) or importlib.metadata.version("mlx")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=["quick", "final"], default="quick")
    ap.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"], choices=["fp16", "bf16"])
    ap.add_argument("--dims", nargs="+", type=int, default=[64, 128], choices=[64, 128])
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--causal-modes", nargs="+", default=["causal", "noncausal"], choices=["causal", "noncausal"])
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    scenarios = QUICK_SCENARIOS if args.profile == "quick" else SCENARIOS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.out_dir / f"paged_varlen_multiseq_{args.profile}_{stamp}.json"
    md_path = args.out_dir / f"paged_varlen_multiseq_{args.profile}_{stamp}.md"

    results = []
    for scenario in scenarios:
        for dtype_name in args.dtypes:
            dtype = _dtype_from_name(dtype_name)
            for D in args.dims:
                for causal_name in args.causal_modes:
                    causal = causal_name == "causal"
                    seed = (
                        1000
                        + D
                        + 17 * len(scenario.q_lens)
                        + (1 if dtype_name == "bf16" else 0)
                        + (101 if causal else 0)
                    )
                    fx = make_fixture(
                        scenario=scenario,
                        D=D,
                        dtype=dtype,
                        causal=causal,
                        seed=seed,
                        block_size=args.block_size,
                    )
                    row = run_config(fx, args.sessions, args.warmup, args.iters)
                    results.append(row)
                    ratios = row["ratios"]
                    print(
                        f"{scenario.name:22s} {causal_name:9s} {dtype_name:4s} D={D:3d} "
                        f"native/materialize={ratios['page_native_over_materialize']:.3f}x "
                        f"public/materialize={ratios['public_over_materialize']:.3f}x"
                    )

    payload = {
        "profile": args.profile,
        "created_at": stamp,
        "commit": _git_head(),
        "python": sys.executable,
        "python_version": sys.version,
        "mlx_version": _mlx_version(),
        "platform": platform.platform(),
        "device": str(mx.default_device()),
        "sessions": args.sessions,
        "warmup": args.warmup,
        "iters": args.iters,
        "arms": list(ARMS.keys()),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09

    _phantom_gate(__file__)
    main()
