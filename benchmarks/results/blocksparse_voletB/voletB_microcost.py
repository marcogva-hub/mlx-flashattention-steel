#!/usr/bin/env python3
"""Volet B Phase 4 — composition micro-costs at Wan shapes.

Breaks sla_attention into its components (selection / sparse / linear / recomb) at
N∈{62752,144288}, B1H40, D128, topk 0.1, and reports the NET op-ratio after paying
the whole composition vs dense SDPA — does composition eat the ~8× skip? Stamp:
M5 Max · macOS 27 · MLX 0.31.2. Single GPU job.

  run <out.jsonl>
"""
from __future__ import annotations
import json, math, os, sys, time
os.environ.setdefault("MFA_SILENCE_NAX_WARNING", "1")
os.environ["MFA_SPARSE_NAX_EXTENDED"] = "1"

import mlx.core as mx
from mlx_mfa.attention import flash_attention_sparse
from mlx_mfa import sla as S


def _med(fn, warm=2, iters=6, slow=60.0):
    mx.eval(fn()); mx.synchronize()
    t0 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
    one = (time.perf_counter() - t0) * 1000.0
    if one > slow:
        warm, iters = 1, 3
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t1 = time.perf_counter(); mx.eval(fn()); mx.synchronize()
        ts.append((time.perf_counter() - t1) * 1000.0)
    ts.sort()
    return round(ts[len(ts) // 2], 3)


def run_cell(N):
    B, H, D = 1, 40, 128
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(0)
    f = lambda: (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
    q, k, v = f(), f(), f(); mx.eval(q, k, v)
    Wl = (mx.random.normal((D, D)) * 0.05).astype(mx.float16)
    bl = (mx.random.normal((D,)) * 0.01).astype(mx.float16); mx.eval(Wl, bl)
    projl = lambda x: x @ Wl.swapaxes(-1, -2) + bl

    # precompute the mask so "sparse alone" excludes selection
    sm = S._sla_block_map(q, k, 0.1, 128, 64)
    nq32 = (N + 31) // 32
    bm32 = S._expand_to_bt32(sm, 128, 64, nq32, nq32); mx.eval(bm32)
    dens = float(mx.mean(bm32.astype(mx.float32)).item())

    mx.reset_peak_memory()
    res = {"N": N, "density_actual": round(dens, 4)}
    res["dense_sdpa_ms"] = _med(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale))
    res["selection_ms"] = _med(lambda: S._expand_to_bt32(S._sla_block_map(q, k, 0.1, 128, 64), 128, 64, nq32, nq32))
    res["sparse_ms"] = _med(lambda: flash_attention_sparse(q, k, v, bm32, scale=scale, auto_pad=True))
    res["linear_ms"] = _med(lambda: projl(S._linear_term(q, k, v, "softmax")))
    res["sla_total_ms"] = _med(lambda: S.sla_attention(q, k, v, topk_ratio=0.1, proj_l=projl, scale=scale))
    res["peak_gb"] = round(mx.get_peak_memory() / 1e9, 3)
    # derived
    parts = res["selection_ms"] + res["sparse_ms"] + res["linear_ms"]
    res["recomb_plus_overhead_ms"] = round(res["sla_total_ms"] - parts, 3)
    res["net_ratio_vs_dense"] = round(res["dense_sdpa_ms"] / res["sla_total_ms"], 3)
    res["sparse_only_ratio_vs_dense"] = round(res["dense_sdpa_ms"] / res["sparse_ms"], 3)
    res["composition_tax_pct"] = round((res["sla_total_ms"] - res["sparse_ms"]) / res["sparse_ms"] * 100, 1)
    res["ts_epoch"] = int(time.time())
    return res


if __name__ == "__main__":
    out = sys.argv[2] if len(sys.argv) > 2 else "voletB_microcost.jsonl"
    for N in (62752, 144288):
        r = run_cell(N)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n"); f.flush()
        print(f"N={N:>6}: dense={r['dense_sdpa_ms']}ms sla_total={r['sla_total_ms']}ms "
              f"(sel={r['selection_ms']} sparse={r['sparse_ms']} lin={r['linear_ms']} "
              f"recomb={r['recomb_plus_overhead_ms']}) | NET {r['net_ratio_vs_dense']}x vs dense "
              f"(sparse-only {r['sparse_only_ratio_vs_dense']}x); composition tax {r['composition_tax_pct']}%", flush=True)
    print("microcost done", flush=True)
