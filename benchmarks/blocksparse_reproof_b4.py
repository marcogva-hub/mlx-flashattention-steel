"""Remediation 2026-09, Phase B4 — re-proof of the Volet A §4 correctness gates at the
campaign's target scale with PER-ROW MAGNITUDE gates (the historical "7/7 PASS" used a
global cosine, blind to the U1 row scaling; they are void).

No timings (brief: the ratios do not move, the semantics did).  Per cell:
  * route / which-binary from the dispatch trace (v6nax_sparse; "auto_pad (kv_valid_len)"
    for non-aligned N),
  * exact fp32 rows (CPU) for the tail query block + first + random rows on a subset of
    heads — a full [N, N] oracle at N=144k is infeasible, sampled rows are exact,
  * gates: max-abs, per-row norm ratio, per-row max-abs, global cosine (complement).

Usage:  .venv/bin/python benchmarks/blocksparse_reproof_b4.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx_mfa  # noqa: E402
from mlx_mfa import _dispatch_trace as dt  # noqa: E402
from mlx_mfa.attention import flash_attention_sparse  # noqa: E402
from tests.sparse_gates import row_gate_report  # noqa: E402

GATES = {mx.float16: dict(max_abs=1e-2, norm_tol=1e-2, cos_min=0.999),
         mx.bfloat16: dict(max_abs=3e-2, norm_tol=2e-2, cos_min=0.999)}
HEADS_CHECKED = 4
ROWS_RANDOM = 64

# (N, B·H, D, dtype, causal, density) — campaign target shapes (Volet A hardened table)
CELLS = [
    (16384, 40, 128, mx.float16, False, 0.10),
    (62730, 40, 128, mx.float16, False, 0.10),    # real Wan count -> auto_pad 62752
    (144279, 40, 128, mx.float16, False, 0.10),   # real Wan count -> auto_pad 144288
    (62730, 40, 128, mx.bfloat16, False, 0.10),
    (32768, 40, 128, mx.float16, True, 0.10),
    (62730, 40, 128, mx.float16, False, 0.50),
    (4100, 40, 64, mx.float16, False, 0.10),
]


def band_mask(nb: int, density: float) -> mx.array:
    """Sliding-window block band with ~`density` fill (the campaign's 'sliding' masks),
    plus the ragged final key block active for every row (worst case for U1)."""
    half = max(0, int(round(density * nb / 2)))
    i = mx.arange(nb)[:, None]
    j = mx.arange(nb)[None, :]
    m = mx.abs(i - j) <= half
    m = m | (j == nb - 1)
    return m


def oracle_rows(q, k, v, bm, rows, heads, causal):
    """Exact fp32 rows on CPU: [1, len(heads), len(rows), D]."""
    N, S, D = q.shape[2], k.shape[2], q.shape[3]
    sc = 1.0 / math.sqrt(D)
    out = []
    with mx.stream(mx.cpu):
        kk = k[:, heads].astype(mx.float32)
        vv = v[:, heads].astype(mx.float32)
        for r in rows:
            keep = mx.repeat(bm[r // 32], 32)[:S]
            if causal:
                keep = keep & (mx.arange(S) <= r + max(0, S - N))
            s = (q[:, heads, r:r + 1].astype(mx.float32) @ mx.swapaxes(kk, -1, -2)) * sc
            s = mx.where(keep, s, float("-inf"))
            out.append(mx.softmax(s, axis=-1) @ vv)
        o = mx.concatenate(out, axis=2)
        mx.eval(o)
    return o


def run_cell(N, BH, D, dtype, causal, density, seed):
    mx.random.seed(seed)
    q, k, v = (mx.random.normal((1, BH, N, D)).astype(dtype) for _ in range(3))
    nb = -(-N // 32)
    bm = band_mask(nb, density)
    mx.eval(q, k, v, bm)
    t0 = time.time()
    with dt.capture() as cap:
        o = flash_attention_sparse(q, k, v, bm, causal=causal, auto_pad=True)
        mx.eval(o)
    wall = time.time() - t0              # recorded for the log only — NOT a perf claim
    rng = np.random.default_rng(seed)
    rows = sorted(set(range(N - 32, N)) | set(range(16))
                  | set(rng.integers(0, N, size=ROWS_RANDOM).tolist()))
    heads = sorted(rng.choice(BH, size=min(HEADS_CHECKED, BH), replace=False).tolist())
    ref = oracle_rows(q, k, v, bm, rows, heads, causal)
    got = o[:, heads][:, :, mx.array(rows)]
    rep = row_gate_report(got, ref)
    g = GATES[dtype]
    ok = (rep["finite"] and rep["max_abs"] < g["max_abs"]
          and rep["worst_row_norm_dev"] < g["norm_tol"] and rep["cos"] > g["cos_min"])
    route = [(r[0], r[1]) for r in cap]
    engaged = any(r[0] == "v6nax_sparse" for r in cap)
    padded = any("kv_valid" in r[1] for r in cap)
    return dict(N=N, BH=BH, D=D, dtype=str(dtype), causal=causal, density=density,
                density_real=float(mx.mean(bm.astype(mx.float32))), route=route,
                engaged_v6nax=engaged, auto_pad_kv_valid=padded,
                rows_checked=len(rows), heads_checked=heads, gates=g, report=rep,
                PASS=bool(ok and engaged), wall_s_not_a_benchmark=round(wall, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results/blocksparse_B4/reproof_b4.json")
    args = ap.parse_args()
    os.environ["MFA_SPARSE_NAX_EXTENDED"] = "1"      # the path the campaign / SLA use
    results = []
    for i, cell in enumerate(CELLS):
        res = run_cell(*cell, seed=1000 + i)
        print(f"{'PASS' if res['PASS'] else 'FAIL'} N={res['N']} BH={res['BH']} D={res['D']} "
              f"{res['dtype']} causal={res['causal']} d={res['density_real']:.3f} "
              f"engaged={res['engaged_v6nax']} padded={res['auto_pad_kv_valid']} "
              f"max_abs={res['report']['max_abs']:.2e} "
              f"row_norm_dev={res['report']['worst_row_norm_dev']:.2e} "
              f"cos={res['report']['cos']:.6f}", flush=True)
        results.append(res)
        mx.clear_cache()
    meta = dict(mlx=mx.__version__, mlx_mfa=mlx_mfa.__version__,
                device=mlx_mfa.get_device_info().get("device_name", "?"),
                date=time.strftime("%Y-%m-%d"), cells=len(results),
                all_pass=all(r["PASS"] for r in results))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(meta=meta, results=results), f, indent=1)
    print(json.dumps(meta))
    return 0 if meta["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
