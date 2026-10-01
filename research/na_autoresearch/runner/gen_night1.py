"""Generate the dimensioned NIGHT-1 queue + a sizing report.

NIGHT-1 = dense D64+D128 EXHAUSTIVE screening (Marco's priority family), sized to >= 8 h
GPU by the measured per-N cell cost. Blocks, in priority order:
  A  tile grid × full shape envelope (base pass, kernel-default knobs)
  B  the two correctness-neutral perf knobs (RELAXED_PRECISION, UNROLL_MODE) × top shapes
  (auto-zoom refines the winners; the SPARSE-BT32 backlog is Phase-1 — needs a sparse
   executor, flagged in the plan, NOT emitted here as no-executor cells.)

Per-cell cost model from the Phase-0 calibration (mean wall, screening, in-process):
  N2048=0.344 s  N4096=0.220 s  N8192=0.852 s  N16384≈1.70 s (≈2×N8192, extrapolated)
The report prints projected hours per block so the queue is provably >= 8 h.

  python -m research.na_autoresearch.runner.gen_night1 --out night1.jsonl [--report night1_sizing.json]
"""
from __future__ import annotations

import argparse
import json

# measured screening cost per cell by N (seconds) — calibration rep_run, 2026-08-09
COST_S = {2048: 0.344, 4096: 0.220, 8192: 0.852, 16384: 1.70, 32768: 3.40}

BQ = [32, 48, 64, 80, 96, 112, 128, 160, 192]
BK = [32, 64, 96, 128]          # 128 = the M5-doctrine target (untested; current default 32)
WM = [2, 4, 8]
Ns = [2048, 4096, 8192, 16384, 32768]  # incl. the large-N regime where NAX matters most
DTYPES = ["float16", "bfloat16"]
BH = [(2, 8), (1, 4), (1, 12)]
BH_KNOB = [(2, 8), (1, 12)]             # block-B applies knobs across two occupancy regimes
# orthogonal correctness-neutral perf knobs (census): non-default values + a few combos
KNOB_VARIANTS = [
    {"MFA_V6_RELAXED_PRECISION": 0},                 # default 1 (relaxed on)
    {"MFA_V6_UNROLL_MODE": "none"},                  # default full
    {"MFA_V6_UNROLL_MODE": "2"},
    {"MFA_V6_UNROLL_MODE": "4"},
    {"MFA_V6_MAX_THREADS": 256},                     # occupancy hint buckets
    {"MFA_V6_MAX_THREADS": 512},
    {"MFA_V6_RELAXED_PRECISION": 0, "MFA_V6_UNROLL_MODE": "none"},  # interaction probe
]


def valid(bq, bk, wm):
    return bk % 32 == 0 and wm in (2, 4, 8) and bq % (wm * 16) == 0  # empirical domain (TQ>1 OK, verified fp32)


def tiles():
    return [(bq, bk, wm) for wm in WM for bq in BQ for bk in BK if valid(bq, bk, wm)]


def sig(D, N, dt, B, H):
    return f"D{D}_N{N}_{dt}_B{B}H{H}"


def gen():
    T = tiles()
    cells, cost = [], 0.0
    # Block A — base tile grid × full envelope
    for D in (64, 128):
        for dt in DTYPES:
            for N in Ns:
                for (B, H) in BH:
                    shape = {"B": B, "H": H, "N": N, "D": D, "dtype": dt, "causal": False}
                    s = sig(D, N, dt, B, H)
                    cells.append({"cell_id": f"A_dense_{s}_default", "kernel": "dense",
                                  "mode": "screening", "shape": shape, "tiles": {}}); cost += COST_S[N]
                    for (bq, bk, wm) in T:
                        cells.append({"cell_id": f"A_dense_{s}_bq{bq}_bk{bk}_wm{wm}", "kernel": "dense",
                                      "mode": "screening", "shape": shape,
                                      "tiles": {"BQ": bq, "BK": bk, "WM": wm}}); cost += COST_S[N]
    blockA = len(cells)
    # Block B — orthogonal knob product × the FULL tile grid × two occupancy regimes.
    # RELAXED_PRECISION / UNROLL_MODE / MAX_THREADS interact with the tile, so they are
    # swept across the whole grid, not just a few tiles.
    knob_tiles = [{}] + [{"BQ": bq, "BK": bk, "WM": wm} for (bq, bk, wm) in T]
    for kv in KNOB_VARIANTS:
        ktag = "_".join(f"{k.split('_')[-1]}{v}" for k, v in kv.items())
        for D in (64, 128):
            for dt in DTYPES:
                for N in Ns:
                    for (B, H) in BH_KNOB:
                        shape = {"B": B, "H": H, "N": N, "D": D, "dtype": dt, "causal": False}
                        s = sig(D, N, dt, B, H)
                        for t in knob_tiles:
                            ttag = "default" if not t else f"bq{t['BQ']}_bk{t['BK']}_wm{t['WM']}"
                            cells.append({"cell_id": f"B_dense_{s}_{ttag}_{ktag}", "kernel": "dense",
                                          "mode": "screening", "shape": shape, "tiles": t, "knobs": kv})
                            cost += COST_S[N]
    return cells, blockA, cost


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    cells, blockA, cost = gen()
    with open(args.out, "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
    rep = {"total_cells": len(cells), "block_A_base_cells": blockA, "block_B_knob_cells": len(cells) - blockA,
           "valid_tiles": len(tiles()) + 1, "projected_gpu_hours": round(cost / 3600, 2),
           "cost_model_s_per_cell_by_N": COST_S, "envelope": {"D": [64, 128], "dtypes": DTYPES,
           "Ns": Ns, "bh": [f"{b}x{h}" for b, h in BH]},
           "knob_variants": KNOB_VARIANTS, "note": "auto-zoom refines winners on top; SPARSE-BT32 backlog is Phase-1 (needs sparse executor)."}
    if args.report:
        json.dump(rep, open(args.report, "w"), indent=1)
    print(f"wrote {len(cells)} cells → {args.out}")
    print(f"  block A (tiles×envelope): {blockA} cells | block B (knobs): {len(cells)-blockA} cells")
    print(f"  valid tiles: {len(tiles())} (+default) | projected GPU: {rep['projected_gpu_hours']} h "
          f"(>= 8 h target: {'OK' if rep['projected_gpu_hours'] >= 8 else 'SHORT — widen'})")


if __name__ == "__main__":
    main()
