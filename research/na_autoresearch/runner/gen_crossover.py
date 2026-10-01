"""Dense crossover-zoom queue (afternoon light day-mode).

Fine BQ×WM grid (the axis along which the optimum flips 32·32·2 → 64·32·4 → 128·32·8) at
INTERMEDIATE light N to pin where the tile optimum crosses, per (D, dtype). Bare tiles only
(this is the tile-threshold map, not a knob sweep). LIGHT ONLY: N ≤ 8192 (the ~2 s cost cap;
the N16384/32768 crossover class belongs to the night). Cost-sorted small-N first.

  python -m research.na_autoresearch.runner.gen_crossover --out crossover.jsonl
"""
from __future__ import annotations

import argparse
import json

BQ = [32, 48, 64, 80, 96, 112, 128, 160, 192]
BK = [32, 64]                       # 32 is the dominant/default; 64 as a check
WM = [2, 4, 8]
N_LIGHT = [3072, 4096, 6144, 8192]  # intermediate + standard light bands; excludes N16384/32768


def valid(bq, bk, wm):
    return bk % 32 == 0 and wm in (2, 4, 8) and bq % (wm * 16) == 0


def tiles():
    return [(b, k, w) for w in WM for b in BQ for k in BK if valid(b, k, w)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    T = tiles()
    cells = []
    for N in N_LIGHT:                # cost-sorted: small N first
        for D in (64, 128):
            for dt in ("float16", "bfloat16"):
                B, H = 2, 8
                shape = {"B": B, "H": H, "N": N, "D": D, "dtype": dt, "causal": False}
                s = f"D{D}_N{N}_{dt}_B{B}H{H}"
                cells.append({"cell_id": f"X_dense_{s}_default", "kernel": "dense",
                              "mode": "screening", "shape": shape, "tiles": {}})
                for (bq, bk, wm) in T:
                    cells.append({"cell_id": f"X_dense_{s}_bq{bq}_bk{bk}_wm{wm}", "kernel": "dense",
                                  "mode": "screening", "shape": shape,
                                  "tiles": {"BQ": bq, "BK": bk, "WM": wm}})
    with open(args.out, "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cells)} crossover cells → {args.out} "
          f"({len(T)} tiles+default × N{N_LIGHT} × D{{64,128}} × {{fp16,bf16}}, light only)")


if __name__ == "__main__":
    main()
