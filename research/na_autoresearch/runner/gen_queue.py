"""Generate a declarative cell queue (JSONL) for the night runner.

Dense D64/D128 tile grid × prod shapes. Only VALID tiles are emitted
(BK %32==0, BQ %(WM*16)==0) so the runner never wastes a slot on a raise.
Extend with other families as their executors land (Phase 1+).

  python -m research.na_autoresearch.runner.gen_queue --out night1.jsonl [--coarse|--fine] \
      [--dtypes float16,bfloat16] [--Ns 2048,4096,8192] [--bh 2x8,1x4,1x12]
"""
from __future__ import annotations

import argparse
import json

# valid tile domains (source: mfa_v6_nax_primitive.cpp)
BQ_COARSE = [32, 64, 96, 128]
BK_COARSE = [32, 64]
WM_COARSE = [2, 4, 8]
BQ_FINE = [32, 48, 64, 80, 96, 112, 128, 160, 192]
BK_FINE = [32, 64, 96, 128]
WM_FINE = [2, 4, 8]


def valid(bq, bk, wm):
    return bk % 32 == 0 and bq > 0 and wm in (2, 4, 8) and bq % (wm * 16) == 0


def tile_grid(fine: bool):
    BQ, BK, WM = (BQ_FINE, BK_FINE, WM_FINE) if fine else (BQ_COARSE, BK_COARSE, WM_COARSE)
    out = []
    for wm in WM:
        for bq in BQ:
            for bk in BK:
                if valid(bq, bk, wm):
                    out.append((bq, bk, wm))
    return out


def gen(dtypes, Ns, bhs, fine: bool):
    cells = []
    tiles = tile_grid(fine)
    for D in (64, 128):
        for dt in dtypes:
            for N in Ns:
                for (B, H) in bhs:
                    shape = {"B": B, "H": H, "N": N, "D": D, "dtype": dt, "causal": False}
                    sig = f"D{D}_N{N}_{dt}_B{B}H{H}"
                    # the current default first (tiles={} → kernel's own default)
                    cells.append({"cell_id": f"dense_{sig}_default", "kernel": "dense",
                                  "mode": "screening", "shape": shape, "tiles": {}})
                    for (bq, bk, wm) in tiles:
                        cells.append({"cell_id": f"dense_{sig}_bq{bq}_bk{bk}_wm{wm}",
                                      "kernel": "dense", "mode": "screening",
                                      "shape": shape, "tiles": {"BQ": bq, "BK": bk, "WM": wm}})
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fine", action="store_true")
    ap.add_argument("--dtypes", default="float16,bfloat16")
    ap.add_argument("--Ns", default="2048,4096,8192")
    ap.add_argument("--bh", default="2x8,1x4,1x12")
    args = ap.parse_args(argv)
    dtypes = args.dtypes.split(",")
    Ns = [int(x) for x in args.Ns.split(",")]
    bhs = [tuple(int(y) for y in b.split("x")) for b in args.bh.split(",")]
    cells = gen(dtypes, Ns, bhs, args.fine)
    with open(args.out, "w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cells)} cells → {args.out} "
          f"({'fine' if args.fine else 'coarse'}; {len(tile_grid(args.fine))} tiles/shape + default; "
          f"D∈{{64,128}} × dtypes {dtypes} × Ns {Ns} × bh {bhs})")


if __name__ == "__main__":
    main()
