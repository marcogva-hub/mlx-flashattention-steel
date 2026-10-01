"""B' = block B re-shaped by the COMPLETED block-A ranking (Marco re-cadrage).

For each block-B shape: knobs × (top-5 non-default tiles from block-A ∪ the default tile).
K=5 not 1 — knobs (esp MAX_THREADS) interact with tile size (M1 occupancy lesson); a knob
can rescue a mediocre bare tile, so don't over-prune. The default is ALWAYS included (it is
the promotion reference). B' is cost-sorted (small N first): if the deadline cuts, it cuts the
slowest, not the most informative.

The complement (knobs on non-top-5 tiles) → a NAMED night-fodder backlog. Nothing deleted;
it feeds NIGHT-2+. Idempotence by cell_id: any B cell already measured is skipped on resume
and counts for B' if it belongs.

  python -m research.na_autoresearch.runner.gen_bprime --results <run>/results.jsonl \
      --bprime bprime.jsonl --backlog night_fodder.jsonl --report bprime_sizing.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from research.na_autoresearch.runner.gen_night1 import (
    KNOB_VARIANTS, BH_KNOB, COST_S, sig, tiles as all_tiles,
)

TOPK = 5


def _tile_tag(t: dict) -> str:
    return "default" if not t else f"bq{t['BQ']}_bk{t['BK']}_wm{t['WM']}"


def _knob_tag(kv: dict) -> str:
    return "_".join(f"{k.split('_')[-1]}{v}" for k, v in kv.items())


def rank_block_a(results_path: str) -> dict:
    """(D,dtype,N,B,H) -> ranked list of NON-default tile dicts (fastest first)."""
    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    a = [r for r in rows if r.get("cell_id", "").startswith("A_")
         and r.get("engaged") and r.get("correct") and r.get("median_ms") is not None]
    groups: dict = defaultdict(list)
    for r in a:
        s = r["shape"]
        groups[(s["D"], s["dtype"], s["N"], s["B"], s["H"])].append(r)
    ranked = {}
    for k, rows_ in groups.items():
        rows_.sort(key=lambda r: r["median_ms"])
        ranked[k] = [r.get("tiles") or {} for r in rows_ if r.get("tiles")]  # non-default only
    return ranked


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--bprime", required=True)
    ap.add_argument("--backlog", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    ranked = rank_block_a(args.results)
    all_T = [{"BQ": b, "BK": k, "WM": w} for (b, k, w) in all_tiles()]
    done_B = set()
    for l in open(args.results):
        try:
            cid = json.loads(l).get("cell_id", "")
            if cid.startswith("B_"):
                done_B.add(cid)
        except Exception:
            pass

    bprime, fodder = [], []
    covered = 0
    shapes = 0
    for D in (64, 128):
        for dt in ("float16", "bfloat16"):
            for N in (2048, 4096, 8192, 16384, 32768):
                for (B, H) in BH_KNOB:
                    shapes += 1
                    s = sig(D, N, dt, B, H)
                    top = ranked.get((D, dt, N, B, H), [])[:TOPK]     # top-5 non-default
                    sel = [{}] + top                                  # default ∪ top-5
                    sel_tags = {_tile_tag(t) for t in sel}
                    for kv in KNOB_VARIANTS:
                        kt = _knob_tag(kv)
                        for t in all_T + [{}]:
                            tt = _tile_tag(t)
                            cid = f"B_dense_{s}_{tt}_{kt}"
                            cell = {"cell_id": cid, "kernel": "dense", "mode": "screening",
                                    "shape": {"B": B, "H": H, "N": N, "D": D, "dtype": dt, "causal": False},
                                    "tiles": t, "knobs": kv}
                            if tt in sel_tags:
                                bprime.append((N, cell))
                                if cid in done_B:
                                    covered += 1
                            else:
                                fodder.append(cell)

    bprime.sort(key=lambda nc: COST_S.get(nc[0], 99))                 # cost-sort: small N first
    bp = [c for _, c in bprime]
    with open(args.bprime, "w") as f:
        for c in bp:
            f.write(json.dumps(c) + "\n")
    with open(args.backlog, "w") as f:
        for c in fodder:
            f.write(json.dumps(c) + "\n")

    rep = {"bprime_cells": len(bp), "night_fodder_cells": len(fodder),
           "shapes": shapes, "topk": TOPK,
           "b_cells_already_measured": len(done_B), "bprime_cells_already_done": covered,
           "note": "B' = knobs × (top-5 non-default ∪ default) per shape, cost-sorted small-N-first; "
                   "fodder = knobs × non-top-5 tiles (NIGHT-2+ backlog, nothing deleted)."}
    if args.report:
        json.dump(rep, open(args.report, "w"), indent=1)
    print(f"B' = {len(bp)} cells (knobs × top{TOPK}∪default, {shapes} shapes, cost-sorted small-N first)")
    print(f"night-fodder backlog = {len(fodder)} cells")
    print(f"B cells already measured = {len(done_B)} (already in B': {covered})")


if __name__ == "__main__":
    main()
