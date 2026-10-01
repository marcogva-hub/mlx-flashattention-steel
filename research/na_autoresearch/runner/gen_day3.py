"""DAY-3 queue (2026-09-30): head floors -> Block 1 (B·H16 map) -> Block 2 (table thresholds)
-> Block 3.1 (night-1 fodder remainder) -> Block 3.2 (Stage-2 INDECIDABLE re-screen).
Blocks in VALUE order (an interrupted run keeps the most important part); 3.3 (surprise
zoom) is generated at run time by day3_report.day3_zoom.

  python -m research.na_autoresearch.runner.gen_day3 --out <queue.jsonl> [--block1-tiers 1,2,3,4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AR = Path("/Users/marcomarcelino/code/mlx-mfa-v2/benchmarks/results/autoresearch")
S2 = AR / "stage2-20260812_104813"
DT = {"fp16": "float16", "bf16": "bfloat16"}


def _solo(cid, block, B, H, N, D, dtype, tiles, knobs=None, procs=5, kind="solo", entry=None):
    return {"cell_id": cid, "kernel": kind, "block": block, "procs": procs, "entry": entry,
            "shape": {"B": B, "H": H, "N": N, "D": D, "dtype": dtype, "causal": False},
            "tiles": tiles, "knobs": knobs or {}}


def _gate(block, mask_kind, value, causal, N, D, dtype, B=2, H=8, kind="gate_remap", n_pairs=None):
    cfg = {"mask_kind": mask_kind, "causal": causal, "B": B, "H": H, "N": N, "D": D,
           "dtype": dtype, "seed": 20260713}
    if mask_kind == "sliding":
        cfg["window"] = int(value); tag = f"sw{int(value)}"
    else:
        cfg["density"] = round(float(value), 4); tag = f"rnd{float(value):.4f}"
    cid = f"B1_{tag}_{'c' if causal else 'nc'}_{dtype}_b{B}_h{H}_n{N}_d{D}"
    if kind == "gate_null":
        cid = f"FLOOR_sparse_AA_{tag}_{dtype}_b{B}_h{H}_n{N}_d{D}"
    cell = {"cell_id": cid, "kernel": kind, "block": block, "cfg": cfg}
    if n_pairs:
        cell["n_pairs"] = n_pairs
    return cell


def head_floors():
    cells = []
    for D, B, H, N in ((128, 2, 8, 4096), (128, 1, 12, 4096), (128, 1, 4, 4096),
                       (64, 1, 12, 8192), (64, 2, 8, 8192)):
        cells.append(_solo(f"FLOOR_dense_AA_D{D}_B{B}H{H}_N{N}_float16", "0-floors",
                           B, H, N, D, "float16", {}, kind="solo_null"))
    cells.append(_gate("0-floors", "random", 0.15, False, 8192, 128, "fp16",
                       kind="gate_null", n_pairs=6))
    return cells


DENSITIES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
NS = (4096, 6144, 8192)


def block1(tiers):
    """B·H16 (B2H8), NAX forced (public contract, extended opt-in). Pruned to what the July
    contract admitted: random for causal (make_random_mask), sliding windows {128,256,512}
    non-causal, bf16 = the encoded July bf16 templates. Tiers in value order."""
    cells = []
    if 1 in tiers:   # the core map: random non-causal fp16
        cells += [_gate("1-bh16", "random", d, False, n, D, "fp16")
                  for D in (128, 64) for n in NS for d in DENSITIES]
    if 2 in tiers:   # causal (random archetype, the harness' causal mask)
        cells += [_gate("1-bh16", "random", d, True, n, D, "fp16")
                  for D in (128, 64) for n in NS for d in DENSITIES]
    if 3 in tiers:   # sliding archetype (July windows)
        cells += [_gate("1-bh16", "sliding", w, False, n, D, "fp16")
                  for D in (128, 64) for n in NS for w in (128, 256, 512)]
    if 4 in tiers:   # bf16: the July encoded templates
        for kind, val, causal, D in (("sliding", 128, False, 128), ("random", 0.15, False, 128),
                                     ("random", 0.10, True, 64), ("random", 0.10, True, 128)):
            cells += [_gate("1-bh16", kind, val, causal, n, D, "bf16") for n in NS]
    return cells


def block2():
    """Crossover zoom of the CONFIRMED tile entries (tile-only, the table's form)."""
    t3232 = {"BQ": 32, "BK": 32, "WM": 2}
    t128 = {"BQ": 128, "BK": 32, "WM": 8}
    t64 = {"BQ": 64, "BK": 32, "WM": 4}
    cells = []
    for dtype in ("float16", "bfloat16"):
        for B, H in ((2, 8), (1, 12), (1, 4)):
            for N in (2048, 3072, 4096, 5120, 6144, 8192):
                cells.append(_solo(f"B2_32-32-2_D128_{dtype}_B{B}H{H}_N{N}", "2-thresholds",
                                   B, H, N, 128, dtype, t3232, entry="32-32-2_D128"))
    for dtype, Ns in (("float16", (6144, 8192, 12288, 16384)), ("bfloat16", (12288, 16384, 24576))):
        for N in Ns:
            cells.append(_solo(f"B2_64-32-4_D64_{dtype}_B1H12_N{N}", "2-thresholds",
                               1, 12, N, 64, dtype, t64, entry="64-32-4_D64"))
    for N in (16384, 24576, 32768):          # heaviest last (N32768 ~20 s/proc)
        for B, H in ((1, 12), (2, 8)):
            cells.append(_solo(f"B2_128-32-8_D128_float16_B{B}H{H}_N{N}", "2-thresholds",
                               B, H, N, 128, "float16", t128, entry="128-32-8_D128"))
    return cells


def block31():
    done = set()
    for f in AR.glob("*/results.jsonl"):
        for ln in f.read_text().splitlines():
            if ln.strip():
                done.add(json.loads(ln).get("cell_id"))
    cells = []
    for ln in (AR / "night1" / "night1_queue.jsonl").read_text().splitlines():
        if ln.strip():
            c = json.loads(ln)
            if c.get("cell_id") not in done:
                c["block"] = "3.1-fodder"
                cells.append(c)
    return cells


def block32():
    specs = {}
    for f in ("shortlist.json", "rerun_shortlist.json", "smallbh_shortlist.json"):
        for s in json.loads((S2 / f).read_text()):
            specs.setdefault(s["cand_id"], s)
    cells = []
    for r in json.loads((S2 / "blockA_master_verdicts.json").read_text()):
        if r["verdict"] != "INDECIDABLE":
            continue
        s = specs[r["cand_id"]]
        sh = s["shape"]
        cells.append(_solo(f"B32_{r['cand_id']}", "3.2-indecidables", sh["B"], sh["H"], sh["N"],
                           sh["D"], sh.get("dtype", "float16"), s["tiles"], s.get("knobs") or {},
                           entry=r["cand_id"]))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--block1-tiers", default="1,2,3,4")
    args = ap.parse_args()
    tiers = {int(x) for x in args.block1_tiers.split(",") if x}
    cells = head_floors() + block1(tiers) + block2() + block31() + block32()
    ids = [c["cell_id"] for c in cells]
    assert len(ids) == len(set(ids)), "duplicate cell_id"
    Path(args.out).write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cells))
    from collections import Counter
    print(json.dumps(Counter(c["block"] for c in cells), indent=1), "total", len(cells))


if __name__ == "__main__":
    main()
