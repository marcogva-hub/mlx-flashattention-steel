"""Auto-zoom: when the main queue is dry, refine grids around the best regions.

Rule (simple, deterministic): take the top-K screening cells per (kernel, shape) by
median_ms, then for each, emit its tile neighbours (adjacent valid BQ/BK/WM) that
haven't been measured yet. One step at a time per axis → a local neighbourhood, not a
blow-up. Zoom only consumes cells that pass the valid-tile constraint (so the runner
never wastes a slot on a cell that will raise).
"""
from __future__ import annotations

from typing import Optional

# valid tile domains (read at source: mfa_v6_nax_primitive.cpp — BK %32==0, BQ %(WM*16)==0)
_BQ = [32, 48, 64, 80, 96, 112, 128, 160, 192, 256]
_BK = [32, 64, 96, 128]
_WM = [2, 4, 8]


def _valid(bq: int, bk: int, wm: int) -> bool:
    return bk % 32 == 0 and bq % (wm * 16) == 0 and bq > 0 and wm in _WM


def _neighbours(bq: int, bk: int, wm: int):
    def adj(seq, v):
        if v not in seq:
            return []
        i = seq.index(v)
        return [seq[j] for j in (i - 1, i + 1) if 0 <= j < len(seq)]
    out = set()
    for nbq in adj(_BQ, bq):
        if _valid(nbq, bk, wm):
            out.add((nbq, bk, wm))
    for nbk in adj(_BK, bk):
        if _valid(bq, nbk, wm):
            out.add((bq, nbk, wm))
    for nwm in adj(_WM, wm):
        if _valid(bq, bk, nwm):
            out.add((bq, bk, nwm))
    return out


def _shape_sig(shape: dict) -> str:
    return f"{shape.get('D')}_{shape.get('N')}_{shape.get('dtype','float16')}_c{int(bool(shape.get('causal',False)))}"


def _cell_id(kernel: str, shape: dict, bq: int, bk: int, wm: int) -> str:
    return f"{kernel}_{_shape_sig(shape)}_bq{bq}_bk{bk}_wm{wm}_zoom"


def zoom_cells(results: list[dict], floors: dict, top_k: int, done_ids: set) -> list[dict]:
    # keep only clean dense screening cells with a time and a tile config
    good = [r for r in results
            if r.get("kernel") == "dense" and r.get("engaged") and r.get("correct")
            and r.get("median_ms") is not None and r.get("tiles")]
    if not good:
        return []
    groups: dict[str, list[dict]] = {}
    for r in good:
        groups.setdefault(_shape_sig(r["shape"]), []).append(r)
    new: dict[str, dict] = {}
    for sig, rows in groups.items():
        rows.sort(key=lambda r: r["median_ms"])
        for r in rows[:top_k]:
            t = r["tiles"]
            bq, bk, wm = t.get("BQ"), t.get("BK"), t.get("WM")
            if None in (bq, bk, wm):
                continue
            for (nbq, nbk, nwm) in _neighbours(int(bq), int(bk), int(wm)):
                cid = _cell_id("dense", r["shape"], nbq, nbk, nwm)
                if cid in done_ids or cid in new:
                    continue
                new[cid] = {"cell_id": cid, "kernel": "dense", "mode": "screening",
                            "shape": r["shape"], "tiles": {"BQ": nbq, "BK": nbk, "WM": nwm}}
    return list(new.values())
