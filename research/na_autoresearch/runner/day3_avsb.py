"""DAY-3 A-vs-B report (2026-09-30): one document, blocks side by side.

Run A = day3-20260930 (contaminated by a concurrent CPU workload — README_CONTAMINATION.md),
run B = day3b-<date> (idle window, strict host preflight). Each run is judged against ITS OWN
floors (pre-registered rules of day3_report.py). Survival logic for the table session:
  band in BOTH runs  -> candidate static default, placed on the INTERSECTION of the two bands;
  band in ONE run    -> regime-sensitive: on-device calibration prior, never a default;
  no band            -> default everywhere measured.

  python -m research.na_autoresearch.runner.day3_avsb <runA> <runB> <out.md>
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from .day3_regime import block_regimes
from .day3_report import (ANCHORS, _class_floor, _load, dense_floors, gate_verdict,
                          solo_verdict, sparse_floor)

BLOCKS = ("0-floors", "1-bh16", "2-thresholds", "3.1-fodder", "3.2-indecidables", "3.3-zoom",
          "9-tail-floors")


def _fmt_stats(s):
    return "—" if not s else f"{s['mean']:.0f} ± {s['sd']:.0f} (n={s['n']})"


def _bands(res, floors):
    """entry key -> dict(N -> (verdict, margin)), band (lo, hi) containing the anchor or None."""
    ent = defaultdict(dict)
    for r in res:
        if r.get("kernel") == "solo" and r.get("block") in ("2-thresholds", "3.3-zoom") \
                and r.get("entry") and not r.get("error"):
            sh = r["shape"]
            F, _ = _class_floor(floors, sh["D"], sh["B"], sh["H"])
            ent[(r["entry"], sh["dtype"], sh["B"], sh["H"])][sh["N"]] = (solo_verdict(r, F), r["margin"])
    out = {}
    for key, pts in ent.items():
        ns = sorted(pts)
        runs, cur = [], []
        for n in ns:
            if pts[n][0] == "WIN":
                cur.append(n)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        anchor = ANCHORS.get((key[0], key[1]))
        placed = next((r for r in runs if anchor in r), None)
        out[key] = {"pts": pts, "band": (placed[0], placed[-1]) if placed else None,
                    "islands": [r for r in runs if r is not placed]}
    return out


def write(run_a: Path, run_b: Path, out: Path) -> Path:
    A, B = _load(run_a), _load(run_b)
    fa, fb = dense_floors(A), dense_floors(B)
    sa, sb = sparse_floor(A), sparse_floor(B)
    ra, rb = block_regimes(run_a, A), block_regimes(run_b, B)
    L = [f"# DAY-3 — rapport A vs B ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    lec = out.with_name(out.stem + "_lecture.md")
    if lec.exists():
        L += [lec.read_text().strip(), ""]
    reg0 = run_b / "regime_start.json"
    L += ["## Régime GPU du run B (en tête)",
          f"- départ (préflight hôte) : `{reg0.read_text().strip() if reg0.exists() else 'n/a'}`",
          "- par bloc (échantillons macmon GPU occupés ≥ 0.5 ; CPU sur tous les échantillons) :", "",
          "| bloc | A : GPU MHz moy ± sd | B : GPU MHz moy ± sd | A : CPU W moy (max) | B : CPU W moy (max) | A : GPU °C | B : GPU °C |",
          "|---|---|---|---|---|---|---|"]
    for b in BLOCKS:
        xa, xb = ra.get(b), rb.get(b)
        if not xa and not xb:
            continue
        c = lambda x: "—" if not x or not x["cpu_w_all"] else f"{x['cpu_w_all']['mean']} ({x['cpu_w_all']['max']})"
        g = lambda x: "—" if not x or not x["gpu_c_all"] else f"{x['gpu_c_all']['mean']}"
        L.append(f"| {b} | {_fmt_stats(xa and xa['gpu_mhz_busy'])} | {_fmt_stats(xb and xb['gpu_mhz_busy'])} | "
                 f"{c(xa)} | {c(xb)} | {g(xa)} | {g(xb)} |")
    dn = Path("/Users/marcomarcelino/code/mlx-mfa-v2/devnotes/day3_20260930.md")
    txt = dn.read_text() if dn.exists() else ""
    s0 = txt.find("## Protocole SOLO")
    contract = txt[s0:txt.find("\n## ", s0 + 5)].strip() if s0 >= 0 else "(contrat introuvable)"
    L += ["", "## Protocole solo — contrat écrit (cité par les confirmés `single_arm`)", "",
          contract.replace("## Protocole SOLO (bras-par-processus) — contrat écrit (cité par les confirmés `single_arm`)", "").strip(), ""]
    # run health side by side
    L += ["## Santé des deux runs", "", "| | A | B |", "|---|---|---|"]
    def health(run, R):
        s = json.loads((run / "run_summary.json").read_text()) if (run / "run_summary.json").exists() else {}
        eps = s.get("episodes", [])
        pp = run / "pauses.jsonl"
        ps = [json.loads(l) for l in pp.read_text().splitlines() if l.strip()] if pp.exists() else []
        cpu = [r["regime"]["cpu_w"] for r in R if r.get("regime") and r["regime"].get("cpu_w") is not None]
        return {"stop": s.get("stop_reason"), "cells": len(R), "errors": sum(1 for r in R if r.get("error")),
                "sent": f"{len(eps)} checks, max drift {max((e['drift_frac'] for e in eps), default=0):+.1%}",
                "pauses": f"{len(ps)} ({sum(x['secs'] for x in ps):.0f} s)",
                "cpu": (f"median {sorted(cpu)[len(cpu) // 2]} W, max {max(cpu)} W" if cpu else "non enregistré par cellule (run A) — voir télémétrie par bloc")}
    ha, hb = health(run_a, A), health(run_b, B)
    for k, lab in (("stop", "arrêt"), ("cells", "cellules"), ("errors", "erreurs"), ("sent", "sentinelle dense"),
                   ("pauses", "pauses (opérateur + garde)"), ("cpu", "CPU par cellule")):
        L.append(f"| {lab} | {ha[k]} | {hb[k]} |")
    L.append("")
    # floors
    L += ["## Planchers A vs B (delta = effet de la contention sur le bruit)", "",
          "| plancher | A | B | delta B−A |", "|---|---|---|---|",
          f"| sparse B·H16 F_s | {sa and round(sa, 4)} | {sb and round(sb, 4)} | {None if sa is None or sb is None else round(sb - sa, 4)} |"]
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key, {}).get("F_perm"), fb.get(key, {}).get("F_perm")
        d = None if va is None or vb is None else round(vb - va, 4)
        L.append(f"| dense D{key[0]} B{key[1]}H{key[2]} F_perm | {va and round(va, 4)} | {vb and round(vb, 4)} | {d} |")
    tb = dense_floors(B, "TAILFLOOR_")
    if tb:
        L.append("")
        L.append("Planchers de queue du run B (stabilité tête → queue) : " + " · ".join(
            f"D{k[0]} B{k[1]}H{k[2]} {v['F_perm']:.4f}" for k, v in sorted(tb.items()))
                 + f" · sparse {round(sparse_floor(B, 'TAILFLOOR_') or 0, 4)}"
                 + " — les planchers varient jusqu'à ×2 entre tête et queue d'un même run : une marge "
                   "inférieure à ~2× le plancher est régime-sensible.")
    # block 1
    ia = {r["cell_id"]: r for r in A if r.get("kernel") == "gate_remap"}
    ib = {r["cell_id"]: r for r in B if r.get("kernel") == "gate_remap"}
    rows, flips = [], 0
    for cid in sorted(set(ia) | set(ib)):
        va = gate_verdict(ia[cid], sa) if cid in ia else "—"
        vb = gate_verdict(ib[cid], sb) if cid in ib else "—"
        fl = va != vb and "—" not in (va, vb)
        flips += fl
        rows.append(f"| {cid} | {va} {ia[cid]['ratio_median']:.2f}× | {vb} {ib[cid]['ratio_median']:.2f}× | {'**FLIP**' if fl else ''} |"
                    if cid in ia and cid in ib else f"| {cid} | {va} | {vb} | |")
    L += ["", f"## Bloc 1 — carte B·H16 (contrôle : attendu identique) — {len(ib)} cellules B, {flips} flips", "",
          "| cellule | A | B | |", "|---|---|---|---|"] + rows
    # block 2
    ba, bb = _bands(A, fa), _bands(B, fb)
    L += ["", "## Bloc 2 — bandes de la table (la question)", ""]
    surv, only = [], []
    for key in sorted(set(ba) | set(bb)):
        xa, xb = ba.get(key), bb.get(key)
        ns = sorted(set(xa["pts"] if xa else {}) | set(xb["pts"] if xb else {}))
        cells = []
        for n in ns:
            pa = xa["pts"].get(n) if xa else None
            pb = xb["pts"].get(n) if xb else None
            fl = pa and pb and pa[0] != pb[0]
            cells.append(f"N{n}: A {pa[0] + f' {pa[1]:+.3f}' if pa else '—'} / B {pb[0] + f' {pb[1]:+.3f}' if pb else '—'}"
                         + (" **FLIP**" if fl else ""))
        bda, bdb = xa and xa["band"], xb and xb["band"]
        L.append(f"- **{key[0]} {key[1]} B{key[2]}H{key[3]}** — bande A {list(bda) if bda else 'aucune'} · "
                 f"bande B {list(bdb) if bdb else 'aucune'}")
        L.append("  " + " · ".join(cells))
        if bda and bdb and max(bda[0], bdb[0]) <= min(bda[1], bdb[1]):
            surv.append((key, (max(bda[0], bdb[0]), min(bda[1], bdb[1]))))
        elif bda or bdb:
            only.append((key, "A" if bda else "B", bda or bdb))
    # 3.1 magnitude control
    fa31 = {r["cell_id"]: r for r in A if r.get("block") == "3.1-fodder" and r.get("ratio_vs_sdpa")}
    fb31 = {r["cell_id"]: r for r in B if r.get("block") == "3.1-fodder" and r.get("ratio_vs_sdpa")}
    common = sorted(set(fa31) & set(fb31))
    if common:
        d = [fb31[c]["ratio_vs_sdpa"] / fa31[c]["ratio_vs_sdpa"] - 1 for c in common]
        best = lambda X: max(x["ratio_vs_sdpa"] for x in X.values())
        L += ["", "## Bloc 3.1 — fodder (contrôle de magnitude : attendu ~0.40× ± qq %)",
              f"- cellules A {len(fa31)} · B {len(fb31)} · communes {len(common)}",
              f"- meilleur ratio NAX/SDPA : A {best(fa31):.3f}× · B {best(fb31):.3f}×",
              f"- écart relatif B/A par cellule : médiane {statistics.median(d):+.1%}, "
              f"min {min(d):+.1%}, max {max(d):+.1%}"]
    # 3.2
    ga = {r["entry"]: r for r in A if r.get("block") == "3.2-indecidables"}
    gb = {r["entry"]: r for r in B if r.get("block") == "3.2-indecidables"}
    L += ["", "## Bloc 3.2 — INDÉCIDABLES du Stage-2 re-jugés", "",
          "| candidat | A | B | |", "|---|---|---|---|"]
    for e in sorted(set(ga) | set(gb)):
        def v(X, F):
            r = X.get(e)
            if not r or r.get("error"):
                return "—"
            sh = r["shape"]; f, _ = _class_floor(F, sh["D"], sh["B"], sh["H"])
            return f"{solo_verdict(r, f)} {r['margin']:+.3f}"
        a_, b_ = v(ga, fa), v(gb, fb)
        L.append(f"| {e} | {a_} | {b_} | {'**FLIP**' if a_ != '—' and b_ != '—' and a_.split()[0] != b_.split()[0] else ''} |")
    # zoom
    zb = [r for r in B if r.get("block") == "3.3-zoom"]
    L += ["", f"## Bloc 3.3 — zoom : {len(zb)} cellules en B (A : zoom jamais exécuté) — intégrées aux bandes ci-dessus"]
    # synthesis
    L += ["", "## Synthèse pour la session table", "",
          "**Bandes survivant aux deux runs → candidates au défaut statique (intersection A ∩ B) :**"]
    L += [f"- {k[0]} {k[1]} B{k[2]}H{k[3]} : [{b[0]}, {b[1]}]" for k, b in surv] or ["- aucune"]
    L += ["", "**Bandes présentes dans un seul run → régime-sensibles : priors de calibration on-device, jamais défauts :**"]
    L += [f"- {k[0]} {k[1]} B{k[2]}H{k[3]} : run {w} seul, {list(b)}" for k, w, b in only] or ["- aucune"]
    isl = []
    for key in sorted(set(ba) | set(bb)):
        for w, X in (("A", ba.get(key)), ("B", bb.get(key))):
            for r in (X or {}).get("islands", []):
                isl.append(f"- {key[0]} {key[1]} B{key[2]}H{key[3]} : run {w}, WIN sur {r} (hors de l'ancre août)")
    L += ["", "**Îlots de WIN hors ancre (information, jamais placés — le brief place la bande qui contient "
          "l'ancre CONFIRMÉE en août) :**"] + (isl or ["- aucun"])
    L += ["", "_Résolution : le run A n'a pas exécuté de zoom (arrêté avant) ; ses bords sont à la grille "
          "1024 → l'intersection A ∩ B est bornée par A, donc conservatrice._"]
    L += ["", f"**Carte B·H16 (Bloc 1) → extension d'allowlist sur pièces** : voir la carte B "
          f"`{run_b.name}/sparse_gate_remap_map_bh16.json` ({len(ib)} cellules, {flips} flips A→B)."]
    out.write_text("\n".join(L) + "\n")
    return out


if __name__ == "__main__":
    p = write(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(p)
