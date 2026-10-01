"""DAY-3 verdicts, surprise zoom and block report (2026-09-30) — PRE-REGISTERED before the run.

Verdicts are computed from the raw per-process data stored in results.jsonl, never at
measurement time, so any rule below can be re-applied later without re-running.

Block 1 (B·H16 map) — the July gate-remap rule, unchanged:
  floor F_s = max |A/A - 1| over the 6 A-vs-A pairs of FLOOR_sparse_AA (same arm, forced NAX).
  WIN  : sdpa/nax ratio > 1 + F_s in BOTH orders ; LOSS : < 1 - F_s in BOTH ; else NOISE.

Block 2 / 3.2 (solo protocol, dense tiles) — margin m = (med_default - med_cand) / med_default,
medians over the 5 procs per arm. Floors from the class A-vs-A cell FLOOR_dense_AA_<D,B,H>
(same protocol, both arms default). PRIMARY floor (pre-registered):
  F_perm = 95th percentile of |m| over all 252 splits of the 10 A-vs-A samples into 5 + 5
           (permutation null of the SAME statistic, today's data, same protocol).
Sensitivity (reported, not used for placement): F_max = max over procs of |A_p/A'_p - 1|
(July-style conservative), F_3 = 0.03 (August Stage-2 fixed margin).
  WIN : m > F ; LOSS : m < -F ; NOISE otherwise ; KILLED if cos_fp32_min < 0.999.
Placement (brief): band = [N_min, N_max] of the contiguous WIN run; an ambiguous (NOISE) N
belongs to the default.
"""
from __future__ import annotations

import itertools
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

F_FIXED = 0.03
# August Stage-2 CONFIRMED anchor N per (entry, dtype) — the band must contain it.
ANCHORS = {("32-32-2_D128", "float16"): 4096, ("32-32-2_D128", "bfloat16"): 4096,
           ("128-32-8_D128", "float16"): 32768,
           ("64-32-4_D64", "float16"): 8192, ("64-32-4_D64", "bfloat16"): 16384}
DENSE_CLASSES = ((128, 2, 8), (128, 1, 12), (128, 1, 4), (64, 1, 12), (64, 2, 8))


def _load(run_dir: Path):
    p = Path(run_dir) / "results.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ----------------------------------------------------------------------------- floors
def sparse_floor(results, prefix="FLOOR_"):
    for r in results:
        if r.get("kernel") == "gate_null" and r.get("floor") is not None \
                and r["cell_id"].startswith(prefix):
            return r["floor"]
    return None


def _perm_floor(a, b, q=0.95):
    pool = list(a) + list(b)
    k = len(a)
    vals = []
    for idx in itertools.combinations(range(len(pool)), k):
        s = set(idx)
        x = [pool[i] for i in range(len(pool)) if i in s]
        y = [pool[i] for i in range(len(pool)) if i not in s]
        mx_, my_ = statistics.median(x), statistics.median(y)
        vals.append(abs((mx_ - my_) / mx_))
    vals.sort()
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def dense_floors(results, prefix="FLOOR_"):
    out = {}
    for r in results:
        if r.get("kernel") != "solo_null" or r.get("error") or not r["cell_id"].startswith(prefix):
            continue
        sh = r["shape"]
        key = (sh["D"], sh["B"], sh["H"])
        a, b = r["default_ms"], r["cand_ms"]
        out[key] = {"cell_id": r["cell_id"], "N": sh["N"], "F_perm": _perm_floor(a, b),
                    "F_max": max(abs(x / y - 1) for x, y in zip(a, b)), "F_3": F_FIXED,
                    "aa_margin": r.get("margin")}
    return out


def _class_floor(floors, D, B, H, rule="F_perm"):
    f = floors.get((D, B, H))
    if f is None:                               # nearest measured class, same D, conservative
        cands = [v for (d, b, h), v in floors.items() if d == D]
        if not cands:
            return None, None
        f = max(cands, key=lambda v: v[rule])
        return f[rule], f"{f['cell_id']} (nearest, max)"
    return f[rule], f["cell_id"]


# ----------------------------------------------------------------------------- verdicts
def gate_verdict(r, F):
    if r.get("error") or F is None:
        return "ERROR" if r.get("error") else "NO-FLOOR"
    ratios = [o["ratio"] for o in r["orders"]]
    if all(x > 1 + F for x in ratios):
        return "WIN"
    if all(x < 1 - F for x in ratios):
        return "LOSS"
    return "NOISE"


def solo_verdict(r, F):
    if r.get("error"):
        return "ERROR"
    if F is None:
        return "NO-FLOOR"
    if r.get("cos_fp32_min", 1.0) < 0.999:
        return "KILLED"
    m = r["margin"]
    return "WIN" if m > F else ("LOSS" if m < -F else "NOISE")


# ----------------------------------------------------------------------------- zoom (3.3)
def _gate_key(c):
    return (c["mask_kind"], c["dtype"], c["causal"], c["N"], c["D"], c["B"], c["H"])


def day3_zoom(results):
    """Surprise = a verdict flip between adjacent grid points of Blocks 1-2 (incl. earlier
    zoom rounds). Insert the midpoint until the gap is below the grid resolution."""
    F_s = sparse_floor(results)
    floors = dense_floors(results)
    have = {r["cell_id"] for r in results}
    new = []
    groups = defaultdict(list)
    for r in results:
        if r.get("kernel") == "gate_remap" and not r.get("error") and F_s is not None:
            c = r["cfg"]
            x = c.get("density") if c["mask_kind"] == "random" else c.get("window")
            groups[_gate_key(c)].append((x, gate_verdict(r, F_s), r))
    for key, pts in groups.items():
        pts.sort(key=lambda t: t[0])
        for (x0, v0, r0), (x1, v1, _) in zip(pts, pts[1:]):
            if v0 == v1:
                continue
            c = dict(r0["cfg"])
            if c["mask_kind"] == "random":
                if x1 - x0 <= 0.025 + 1e-9:
                    continue
                mid = round(round((x0 + x1) / 2 / 0.0125) * 0.0125, 4)
                c["density"] = mid; tag = f"rnd{mid:.4f}"
            else:
                if x1 - x0 <= 64:
                    continue
                mid = int(round((x0 + x1) / 2 / 32) * 32); c["window"] = mid; tag = f"sw{mid}"
            cid = f"B1_{tag}_{'c' if c['causal'] else 'nc'}_{c['dtype']}_b{c['B']}_h{c['H']}_n{c['N']}_d{c['D']}"
            if cid not in have:
                new.append({"cell_id": cid, "kernel": "gate_remap", "block": "3.3-zoom", "cfg": c})
                have.add(cid)
    entries = defaultdict(list)
    for r in results:
        if r.get("kernel") == "solo" and (r.get("block") in ("2-thresholds", "3.3-zoom")) \
                and r.get("entry") and not r.get("error"):
            sh = r["shape"]
            F, _ = _class_floor(floors, sh["D"], sh["B"], sh["H"])
            entries[(r["entry"], sh["dtype"], sh["B"], sh["H"])].append((sh["N"], solo_verdict(r, F), r))
    for key, pts in entries.items():
        pts.sort(key=lambda t: t[0])
        for (n0, v0, r0), (n1, v1, _) in zip(pts, pts[1:]):
            if v0 == v1 or n1 - n0 <= 512:           # DAY-3: edge resolution 512 (the N5120 dip)
                continue
            mid = int(round((n0 + n1) / 2 / 512) * 512)
            sh = dict(r0["shape"]); sh["N"] = mid
            entry, dtype, B, H = key
            cid = f"B2_{entry.replace('_D', '_D')}_{dtype}_B{B}H{H}_N{mid}"
            if cid not in have:
                new.append({"cell_id": cid, "kernel": "solo", "block": "3.3-zoom", "procs": 5,
                            "entry": entry, "shape": sh, "tiles": r0["tiles"], "knobs": {}})
                have.add(cid)
    return new


# ----------------------------------------------------------------------------- report
def _macmon_regime(run_dir: Path):
    rows = []
    for name in ("telemetry/macmon.jsonl", "macmon.jsonl"):
        p = Path(run_dir) / name
        if p.exists():
            for ln in p.read_text().splitlines():
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    continue
    freq = [r.get("gpu_usage", [None])[0] for r in rows if isinstance(r.get("gpu_usage"), list)]
    power = [r.get("gpu_power") for r in rows if r.get("gpu_power") is not None]
    busy = [(f, p) for f, p, r in zip(freq, power, rows)
            if isinstance(r.get("gpu_usage"), list) and len(r["gpu_usage"]) > 1 and r["gpu_usage"][1] > 0.5]
    med = lambda xs: round(statistics.median(xs), 1) if xs else None
    return {"samples": len(rows), "gpu_mhz_median_busy": med([f for f, _ in busy]),
            "gpu_w_median_busy": med([p for _, p in busy]), "busy_samples": len(busy)}


def write_report(run_dir) -> Path:
    run_dir = Path(run_dir)
    res = _load(run_dir)
    F_s = sparse_floor(res)
    floors = dense_floors(res)
    regime = _macmon_regime(run_dir)
    by_block = defaultdict(list)
    for r in res:
        by_block[r.get("block") or r.get("source")].append(r)
    L = [f"# DAY-3 report — {run_dir.name}  (generated {time.strftime('%Y-%m-%d %H:%M:%S')})", "",
         f"GPU regime (macmon, busy samples): {regime}", ""]
    # floors
    L += ["## Head floors (today)", f"- sparse A-vs-A B·H16 floor F_s = {F_s}"]
    for (D, B, H), f in sorted(floors.items()):
        L.append(f"- dense D{D} B{B}H{H} (N{f['N']}): F_perm={f['F_perm']:.4f} (primary) · "
                 f"F_max={f['F_max']:.4f} · F_3={f['F_3']} · A-vs-A margin {f['aa_margin']:+.4f}")
    tail_d, tail_s = dense_floors(res, "TAILFLOOR_"), sparse_floor(res, "TAILFLOOR_")
    if tail_d or tail_s is not None:
        L.append(f"- TAIL floors (same cells, end of run, warm GPU) — stability check, not used for verdicts: "
                 f"sparse F_s={tail_s}")
        for (D, B, H), f in sorted(tail_d.items()):
            h = floors.get((D, B, H), {})
            L.append(f"  - D{D} B{B}H{H}: tail F_perm={f['F_perm']:.4f} (head {h.get('F_perm', float('nan')):.4f}) · "
                     f"tail F_max={f['F_max']:.4f} · tail A-vs-A {f['aa_margin']:+.4f}")
    df = run_dir / "day_floors.json"
    if df.exists():
        L.append(f"- timing floors (usual generator): {df.read_text().strip()}")
    # block 1 map
    b1 = [r for r in res if r.get("kernel") == "gate_remap"]
    rows = []
    for r in b1:
        v = gate_verdict(r, F_s)
        rows.append({"id": r["cell_id"], "requested": r.get("cfg"), "orders": r.get("orders"),
                     "ratio_median": r.get("ratio_median"), "verdict": v, "error": r.get("error"),
                     "evidence": f"arms/{r['cell_id']}/", "block": r.get("block")})
    (run_dir / "sparse_gate_remap_map_bh16.json").write_text(json.dumps({
        "schema": "mlx-mfa.sparse-gate-remap.map.v1", "phase": "map",
        "variant": "DAY-3 B·H16 (B2H8), NAX forced via MFA_SPARSE_NAX_EXTENDED=1, per-row gates",
        "ratio_direction": "sdpa_median_ms / public_v6nax_sparse_median_ms (>1 means sparse wins)",
        "decision_floor": F_s,
        "method": {"process": "fresh process per arm/order", "sessions_per_process": 5,
                   "dispatches_per_sample": 20, "orders": 2},
        "results": rows}, indent=1) + "\n")
    L += ["", f"## Block 1 — B·H16 map ({len(b1)} cells measured)"]
    region = defaultdict(list)
    for row in rows:
        c = row["requested"] or {}
        key = (c.get("mask_kind"), c.get("dtype"), "c" if c.get("causal") else "nc", c.get("D"), c.get("N"))
        x = c.get("density") if c.get("mask_kind") == "random" else c.get("window")
        region[key].append((x, row["verdict"], row["ratio_median"]))
    L.append("| archetype | dtype | causal | D | N | points (x: verdict ratio) | extensible ceiling |")
    L.append("|---|---|---|---|---|---|---|")
    for key in sorted(region, key=lambda k: tuple(str(x) for x in k)):
        pts = sorted(region[key], key=lambda t: t[0])
        ceil = None
        for x, v, _ in pts:                      # contiguous WIN run from the sparsest point
            if v != "WIN":
                break
            ceil = x
        txt = " · ".join(f"{x}: {v} {(rm or 0):.2f}" for x, v, rm in pts)
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {txt} | {ceil if ceil is not None else '—'} |")
    # block 2
    b2 = [r for r in res if r.get("kernel") == "solo" and r.get("block") in ("2-thresholds", "3.3-zoom")
          and r.get("entry") and not str(r.get("entry")).startswith("D1_") and not str(r.get("entry")).startswith("N2_")]
    L += ["", f"## Block 2 — dispatch-table bands ({len(b2)} cells measured)"]
    ent = defaultdict(list)
    for r in b2:
        sh = r["shape"]
        ent[(r["entry"], sh["dtype"], sh["B"], sh["H"])].append(r)
    for key in sorted(ent):
        pts = sorted(ent[key], key=lambda r: r["shape"]["N"])
        D, B, H = pts[0]["shape"]["D"], key[2], key[3]
        F, src = _class_floor(floors, D, B, H)
        Fm, _ = _class_floor(floors, D, B, H, "F_max")
        seq = [(r["shape"]["N"], solo_verdict(r, F), r["margin"] if not r.get("error") else None,
                solo_verdict(r, Fm), solo_verdict(r, F_FIXED)) for r in pts]
        runs = []
        cur = []
        for n, v, *_ in seq:
            if v == "WIN":
                cur.append(n)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        # Placement (brief): the band is the CONTIGUOUS WIN run containing the entry's August
        # anchor N (where it was CONFIRMED); an ambiguous / losing N belongs to the default,
        # so other WIN islands are reported but NOT placed.
        anchor = ANCHORS.get((key[0], key[1]))
        placed = next((r for r in runs if anchor in r), None)
        if placed:
            lo, hi = placed[0], placed[-1]
            idx = {n: i for i, (n, *_r) in enumerate(seq)}
            below = seq[idx[lo] - 1] if idx[lo] > 0 else None
            above = seq[idx[hi] + 1] if idx[hi] + 1 < len(seq) else None
            edge = (f"edges: N{lo} {seq[idx[lo]][2]:+.3f} / N{hi} {seq[idx[hi]][2]:+.3f}; outside: "
                    + (f"N{below[0]} {below[1]} {below[2]:+.3f}" if below else "none measured below")
                    + " · " + (f"N{above[0]} {above[1]} {above[2]:+.3f}" if above else "none measured above"))
            band = f"[{lo}, {hi}] ({edge})"
        else:
            band = f"none — anchor N{anchor} not WIN today (default everywhere measured)"
        islands = [r for r in runs if r is not placed]
        L.append(f"- **{key[0]} {key[1]} B{B}H{H}** floor {F if F is None else round(F, 4)} ({src}): band {band}"
                 + (f" — WIN islands NOT placed: {islands}" if islands else ""))
        L.append("  " + " · ".join(f"N{n}: {v} {m:+.3f} [F_max {vm}, F_3 {v3}]" if m is not None else f"N{n}: {v}"
                                   for n, v, m, vm, v3 in seq))
    # 3.1 / 3.2
    b31 = [r for r in res if r.get("block") == "3.1-fodder"]
    L += ["", f"## Block 3.1 — night-1 fodder remainder: {len(b31)}/123 screened "
              f"(errors {sum(1 for r in b31 if r.get('error'))})"]
    best = defaultdict(list)
    for r in b31:
        if r.get("median_ms") and not r.get("error"):
            s = r.get("shape") or {}
            best[(s.get("D"), s.get("dtype"), s.get("B"), s.get("H"), s.get("N"))].append(
                (r["median_ms"], r["cell_id"], r.get("sdpa_ms")))
    for k in sorted(best, key=lambda t: tuple(str(x) for x in t)):
        ms, cid, sd = sorted(best[k])[0]
        L.append(f"- {k}: fastest {cid} {ms:.3f} ms (sdpa {sd}) of {len(best[k])} tiles — screening only")
    b32 = [r for r in res if r.get("block") == "3.2-indecidables"]
    L += ["", f"## Block 3.2 — Stage-2 INDECIDABLE re-screen: {len(b32)}/25"]
    for r in b32:
        sh = r.get("shape") or {}
        F, src = _class_floor(floors, sh.get("D"), sh.get("B"), sh.get("H"))
        v = solo_verdict(r, F)
        m = r.get("margin")
        L.append(f"- {r['entry']}: {v} margin {m:+.3f} (floor {F if F is None else round(F, 4)})"
                 if m is not None else f"- {r['entry']}: {v} {r.get('error', '')[:120]}")
    z = [r for r in res if r.get("block") == "3.3-zoom"]
    L += ["", f"## Block 3.3 — surprise zoom: {len(z)} cells"]
    # health
    L += ["", "## Run health"]
    rs = run_dir / "run_summary.json"
    if rs.exists():
        s = json.loads(rs.read_text())
        L.append(f"- stop: {s.get('stop_reason')} · cells {s.get('cells_done')} · elapsed {s.get('elapsed_s')} s · "
                 f"sentinel episodes {len(s.get('episodes', []))} · watchdog pauses {s.get('pauses')}")
        for ep in s.get("episodes", []):
            L.append(f"  - sentinel @cell {ep['at_cell']}: {ep['sentinel_ms']} ms vs {ep['baseline_ms']} "
                     f"({ep['drift_frac']:+.1%}) {ep['verdict']}")
    pp = run_dir / "pauses.jsonl"
    if pp.exists():
        ps = [json.loads(l) for l in pp.read_text().splitlines() if l.strip()]
        op = [x for x in ps if x.get("kind") != "foreign-job"]
        fg = [x for x in ps if x.get("kind") == "foreign-job"]
        L.append(f"- operator pauses: {len(op)} · total {sum(x['secs'] for x in op):.0f} s · "
                 f"foreign-job pauses: {len(fg)} · total {sum(x['secs'] for x in fg):.0f} s")
        for x in ps:          # a paused cell straddles two regimes: named, so it can be discounted
            L.append(f"  - {x.get('kind', 'operator')} pause {x['secs']} s inside {x['cell']} ({x['block']})")
    part = run_dir / "partial.jsonl"
    if part.exists():
        L.append(f"- interrupted cells (partial.jsonl): {len(part.read_text().splitlines())}")
    errs = [r["cell_id"] for r in res if r.get("error")]
    L.append(f"- cells with errors: {len(errs)} {errs[:10]}")
    out = run_dir / "day3_report.md"
    out.write_text("\n".join(L) + "\n")
    return out
