"""Stage-2 verdict aggregator (measure-only). Reads a blockA_results.jsonl of per-(cand,order,
proc) A/B measurements → per-candidate verdict vs the block-head floor noise.

Verdict rule (both orders must agree — order-bias is the enemy):
  CONFIRMED   : candidate median faster than default by > MARGIN in BOTH orders,
                AND cand_cos_fp32 >= 0.999 (correct), AND engaged (fp>0),
                AND (no knobs OR knob_changed) — the RELAXED no-op guard.
  KILLED      : not faster in an order / slower / incorrect / disengaged / knob no-op.
  INDECIDABLE : orders disagree, or |margin| within noise in one order.
Outliers are DECLARED (per-arm min/median/max/spread across procs), never silently dropped.

  s2_verdict.py <blockA_results.jsonl> [margin=0.03]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def _stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {"n": len(xs), "min": round(xs[0], 4), "median": round(xs[len(xs) // 2], 4),
            "max": round(xs[-1], 4), "spread_pct": round((xs[-1] - xs[0]) / xs[len(xs) // 2] * 100, 1) if xs[len(xs)//2] else None}


def aggregate(path, margin=0.03):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = defaultdict(lambda: defaultdict(lambda: {"d": [], "c": []}))
    meta = {}
    for r in rows:
        if r.get("error") or r.get("cand_ms") is None:
            continue
        cid = r["cand_id"]; o = r["order"]
        by[cid][o]["d"].append(r["default_ms"]); by[cid][o]["c"].append(r["cand_ms"])
        meta.setdefault(cid, r)
    out = []
    for cid, orders in by.items():
        m = meta[cid]
        rec = {"cand_id": cid, "cos_fp32": m.get("cand_cos_fp32"), "engaged": m.get("cand_fp", 0) > 0,
               "knob_changed": m.get("knob_changed"), "has_knob": bool(m.get("knobs") if isinstance(m.get("knobs"), dict) else False)}
        order_margins = {}
        for o in ("AB", "BA"):
            if o not in orders:
                continue
            ds = _stats(orders[o]["d"]); cs = _stats(orders[o]["c"])
            rec[f"default_{o}"] = ds; rec[f"cand_{o}"] = cs
            if ds and cs and ds["median"]:
                order_margins[o] = (ds["median"] - cs["median"]) / ds["median"]
        rec["margins"] = {o: round(v, 4) for o, v in order_margins.items()}
        # verdict
        if rec["cos_fp32"] is not None and rec["cos_fp32"] < 0.999:
            v = "KILLED"; why = "incorrect (cos<0.999)"
        elif not rec["engaged"]:
            v = "KILLED"; why = "disengaged (SDPA fallback)"
        elif rec["has_knob"] and not rec["knob_changed"]:
            v = "KILLED"; why = "knob no-op (RELAXED-class)"
        elif len(order_margins) < 2:
            v = "INDECIDABLE"; why = "missing an order"
        elif all(mv > margin for mv in order_margins.values()):
            v = "CONFIRMED"; why = f"faster both orders (>{margin:.0%})"
        elif all(mv < -margin for mv in order_margins.values()):
            v = "KILLED"; why = "slower both orders"
        elif min(order_margins.values()) <= margin <= max(order_margins.values()) or \
                (order_margins.get("AB", 0) > margin) != (order_margins.get("BA", 0) > margin):
            v = "INDECIDABLE"; why = "orders disagree"
        else:
            v = "INDECIDABLE"; why = "within noise"
        rec["verdict"] = v; rec["why"] = why
        out.append(rec)
    order = {"CONFIRMED": 0, "INDECIDABLE": 1, "KILLED": 2}
    out.sort(key=lambda r: (order[r["verdict"]], -(min(r["margins"].values()) if r["margins"] else -1)))
    return out


def main():
    path = sys.argv[1]
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    res = aggregate(path, margin)
    from collections import Counter
    c = Counter(r["verdict"] for r in res)
    print(f"=== Stage-2 verdicts ({len(res)} candidates, margin {margin:.0%}) : {dict(c)} ===")
    for r in res:
        mg = " ".join(f"{o}{v:+.1%}" for o, v in r["margins"].items())
        print(f"  [{r['verdict']:11}] {r['cand_id'][:46]:46} {mg}  cos={r['cos_fp32']} — {r['why']}")
    out = path.replace(".jsonl", "_verdicts.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
