"""DAY-3 cell kinds (2026-09-30) — fresh-process protocols driven by the August runner.

Every kind launches EXISTING measurement children, one fresh process per arm, and calls
``gate()`` before each process: PAUSE blocks there, STOP raises ``Interrupted`` (the current
process always finishes first — pause latency <= 1 process). An interrupted cell is NOT
recorded as done (its completed processes go to partial.jsonl); a relaunch re-runs it whole,
because a cell's arms must share one regime.

Kinds (measurement logic lives in the children — nothing here times anything):
  gate_remap : Block 1. July gate-remap arm (day3_gate_arm.py = bench_sparse_gate_remap.py +
               per-row gates). Orders (nax, sdpa) then (sdpa, nax); ratio per order =
               sdpa_ms / nax_ms (>1: sparse wins). NAX forced via the public contract
               (MFA_SPARSE_NAX_EXTENDED=1 on every process of the cell).
  gate_null  : Block-1 floor. A-vs-A of the same arm (nax, nax) x n_pairs; ratio = first/second.
  solo       : Block 2 / 3.2. s2_harness.py ``solo`` child (August): per process index p,
               arm 'default' then arm 'cand' (August launch order), procs per arm = cell['procs'].
  solo_null  : Block-2 floors. Same, both arms = default config (A-vs-A).
Verdicts are NOT computed here (day3_report.py), so a stored cell never depends on a floor.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

MFA_ROOT = Path(os.environ.get("DAY3_MFA_ROOT", "/Users/marcomarcelino/code/mlx-mfa-v2"))
GATE_ARM = Path(__file__).with_name("day3_gate_arm.py")
S2 = Path(__file__).with_name("s2_harness.py")


class Interrupted(Exception):
    """STOP requested between two processes; carries the partial process records."""

    def __init__(self, partial):
        super().__init__("stop requested")
        self.partial = partial


def _python() -> str:
    return sys.executable


def _run(cmd, env_extra, gate, partial, timeout):
    try:
        gate()
    except Interrupted:
        raise Interrupted(partial)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MFA_") or k == "MFA_SILENCE_NAX_WARNING"}
    env.update(env_extra)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(MFA_ROOT), timeout=timeout)
    return r, round(time.time() - t0, 2)


# --------------------------------------------------------------------------- gate-remap
def _gate_cmd(cell, arm, out_path):
    c = cell["cfg"]
    cmd = [_python(), str(GATE_ARM), "--mfa-root", str(MFA_ROOT), "--arm", arm,
           "--mask-kind", c["mask_kind"], "--B", str(c["B"]), "--H", str(c["H"]),
           "--N", str(c["N"]), "--D", str(c["D"]), "--dtype", c["dtype"],
           "--seed", str(c.get("seed", 20260713)), "--output", str(out_path)]
    if c["mask_kind"] == "sliding":
        cmd += ["--window", str(c["window"])]
    else:
        cmd += ["--density", str(c["density"])]
    if c.get("causal"):
        cmd.append("--causal")
    return cmd


def _gate_proc(cell, arm, tag, run_dir, gate, partial):
    out = Path(run_dir) / "arms" / cell["cell_id"] / f"{tag}_{arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    r, secs = _run(_gate_cmd(cell, arm, out), {"MFA_SPARSE_NAX_EXTENDED": "1",
                                               "MFA_SILENCE_NAX_WARNING": "1"},
                   gate, partial, timeout=900)
    rec = {"tag": tag, "arm": arm, "secs": secs, "rc": r.returncode}
    if r.returncode != 0 or not out.exists():
        rec["error"] = (r.stderr or r.stdout or "")[-600:]
    else:
        payload = json.loads(out.read_text())
        row = payload["row"]
        rec.update(median_ms=row["timing"]["median_ms"], p95_ms=row["timing"]["p95_ms"],
                   samples_ms=row["timing"]["samples_ms"], cell=row["cell"],
                   terminal_public=row["which_binary"]["public"]["terminal"],
                   terminal_baseline=row["which_binary"]["baseline"]["terminal"],
                   correction=row["which_binary"]["correction"],
                   row_gates=row["which_binary"].get("row_gates"),
                   commit=payload.get("commit"), mlx=payload.get("mlx"))
    partial.append(rec)
    return rec


def run_gate_remap(cell, run_dir, gate):
    partial = []
    orders = []
    for oi, seq in enumerate((("public", "sdpa"), ("sdpa", "public")), 1):
        recs = [_gate_proc(cell, arm, f"order{oi}_pos{pos}", run_dir, gate, partial)
                for pos, arm in enumerate(seq, 1)]
        if any("error" in r for r in recs):
            return {"error": "; ".join(r.get("error", "")[-300:] for r in recs if "error" in r),
                    "procs": partial}
        ms = {r["arm"]: r["median_ms"] for r in recs}
        orders.append({"order": oi, "sequence": list(seq), "nax_ms": ms["public"],
                       "sdpa_ms": ms["sdpa"], "ratio": ms["sdpa"] / ms["public"]})
    return {"orders": orders, "ratio_median": statistics.median(o["ratio"] for o in orders),
            "procs": partial, "engaged": all(p["terminal_public"][0] == "v6nax_sparse" for p in partial),
            "correct": True, "median_ms": statistics.median(o["nax_ms"] for o in orders)}


def run_gate_null(cell, run_dir, gate):
    partial = []
    pairs = []
    for i in range(cell.get("n_pairs", 6)):
        a = _gate_proc(cell, "public", f"pair{i}_pos1", run_dir, gate, partial)
        b = _gate_proc(cell, "public", f"pair{i}_pos2", run_dir, gate, partial)
        if "error" in a or "error" in b:
            return {"error": (a.get("error") or b.get("error"))[-300:], "procs": partial}
        pairs.append({"pair": i, "first_ms": a["median_ms"], "second_ms": b["median_ms"],
                      "ratio": a["median_ms"] / b["median_ms"]})
    dev = [abs(p["ratio"] - 1.0) for p in pairs]
    return {"pairs": pairs, "deviations": dev, "floor": max(dev),
            "floor_rule": "July gate-remap rule: max |A/A - 1| over all pairs",
            "procs": partial, "engaged": all(p["terminal_public"][0] == "v6nax_sparse" for p in partial),
            "correct": True, "median_ms": statistics.median(p["first_ms"] for p in pairs)}


# --------------------------------------------------------------------------- solo (dense)
def _solo_proc(spec, arm, proc, gate, partial):
    r, secs = _run([_python(), str(S2), "solo", json.dumps(spec), arm],
                   {"MFA_SILENCE_NAX_WARNING": "1"}, gate, partial, timeout=900)
    rec = {"arm": arm, "proc": proc, "secs": secs, "rc": r.returncode}
    if r.returncode != 0:
        rec["error"] = (r.stderr or "")[-600:]
    else:
        out = json.loads(r.stdout.strip().splitlines()[-1])
        rec.update(ms=out["ms"], cos_fp32=out["cos_fp32"])
    partial.append(rec)
    return rec


def run_solo(cell, run_dir, gate):
    spec = {"cand_id": cell["cell_id"], "shape": cell["shape"],
            "tiles": cell.get("tiles", {}), "knobs": cell.get("knobs", {})}
    partial = []
    for p in range(cell.get("procs", 5)):
        for arm in ("default", "cand"):            # August _parent_solo launch order
            rec = _solo_proc(spec, arm, p, gate, partial)
            if "error" in rec:
                return {"error": rec["error"][-300:], "procs": partial}
    d = [r["ms"] for r in partial if r["arm"] == "default"]
    c = [r["ms"] for r in partial if r["arm"] == "cand"]
    md, mc = statistics.median(d), statistics.median(c)
    cos = min(r["cos_fp32"] for r in partial)
    return {"default_ms": d, "cand_ms": c, "default_median": md, "cand_median": mc,
            "margin": (md - mc) / md, "cos_fp32_min": cos, "procs": partial,
            "engaged": True, "correct": cos >= 0.999, "median_ms": mc}


def run_day3_cell(cell, run_dir, gate):
    kind = cell["kernel"]
    fn = {"gate_remap": run_gate_remap, "gate_null": run_gate_null,
          "solo": run_solo, "solo_null": run_solo}[kind]
    t0 = time.time()
    res = fn(cell, run_dir, gate)
    base = {"cell_id": cell["cell_id"], "kernel": kind, "block": cell.get("block"),
            "cfg": cell.get("cfg"), "shape": cell.get("shape"), "tiles": cell.get("tiles"),
            "entry": cell.get("entry"), "cell_secs": round(time.time() - t0, 1),
            "ts_epoch": int(time.time())}
    base.update(res)
    return base


DAY3_KINDS = {"gate_remap", "gate_null", "solo", "solo_null"}
