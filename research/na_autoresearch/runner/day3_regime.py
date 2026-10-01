"""DAY-3 regime per block (2026-09-30): GPU frequency / power and CPU power from the run's macmon
LaunchAgent telemetry (5 s), windowed on each block's cells (start = end - cell_secs).
Same computation for run A and run B, so the A-vs-B report compares like with like.

A sample is "GPU busy" when gpu_usage[1] >= BUSY; frequency statistics use busy samples only
(idle samples sit at 338 MHz and would say nothing about the regime under load).
"""
from __future__ import annotations

import datetime as _dt
import json
import statistics
from collections import defaultdict
from pathlib import Path

BUSY = 0.5


def _ts(s: str) -> float:
    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_telemetry(run_dir: Path):
    p = Path(run_dir) / "telemetry" / "macmon.jsonl"
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            try:
                d = json.loads(ln)
                out.append({"t": _ts(d["timestamp"]), "gpu_mhz": d["gpu_usage"][0],
                            "gpu_busy": d["gpu_usage"][1], "gpu_w": d.get("gpu_power"),
                            "cpu_w": d.get("cpu_power"), "sys_w": d.get("sys_power"),
                            "gpu_c": (d.get("temp") or {}).get("gpu_temp_avg")})
            except Exception:
                continue
    return out


def block_windows(results):
    win = defaultdict(lambda: [float("inf"), 0.0])
    for r in results:
        b = r.get("block") or r.get("source")
        end = r.get("ts_epoch")
        if not end:
            continue
        start = end - float(r.get("cell_secs") or r.get("cell_wall_s") or 0.0)
        win[b][0] = min(win[b][0], start)
        win[b][1] = max(win[b][1], end)
    return dict(win)


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1),
            "sd": round(statistics.pstdev(xs), 1) if len(xs) > 1 else 0.0,
            "min": round(min(xs), 1), "max": round(max(xs), 1)}


def block_regimes(run_dir: Path, results):
    tel = load_telemetry(run_dir)
    out = {}
    for b, (t0, t1) in sorted(block_windows(results).items(), key=lambda kv: kv[1][0]):
        s = [x for x in tel if t0 <= x["t"] <= t1]
        busy = [x for x in s if x["gpu_busy"] >= BUSY]
        out[b] = {"window": [int(t0), int(t1)], "minutes": round((t1 - t0) / 60, 1),
                  "samples": len(s), "busy_samples": len(busy),
                  "gpu_mhz_busy": _stats([x["gpu_mhz"] for x in busy]),
                  "gpu_w_busy": _stats([x["gpu_w"] for x in busy]),
                  "cpu_w_all": _stats([x["cpu_w"] for x in s]),
                  "gpu_c_all": _stats([x["gpu_c"] for x in s])}
    return out
