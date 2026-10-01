"""NA-autoresearch detached night runner — the self-sufficient worker.

THE WORKER IS THIS SCRIPT, NOT THE AGENT. Launched detached under caffeinate; the
agent's session may end, this keeps going until the night is done.

Contract:
  - Consumes a declarative queue of cells (JSONL: one cell per line).
  - Executes them SERIALLY (one GPU job at a time, always) via the Stage-1 screening
    executors. Screening only — Stage-2 hardening + Stage-3 promotion never run here.
  - Appends each result to a crash-safe results JSONL (one fsync'd line per cell).
  - Idempotent resume by cell_id: on restart, cells already in results are skipped.
  - Day-floor head block first; watchdog between cells; auto-zoom on exhaustion; then
    the overflow backlog. Stops ONLY on: queue+zoom+backlog exhausted, watchdog STOP,
    deadline (07:00), or a stop-file. Every stop leaves a legible reason.

Run:  caffeinate -dims python -m research.na_autoresearch.runner.night_runner \
          --queue <night.jsonl> --out <run_dir> [--backlog <overflow.jsonl>] \
          [--deadline-epoch N] [--zoom-topk 5] [--no-zoom]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .executors import run_cell
from .executors_day3 import DAY3_KINDS, Interrupted, run_day3_cell
from .health import (Watchdog, foreign_jobs, host_regime, measure_day_floors, preflight_host,
                     ram_ok)
from .zoom import zoom_cells


def _load_jsonl(path: Path) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    out = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def _done_ids(results_path: Path) -> set[str]:
    ids = set()
    if results_path.exists():
        for ln in results_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                ids.add(json.loads(ln)["cell_id"])
            except Exception:
                continue
    return ids


def _append(results_path: Path, rec: dict) -> None:
    with results_path.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _log(run_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (run_dir / "runner.log").open("a") as f:
        f.write(line + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backlog", default=None,
                    help="comma-separated fall-through queues, processed in order after queue+zoom "
                         "(permanent rule: dry source → next source, never idle)")
    ap.add_argument("--deadline-epoch", type=float, default=None)
    ap.add_argument("--zoom-topk", type=int, default=5)
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--max-pauses", type=int, default=6)
    ap.add_argument("--pause-backoff-s", type=float, default=120.0)
    ap.add_argument("--tail-queue", default=None,
                    help="DAY-3: queue processed LAST, after the surprise zoom (tail floors)")
    ap.add_argument("--day3", action="store_true",
                    help="DAY-3 mode: PAUSE/STOP sentinels between processes, status.txt, "
                         "per-block ETA, surprise zoom (day3_zoom), block report")
    args = ap.parse_args(argv)

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    telemetry_path = run_dir / "macmon.jsonl"
    floors_path = run_dir / "day_floors.json"
    stop_file = run_dir / "STOP"
    pause_file = run_dir / "PAUSE"
    status_path = run_dir / "status.txt"
    pauses_path = run_dir / "pauses.jsonl"
    deadline = args.deadline_epoch if args.deadline_epoch else (time.time() + 8 * 3600)
    t_start = time.time()
    stat = {"block": "-", "i": 0, "n": 0, "cell": "-", "state": "RUNNING", "floor": "-", "regime": {}}
    block_tot: dict[str, int] = {}
    block_done: dict[str, list[float]] = {}

    def write_status() -> None:
        el = time.time() - t_start
        lines = [f"state: {stat['state']}",
                 f"block: {stat['block']}  cell {stat['i']}/{stat['n']}  ({stat['cell']})",
                 f"elapsed: {el / 60:.1f} min",
                 "remaining (estimate, per block, mean cell time so far):"]
        for b, tot in block_tot.items():
            secs = block_done.get(b, [])
            left = tot - len(secs)
            if secs:
                lines.append(f"  {b}: {len(secs)}/{tot} done, ~{left * (sum(secs) / len(secs)) / 60:.1f} min left")
            else:
                lines.append(f"  {b}: 0/{tot} done, no estimate yet")
        lines.append(f"last floor: {stat['floor']}")
        r = stat.get("regime") or {}
        if r:
            lines.append(f"regime (last cell): GPU {r.get('gpu_mhz')} MHz {r.get('gpu_w')} W busy {r.get('gpu_busy')} "
                         f"{r.get('gpu_c')} °C | CPU {r.get('cpu_w')} W | sys {r.get('sys_w')} W | "
                         f"load {r.get('load1')}/{r.get('load5')}")
        lines.append(f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(status_path)

    def gate() -> None:
        """Between two processes: STOP -> Interrupted; PAUSE -> wait (status PAUSED);
        a FOREIGN GPU/MLX job (not ours) -> wait until it is gone (status PAUSED-FOREIGN)."""
        if stop_file.exists():
            raise Interrupted([])
        foreign = foreign_jobs(strict=False)
        if foreign:
            t_f = time.time()
            stat["state"] = f"PAUSED-FOREIGN ({len(foreign)} job(s): {foreign[0][:90]})"
            write_status()
            _log(run_dir, f"FOREIGN GPU/MLX job(s) detected — paused: {foreign}")
            while foreign and not stop_file.exists():
                time.sleep(5.0)
                foreign = foreign_jobs(strict=False)
            dur = round(time.time() - t_f, 1)
            with pauses_path.open("a") as f:
                f.write(json.dumps({"start": int(t_f), "secs": dur, "kind": "foreign-job",
                                    "block": stat["block"], "cell": stat["cell"]}) + "\n")
            _log(run_dir, f"foreign job(s) gone after {dur}s — resuming")
            stat["state"] = "RUNNING"
            write_status()
            if stop_file.exists():
                raise Interrupted([])
        if pause_file.exists():
            t_p = time.time()
            stat["state"] = "PAUSED"
            write_status()
            _log(run_dir, "PAUSE file present — paused (current process finished)")
            while pause_file.exists() and not stop_file.exists():
                time.sleep(2.0)
            dur = round(time.time() - t_p, 1)
            with pauses_path.open("a") as f:
                f.write(json.dumps({"start": int(t_p), "secs": dur, "block": stat["block"],
                                    "cell": stat["cell"]}) + "\n")
            _log(run_dir, f"PAUSE released after {dur}s")
            stat["state"] = "RUNNING"
            write_status()
            if stop_file.exists():
                raise Interrupted([])

    def stop_reason() -> str | None:
        if stop_file.exists():
            return "stop-file"
        if time.time() >= deadline:
            return "deadline (07:00)"
        return None

    ok, free = ram_ok()
    if not ok:
        _log(run_dir, f"REFUSE launch: RAM {free:.1f} GB below floor"); return 3
    tel_file = run_dir / "telemetry" / "macmon.jsonl"
    if args.day3:
        ok_h, why, reg0 = preflight_host(tel_file)
        (run_dir / "regime_start.json").write_text(json.dumps(reg0, indent=1))
        if not ok_h:
            _log(run_dir, f"REFUSE launch (host preflight): {why} · regime {reg0}")
            status_path.write_text(f"state: REFUSED at launch — {why}\nregime: {reg0}\n")
            return 5
        _log(run_dir, f"host preflight OK: {reg0}")

    # --- day-floor head block (judged-against baseline) ---
    if floors_path.exists():
        floors = json.loads(floors_path.read_text())
        _log(run_dir, f"reuse day-floors {floors_path.name}: sentinel={floors.get('sentinel_ms')} ms")
    else:
        _log(run_dir, "measuring day-floors (dense clean-regime head block)…")
        floors = measure_day_floors()
        floors_path.write_text(json.dumps(floors, indent=1))
        _log(run_dir, f"day-floors: D128={floors.get('dense_d128_default_ms')} ms "
                      f"D64={floors.get('dense_d64_default_ms')} ms (cos "
                      f"{floors.get('dense_d128_default_cos')}/{floors.get('dense_d64_default_cos')})")

    wd = Watchdog(floors, telemetry_path)
    done = _done_ids(results_path)
    if args.day3:
        for cmd in (f"pause : touch {run_dir.resolve()}/PAUSE", f"resume: rm {run_dir.resolve()}/PAUSE",
                    f"stop  : touch {run_dir.resolve()}/STOP", f"status: cat {run_dir.resolve()}/status.txt"):
            _log(run_dir, cmd)
    _log(run_dir, f"resume: {len(done)} cells already in results.jsonl (idempotent skip)")

    # --- work sources, in order: main queue -> zoom -> backlog ---
    main_q = _load_jsonl(Path(args.queue))
    backlog_paths = [p.strip() for p in (args.backlog or "").split(",") if p.strip()]
    t0 = time.time()
    cells_done = 0
    pauses = 0
    zoom_rounds = 0
    all_results: list[dict] = _load_jsonl(results_path)

    def process(queue: list[dict], src: str) -> str | None:
        nonlocal cells_done, pauses
        if args.day3:
            for c in queue:
                b = c.get("block", src)
                block_tot[b] = block_tot.get(b, 0) + 1
                if c.get("cell_id") in done:
                    block_done.setdefault(b, [])
        for cell in queue:
            cid = cell.get("cell_id")
            if not cid or cid in done:
                continue
            sr = stop_reason()
            if sr:
                return sr
            if args.day3:
                b = cell.get("block", src)
                stat.update(block=b, cell=cid, n=block_tot.get(b, 0),
                          i=len(block_done.get(b, [])) + 1)
                write_status()
                try:
                    gate()
                except Interrupted:
                    return "stop-file"
            st, reason = wd.check(cells_done)
            if st == "STOP":
                _log(run_dir, f"WATCHDOG STOP: {reason}")
                return f"watchdog: {reason}"
            while st == "PAUSE":
                pauses += 1
                _log(run_dir, f"WATCHDOG PAUSE ({pauses}/{args.max_pauses}): {reason} — backoff {args.pause_backoff_s}s")
                if pauses >= args.max_pauses:
                    return f"watchdog: too many pauses ({pauses}) — last {reason}"
                time.sleep(args.pause_backoff_s)
                if stop_reason():
                    return stop_reason()
                st, reason = wd.check(cells_done)
                if st == "STOP":
                    return f"watchdog: {reason}"
            if args.day3 and cell.get("kernel") in DAY3_KINDS:
                try:
                    rec = run_day3_cell(cell, run_dir, gate)
                except Interrupted as e:
                    with (run_dir / "partial.jsonl").open("a") as f:
                        f.write(json.dumps({"cell_id": cid, "procs": e.partial,
                                            "ts": int(time.time())}) + "\n")
                    _log(run_dir, f"STOP inside {cid}: {len(e.partial)} finished processes kept in "
                                  f"partial.jsonl; the cell is NOT marked done (re-run whole on resume)")
                    return "stop-file"
            else:
                t_c = time.time()
                rec = run_cell(cell)
                rec["cell_secs"] = round(time.time() - t_c, 1)
                rec["block"] = cell.get("block")
            rec["source"] = src
            if args.day3:
                rec["regime"] = host_regime(tel_file)
                stat["regime"] = rec["regime"]
                block_done.setdefault(cell.get("block", src), []).append(rec.get("cell_secs") or 0.0)
                if rec.get("floor") is not None:
                    stat["floor"] = f"{cid}: {rec['floor']:.4f}"
                elif rec.get("kernel") == "solo_null" and rec.get("margin") is not None:
                    stat["floor"] = f"{cid}: A-vs-A margin {rec['margin']:+.4f}"
                write_status()
            _append(results_path, rec)
            all_results.append(rec)
            done.add(cid)
            cells_done += 1
            if cells_done % 10 == 0:
                rate = cells_done / max(time.time() - t0, 1e-9) * 3600
                _log(run_dir, f"{cells_done} cells done · {rate:.0f} cells/h · last {cid} "
                              f"{rec.get('median_ms')}ms eng={rec.get('engaged')}")
        return None

    reason = process(main_q, "queue")

    # --- DAY-3 surprise zoom (blocks 1-2 verdict flips), until no new cell ---
    while args.day3 and reason is None and not args.no_zoom:
        from .day3_report import day3_zoom
        zoom = [z for z in day3_zoom(all_results) if z["cell_id"] not in done]
        if not zoom:
            break
        zoom_rounds += 1
        _log(run_dir, f"day3 surprise zoom round {zoom_rounds}: {len(zoom)} cells")
        reason = process(zoom, f"zoom{zoom_rounds}")

    # --- auto-zoom: refine around the best regions until dry or deadline ---
    while reason is None and not args.no_zoom and not args.day3:
        zoom = zoom_cells(all_results, floors, top_k=args.zoom_topk, done_ids=done)
        if not zoom:
            break
        zoom_rounds += 1
        _log(run_dir, f"auto-zoom round {zoom_rounds}: {len(zoom)} refined cells around top-{args.zoom_topk}")
        reason = process(zoom, f"zoom{zoom_rounds}")

    if args.day3 and reason is None and args.tail_queue:
        tail = _load_jsonl(Path(args.tail_queue))
        _log(run_dir, f"tail queue: {len(tail)} cells (after zoom)")
        reason = process(tail, "tail")

    for i, bp in enumerate(backlog_paths):        # permanent fall-through: dry → next source, never idle
        if reason is not None:
            break
        cells = _load_jsonl(Path(bp))
        if not cells:
            continue
        _log(run_dir, f"fall-through → source {i + 1}/{len(backlog_paths)}: {Path(bp).name} ({len(cells)} cells)")
        reason = process(cells, f"backlog:{Path(bp).stem}")

    if reason is None:
        reason = "all sources exhausted (queue + zoom + backlog)"

    elapsed = time.time() - t0
    summary = {"stop_reason": reason, "cells_done": cells_done, "elapsed_s": round(elapsed, 1),
               "cells_per_hour": round(cells_done / max(elapsed, 1e-9) * 3600, 1),
               "zoom_rounds": zoom_rounds, "pauses": pauses,
               "episodes": wd.episodes, "day_floors": floors, "deadline_epoch": deadline}
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=1))
    _log(run_dir, f"STOP: {reason} · {cells_done} cells · {summary['cells_per_hour']} cells/h · "
                  f"{zoom_rounds} zoom rounds · {pauses} pauses · {len(wd.episodes)} sentinel checks")
    if args.day3:
        stat.update(state=f"EXITED: {reason}")
        write_status()
    try:
        if args.day3:
            from .day3_report import write_report
        else:
            from .report import write_report
        rpt = write_report(run_dir)
        _log(run_dir, f"morning report → {rpt}")
    except Exception as e:
        _log(run_dir, f"report generation failed (non-fatal): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
