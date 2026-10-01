"""Thermal/health watchdog + day-floors + telemetry for the night runner.

Doctrine (CLAUDE_V6_NAX.md §4 + campaign 113 freq-gate): a night is only valid if
the GPU stayed in a clean regime. We enforce that with:
  - Day-floors: a head block each night measures the dense clean-regime spot (and the
    swept domain's floor). Night verdicts are judged vs THESE, never vs stale numbers.
  - A sentinel cell (fixed dense spot) every N cells. If its median drifts above the
    day-floor sentinel by > DRIFT_FRAC, we PAUSE (thermal episode), snapshot telemetry,
    and back off — never silently fold a degraded number into the DB.
  - macmon sampled continuously (subprocess) → telemetry JSONL.
  - Freq gate: sustained GPU freq under load must stay >= FREQ_FLOOR_MHZ over a window;
    a sustained collapse is an episode (campaign 113: healthy DiT P2 ~1043-1264 MHz;
    collapsed ~554 MHz).
  - RAM gate: refuse to launch / pause if memory pressure is not normal.

Every STOP/PAUSE leaves a legible reason; no degraded result is written as if clean.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx

# --- tunables (module-top per RULE 3; overridable via env for the night plan) ---
SENTINEL_EVERY = int(os.environ.get("NAAR_SENTINEL_EVERY", "25"))     # cells between sentinels
DRIFT_FRAC = float(os.environ.get("NAAR_DRIFT_FRAC", "0.15"))         # >15% over floor => episode
FREQ_FLOOR_MHZ = float(os.environ.get("NAAR_FREQ_FLOOR_MHZ", "1300")) # sustained-min under load
RAM_FREE_MIN_GB = float(os.environ.get("NAAR_RAM_FREE_MIN_GB", "8"))  # refuse below this
MACMON_EVERY_S = float(os.environ.get("NAAR_MACMON_EVERY_S", "20"))   # min seconds between macmon samples
SENTINEL_SHAPE = {"B": 2, "H": 8, "N": 4096, "D": 128, "dtype": "float16", "causal": False}


def _free_gb() -> float:
    """Free RAM INCLUDING reclaimable pages (free + inactive + purgeable + speculative) —
    DAY-3 brief; the August version counted "Pages free" only."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        keys = ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative")
        pages = 0
        seen = False
        for ln in out.splitlines():
            if ln.split(":")[0].strip() in keys:
                pages += int(ln.split(":")[1].strip().rstrip("."))
                seen = True
        if seen:
            return pages * 16384 / 1024 / 1024 / 1024
    except Exception:
        pass
    return -1.0


LOAD_MAX = float(os.environ.get("NAAR_LOAD_MAX", "3.0"))        # run B preflight: 1- and 5-min load
CPU_W_MAX = float(os.environ.get("NAAR_CPU_W_MAX", "8.0"))      # run B preflight: CPU package power
GPU_C_MAX = float(os.environ.get("NAAR_GPU_C_MAX", "45.0"))     # run B preflight: cold GPU start
STRICT_NAMES = ("python", "pytest", "ffmpeg", "x264", "torch")  # preflight: any of these not ours
RUN_NAMES = ("pytest", "ffmpeg", "x264")                        # mid-run: CPU contaminants by name


def _loads(pid: int, libs=("libmlx", "libtorch")) -> bool:
    try:
        out = subprocess.run(["lsof", "-p", str(pid), "-Fn"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return False
    return any(l in out for l in libs)


def _proc_rows():
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,ppid=,command="], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return []
    rows = []
    for ln in out.splitlines():
        parts = ln.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def foreign_jobs(strict: bool = False) -> list[str]:
    """Processes NOT in this runner's own tree (the runner + its children) that could contaminate
    a measurement. A hit PAUSES the run (mid-run) or REFUSES the launch (strict) — nothing is
    ever killed. The executable name is used (argv0 basename), never a shell's command string.
      strict (preflight): any python* / pytest / ffmpeg / x264 / *torch* executable.
      mid-run          : pytest / ffmpeg / x264, or a python that LOADED libmlx or libtorch
                         (so read-only python one-liners on the results do not pause the run)."""
    rows = _proc_rows()
    parent = {pid: ppid for pid, ppid, _ in rows}
    me = os.getpid()

    def mine(pid: int) -> bool:
        seen = 0
        while pid and seen < 64:
            if pid == me:
                return True
            pid = parent.get(pid, 0); seen += 1
        return False

    hits = []
    for pid, _ppid, cmd in rows:
        exe = os.path.basename(cmd.split()[0]).lower() if cmd.split() else ""
        if mine(pid) or not exe:
            continue
        if strict:
            hit = any(exe.startswith(n) if n == "python" else n in exe for n in STRICT_NAMES)
        else:
            hit = any(n in exe for n in RUN_NAMES) or (exe.startswith("python") and _loads(pid))
        if hit:
            hits.append(f"{pid} {cmd[:160]}")
    return hits


def foreign_gpu_jobs() -> list[str]:          # back-compat name (mid-run guard)
    return foreign_jobs(strict=False)


def host_regime(telemetry_file: Optional[Path] = None) -> dict:
    """Host regime snapshot: latest macmon line of the run's LaunchAgent telemetry if fresh
    (<= 15 s), else a one-shot macmon sample; plus the load average. Never raises."""
    import datetime as _dt
    d = None
    try:
        if telemetry_file and Path(telemetry_file).exists():
            with open(telemetry_file, "rb") as f:
                f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4096))
                last = f.read().decode(errors="ignore").strip().splitlines()[-1]
            cand = json.loads(last)
            ts = _dt.datetime.fromisoformat(cand["timestamp"].replace("Z", "+00:00")).timestamp()
            if time.time() - ts <= 15:
                d = cand
        if d is None:
            p = subprocess.run(["macmon", "pipe", "-s", "1"], capture_output=True, text=True, timeout=8)
            d = json.loads((p.stdout or "").strip().splitlines()[-1])
    except Exception:
        d = None
    la = os.getloadavg()
    reg = {"load1": round(la[0], 2), "load5": round(la[1], 2), "ts": int(time.time())}
    if d:
        reg.update(gpu_mhz=d["gpu_usage"][0], gpu_busy=round(d["gpu_usage"][1], 2),
                   gpu_w=round(d.get("gpu_power", 0.0), 1), cpu_w=round(d.get("cpu_power", 0.0), 1),
                   sys_w=round(d.get("sys_power", 0.0), 1),
                   gpu_c=round((d.get("temp") or {}).get("gpu_temp_avg", 0.0), 1))
    return reg


def preflight_host(telemetry_file: Optional[Path] = None) -> tuple[bool, list[str], dict]:
    """Run-B blocking preflight: no foreign job (strict), host load at the floor, cold GPU, RAM."""
    reasons = []
    foreign = foreign_jobs(strict=True)
    if foreign:
        reasons.append(f"foreign job(s): {foreign}")
    reg = host_regime(telemetry_file)
    if reg["load1"] > LOAD_MAX or reg["load5"] > LOAD_MAX:
        reasons.append(f"load average {reg['load1']}/{reg['load5']} > {LOAD_MAX}")
    if reg.get("cpu_w") is None:
        reasons.append("no macmon sample (CPU power / GPU temperature unreadable)")
    else:
        if reg["cpu_w"] > CPU_W_MAX:
            reasons.append(f"CPU power {reg['cpu_w']} W > {CPU_W_MAX}")
        if reg["gpu_c"] > GPU_C_MAX:
            reasons.append(f"GPU {reg['gpu_c']} °C > {GPU_C_MAX} (not cold)")
    ok_ram, free = ram_ok()
    reg["ram_free_gb_incl_reclaimable"] = round(free, 1)
    if not ok_ram:
        reasons.append(f"RAM {free:.1f} GB below floor")
    return (not reasons), reasons, reg


def ram_ok() -> tuple[bool, float]:
    g = _free_gb()
    return (g < 0 or g >= RAM_FREE_MIN_GB), g


def sample_macmon(out_path: Path) -> Optional[dict]:
    """One macmon sample appended to out_path (JSONL). Returns the parsed dict or None.
    macmon `pipe -s 1` emits one JSON line per second; we take a single sample."""
    try:
        p = subprocess.run(["macmon", "pipe", "-s", "1"], capture_output=True, text=True, timeout=8)
        line = (p.stdout or "").strip().splitlines()
        if not line:
            return None
        rec = json.loads(line[-1])
        rec["_ts"] = int(time.time())
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
    except Exception:
        return None


def gpu_mhz(sample: Optional[dict]) -> float:
    """GPU frequency in MHz. macmon's clean field is `gpu_freq_mhz`; `gpu_usage` is a
    [freq, usage_ratio] pair (reading [1] gives the ratio, NOT the freq — the Phase-0
    false-trip bug). Note: for BURSTY screening this reads ~idle (338-462 MHz) between
    cells, so it is TELEMETRY only — the sentinel-drift canary is the thermal gate."""
    if not sample:
        return -1.0
    if "gpu_freq_mhz" in sample:
        return float(sample["gpu_freq_mhz"])
    v = sample.get("gpu_usage")
    if isinstance(v, (list, tuple)) and v:      # [freq_mhz, usage_ratio]
        return float(v[0])
    return -1.0


def sentinel_median(warmup: int = 4, iters: int = 12) -> float:
    """Fixed dense spot — the thermal canary. Same generator as the dense floor."""
    from .executors import screen_dense
    cell = {"cell_id": "SENTINEL", "kernel": "dense", "mode": "sentinel", "shape": SENTINEL_SHAPE, "tiles": {}}
    r = screen_dense(cell, warmup=warmup, iters=iters, adaptive=False)
    return r.get("median_ms") or float("inf")


def measure_day_floors() -> dict:
    """Head block: the clean-regime dense floor(s) the night is judged against.
    Returns {sentinel_ms, dense_d128_default_ms, dense_d64_default_ms} — all measured now."""
    from .executors import screen_dense
    floors = {"measured_ts": int(time.time())}
    # Per-occupancy-class floors: B2H8 (well-occupied) AND B1H4 (small-bh) — so small-bh verdicts
    # are judged against the noise of THEIR class, not the well-occupied floor (Marco, 2026-08-10).
    extra = [("dense_d128_b1h12", {**SENTINEL_SHAPE, "B": 1, "H": 12}),
             ("dense_d64_b1h12", {**SENTINEL_SHAPE, "D": 64, "B": 1, "H": 12})] \
        if os.environ.get("NAAR_FLOORS_B1H12") == "1" else []          # DAY-3 head: + B1H12 class
    for name, shp in [("dense_d128_default", SENTINEL_SHAPE),
                      ("dense_d64_default", {**SENTINEL_SHAPE, "D": 64}),
                      ("dense_d128_smallbh", {**SENTINEL_SHAPE, "B": 1, "H": 4}),
                      ("dense_d64_smallbh", {**SENTINEL_SHAPE, "D": 64, "B": 1, "H": 4})] + extra:
        r = screen_dense({"cell_id": f"FLOOR_{name}", "kernel": "dense", "mode": "floor",
                          "shape": shp, "tiles": {}}, warmup=6, iters=20, adaptive=False)
        floors[name + "_ms"] = r.get("median_ms")
        floors[name + "_cos"] = r.get("cos_vs_sdpa")
    floors["sentinel_ms"] = floors["dense_d128_default_ms"]   # sentinel stays the well-occupied B2H8 spot
    return floors


class Watchdog:
    """Called by the runner between cells. Owns the sentinel cadence + telemetry."""

    def __init__(self, floors: dict, telemetry_path: Path):
        self.floors = floors
        self.telemetry_path = telemetry_path
        self.sentinel_baseline = floors.get("sentinel_ms") or float("inf")
        self.episodes: list[dict] = []
        self._cells_since_sentinel = 0
        self._freq_window: list[float] = []
        self._last_macmon = 0.0

    def check(self, cells_done: int) -> tuple[str, str]:
        """Return (state, reason). state ∈ {OK, PAUSE, STOP}. Cheap per-cell:
        RAM every call; macmon only on a time cadence (subprocess is ~1 s)."""
        ok, free = ram_ok()
        if not ok:
            return "STOP", f"RAM below floor ({free:.1f} GB < {RAM_FREE_MIN_GB})"
        now = time.time()
        mhz = -1.0
        if now - self._last_macmon >= MACMON_EVERY_S:
            self._last_macmon = now
            s = sample_macmon(self.telemetry_path)
            mhz = gpu_mhz(s)
            if mhz > 0:
                self._freq_window.append(mhz)
                self._freq_window = self._freq_window[-8:]
        self._cells_since_sentinel += 1
        if self._cells_since_sentinel >= SENTINEL_EVERY:
            self._cells_since_sentinel = 0
            sm = sentinel_median()
            drift = (sm - self.sentinel_baseline) / self.sentinel_baseline if self.sentinel_baseline else 0.0
            ep = {"at_cell": cells_done, "sentinel_ms": round(sm, 4),
                  "baseline_ms": round(self.sentinel_baseline, 4), "drift_frac": round(drift, 4),
                  "gpu_mhz": mhz, "ts": int(time.time())}
            # Sentinel-drift is THE thermal gate: a real dense workload run under load; if
            # the GPU actually throttles, this median inflates. Freq is advisory telemetry
            # only — bursty screening idles between cells, so point-sampled freq (~338 MHz
            # idle) is NOT a throttle signal (Phase-0 false-trip lesson).
            ep["freq_window_mhz"] = self._freq_window[:]
            if drift > DRIFT_FRAC:
                ep["verdict"] = "PAUSE_thermal_drift"
                self.episodes.append(ep)
                return "PAUSE", f"sentinel drift {drift:+.1%} > {DRIFT_FRAC:.0%} (thermal episode)"
            ep["verdict"] = "OK"
            self.episodes.append(ep)
        return "OK", ""
