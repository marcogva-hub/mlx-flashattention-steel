"""Capture one multi-session bench record for the v2.31.0 drift investigation.

Records system conditions + V6NAX/legacy A/B/A bench results across all 5
production shapes, appends to a shared JSON dataset that aggregates
multiple sessions.

Conditions captured:
- timestamp, time-of-day bucket (early-morning/morning/afternoon/evening/night)
- macOS version, hardware uptime
- Metal PSO cache size before clear (set per macOS 26 path)
- Background GPU activity (best-effort via ioreg)

Bench: standard wrapper protocol (3 runs/round, A/B/A across 5 shapes).
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "docs/v6-nax/v32-multisession-data.json"


def _user_cache_dir():
    """Auto-detect macOS user cache dir via getconf (macOS 26+ moved here)."""
    res = subprocess.run(["/usr/bin/getconf", "DARWIN_USER_CACHE_DIR"],
                         capture_output=True, text=True)
    p = res.stdout.strip()
    if not p:
        raise RuntimeError("getconf DARWIN_USER_CACHE_DIR returned empty — running outside macOS?")
    return Path(p)


USER_CACHE_DIR = _user_cache_dir()
PYTHON_METAL_CACHE = USER_CACHE_DIR / "org.python.python/com.apple.metal"


def time_of_day_bucket(t):
    h = t.hour
    if 0 <= h < 6:    return "early-morning"
    if 6 <= h < 12:   return "morning"
    if 12 <= h < 18:  return "afternoon"
    if 18 <= h < 22:  return "evening"
    return "night"


def capture_conditions(label, clear_cache):
    now = datetime.now()
    out = {
        "session_label": label,
        "timestamp_iso": now.isoformat(),
        "time_of_day_bucket": time_of_day_bucket(now),
        "uptime_raw": subprocess.run(["/usr/bin/uptime"], capture_output=True, text=True).stdout.strip(),
        "sw_vers": subprocess.run(["/usr/bin/sw_vers"], capture_output=True, text=True).stdout.strip(),
    }
    if PYTHON_METAL_CACHE.exists():
        sz = subprocess.run(["/usr/bin/du", "-sh", str(PYTHON_METAL_CACHE)],
                            capture_output=True, text=True).stdout.strip()
        out["metal_cache_size_before"] = sz
        # Find oldest and newest cache file mtimes
        files = list(PYTHON_METAL_CACHE.rglob("*"))
        files = [f for f in files if f.is_file()]
        if files:
            mtimes = [f.stat().st_mtime for f in files]
            out["metal_cache_oldest_iso"] = datetime.fromtimestamp(min(mtimes)).isoformat()
            out["metal_cache_newest_iso"] = datetime.fromtimestamp(max(mtimes)).isoformat()
            out["metal_cache_file_count"] = len(files)
    else:
        out["metal_cache_size_before"] = "0 (path missing)"

    if clear_cache:
        out["cache_cleared_pre_bench"] = True
        if PYTHON_METAL_CACHE.exists():
            shutil.rmtree(PYTHON_METAL_CACHE, ignore_errors=True)
            PYTHON_METAL_CACHE.mkdir(parents=True, exist_ok=True)
        metalfe = USER_CACHE_DIR / "org.python.python/com.apple.metalfe"
        if metalfe.exists():
            shutil.rmtree(metalfe, ignore_errors=True)
            metalfe.mkdir(parents=True, exist_ok=True)
    else:
        out["cache_cleared_pre_bench"] = False

    return out


def run_one_subprocess_bench(shape, mode, n_runs, output_path):
    """Invoke bench/v6nax_bench.py as a subprocess (isolation per CLAUDE_V6_NAX.md)."""
    if output_path.exists():
        output_path.unlink()
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "bench/v6nax_bench.py"),
        "--shape", shape,
        "--mode", mode,
        "--runs", str(n_runs),
        "--include-sdpa",
        "--output", str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"bench failed for {shape}/{mode}: {res.stderr}")
    return json.loads(output_path.read_text())


def run_aba_session(label, clear_cache_first, runs_per_round, cooldown_inter_round, cooldown_inter_shape):
    """Run the full A/B/A wrapper inside Python so we can capture metadata cleanly."""
    rec = capture_conditions(label, clear_cache=clear_cache_first)
    rec["protocol"] = {
        "runs_per_round": runs_per_round,
        "cooldown_inter_round_s": cooldown_inter_round,
        "cooldown_inter_shape_s": cooldown_inter_shape,
    }

    print(f"[capture] Conditions: {json.dumps({k: v for k, v in rec.items() if k != 'sw_vers'}, indent=2)}")

    print("[capture] 90s initial cooldown ...")
    time.sleep(90)

    shapes = ["FlashVSR-dense", "LTX2-cross", "SeedVR2-small", "CogVideoX", "SeedVR2-large"]
    rec["bench"] = {}

    out_dir = REPO_ROOT / "outputs/diagnostic/multisession" / label.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    for shape in shapes:
        print(f"[capture] === {shape} ===")
        rounds = []
        for round_label, mode in [("R1", "legacy"), ("R2", "v6nax"), ("R3", "legacy")]:
            opath = out_dir / f"{shape}_{round_label}_{mode}.json"
            d = run_one_subprocess_bench(shape, mode, runs_per_round, opath)
            r = d["records"][-1]
            rounds.append({
                "round": round_label,
                "mode": mode,
                "v6_runs_ms": r.get("v6_runs_ms", []),
                "v6_median_ms": r.get("v6_median_ms"),
                "sdpa_runs_ms": r.get("sdpa_runs_ms", []),
                "sdpa_median_ms": r.get("sdpa_median_ms"),
                "rmse": r.get("rmse"),
                "correctness_ok": r.get("correctness_ok", False),
            })
            print(f"  {round_label} {mode:<7}: v6={r.get('v6_median_ms', 0):.2f}ms sdpa={r.get('sdpa_median_ms', 0):.2f}ms rmse={r.get('rmse', 0):.2e}")
            if round_label != "R3":
                print(f"  cooldown {cooldown_inter_round}s ...")
                time.sleep(cooldown_inter_round)
        rec["bench"][shape] = rounds
        print(f"  inter-shape cooldown {cooldown_inter_shape}s ...")
        time.sleep(cooldown_inter_shape)

    return rec


def append_to_dataset(rec):
    if DATA_FILE.exists():
        ds = json.loads(DATA_FILE.read_text())
    else:
        ds = {"sessions": []}
    ds["sessions"].append(rec)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(ds, indent=2))
    print(f"[capture] Appended session to {DATA_FILE}")
    print(f"[capture] Total sessions in dataset: {len(ds['sessions'])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True,
                   help='Free-text session label, e.g. "session-2-morning-cold-boot"')
    p.add_argument("--clear-cache", action="store_true",
                   help="Clear the Python Metal PSO cache before bench (cold-cache session)")
    p.add_argument("--runs", type=int, default=3, help="Runs per round (A/B/A)")
    p.add_argument("--cooldown-round", type=int, default=60,
                   help="Inter-round cooldown seconds")
    p.add_argument("--cooldown-shape", type=int, default=30,
                   help="Inter-shape cooldown seconds")
    args = p.parse_args()

    rec = run_aba_session(
        label=args.label,
        clear_cache_first=args.clear_cache,
        runs_per_round=args.runs,
        cooldown_inter_round=args.cooldown_round,
        cooldown_inter_shape=args.cooldown_shape,
    )
    append_to_dataset(rec)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
