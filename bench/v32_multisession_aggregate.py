"""Aggregate across multi-session drift-investigation data.

Reads docs/v6-nax/v32-multisession-data.json (built by
bench/v32_multisession_capture.py) and summarizes per-shape, per-mode
median timings and natural variance across sessions.

Helps answer:
- Is today's "fast legacy" regime stable across days/conditions?
- Does a "deep-idle morning" session reproduce v2.31.0's slower regime?
- What's the realistic bench-to-bench variance to assume for shipping
  decisions?
"""
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "docs/v6-nax/v32-multisession-data.json"

# Reference values for context
V231_LEG = {
    "FlashVSR-dense": 1.115,
    "LTX2-cross":     1.65,
    "SeedVR2-small":  275.6,
    "CogVideoX":      3669.0,
    "SeedVR2-large":  6780.0,
}
V231_V34 = {
    "FlashVSR-dense": 1.55,   # was claimed regression
    "LTX2-cross":     1.42,
    "SeedVR2-small":  170.92,
    "CogVideoX":      2399.19,
    "SeedVR2-large":  4042.73,
}


def main():
    if not DATA_FILE.exists():
        print(f"No multi-session data yet at {DATA_FILE}")
        sys.exit(1)

    ds = json.loads(DATA_FILE.read_text())
    sessions = ds.get("sessions", [])
    if not sessions:
        print("No sessions in dataset")
        sys.exit(1)

    print(f"=== {len(sessions)} sessions in dataset ===")
    for s in sessions:
        print(f"  - {s['session_label']:<40}  {s['time_of_day_bucket']:<14}  cleared={s['cache_cleared_pre_bench']}")
    print()

    # Per shape, per mode, collect medians across sessions
    shapes = ["FlashVSR-dense", "LTX2-cross", "SeedVR2-small", "CogVideoX", "SeedVR2-large"]

    for shape in shapes:
        print(f"=== {shape} ===")
        legacy_medians = []
        v34_medians = []
        for s in sessions:
            rounds = s.get("bench", {}).get(shape, [])
            for r in rounds:
                if not r.get("correctness_ok"):
                    continue
                m = r.get("v6_median_ms")
                if m is None:
                    continue
                if r["mode"] == "legacy":
                    legacy_medians.append(m)
                elif r["mode"] == "v34":
                    v34_medians.append(m)

        if legacy_medians:
            leg_med = statistics.median(legacy_medians)
            leg_min = min(legacy_medians)
            leg_max = max(legacy_medians)
            leg_var = (leg_max - leg_min) / leg_med * 100
            print(f"  legacy:  n={len(legacy_medians):2d}  median={leg_med:8.2f}ms  range=[{leg_min:.2f}, {leg_max:.2f}]  var=±{leg_var/2:.1f}%  v2.31.0 ref={V231_LEG[shape]:.2f}ms")
        if v34_medians:
            v34_med = statistics.median(v34_medians)
            v34_min = min(v34_medians)
            v34_max = max(v34_medians)
            v34_var = (v34_max - v34_min) / v34_med * 100
            print(f"  v34:     n={len(v34_medians):2d}  median={v34_med:8.2f}ms  range=[{v34_min:.2f}, {v34_max:.2f}]  var=±{v34_var/2:.1f}%  v2.31.0 ref={V231_V34[shape]:.2f}ms")

        # Cross-mode delta
        if legacy_medians and v34_medians:
            leg_med = statistics.median(legacy_medians)
            v34_med = statistics.median(v34_medians)
            delta = (v34_med - leg_med) / leg_med * 100
            print(f"  V34 vs legacy median: {delta:+.1f}%")
        print()

    # Summary verdict
    print("=== Verdict so far ===")
    flagged = []
    for shape in shapes:
        rec = []
        for s in sessions:
            rounds = s.get("bench", {}).get(shape, [])
            for r in rounds:
                if r.get("mode") == "legacy" and r.get("correctness_ok"):
                    m = r.get("v6_median_ms", 0)
                    rec.append((s["session_label"], s["time_of_day_bucket"], s["cache_cleared_pre_bench"], m))
        # Find any session within 10% of v2.31.0 baseline
        v231 = V231_LEG[shape]
        for label, tod, cleared, m in rec:
            if abs(m - v231) / v231 < 0.10:
                flagged.append((shape, label, tod, cleared, m, v231))

    if flagged:
        print("Sessions reproducing v2.31.0's slow regime (legacy within ±10%):")
        for shape, label, tod, cleared, m, v231 in flagged:
            print(f"  {shape:<18}  {label:<40}  {tod:<14}  cleared={cleared}  {m:.2f}ms (v2.31.0={v231:.2f})")
    else:
        print("No session reproduces v2.31.0's slow regime within ±10%.")


if __name__ == "__main__":
    main()
