"""Analyze v32_kernel_sweep.json — per-shape MFA vs SDPA winner + margins.

Usage:
    .venv/bin/python bench/v32_kernel_sweep_analyze.py docs/v6-nax/v32-kernel-sweep.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("usage: v32_kernel_sweep_analyze.py <sweep.json>", file=sys.stderr)
        sys.exit(2)

    data = json.loads(Path(sys.argv[1]).read_text())
    records = data["records"]

    # Group by shape
    by_shape = defaultdict(dict)
    for r in records:
        if "error" in r or r.get("supported") is False:
            by_shape[r["shape"]][r["backend"]] = r
            continue
        by_shape[r["shape"]][r["backend"]] = r

    print(f"=== V32 niche-shape kernel sweep — analysis ===\n")
    print(f"{'Shape':<22} {'D':>4} {'qL/kL':>13} "
          f"{'sdpa ms':>10} {'mfa ms':>10} {'auto ms':>10} "
          f"{'mfa/sdpa':>10} {'auto/sdpa':>11} {'verdict':<14}")
    print("-" * 116)

    counts = {"sdpa-wins": 0, "mfa-wins": 0, "tied": 0, "mfa-unsupported": 0, "error": 0}
    findings = []

    for shape in sorted(by_shape.keys()):
        b = by_shape[shape]
        sdpa_r = b.get("sdpa", {})
        mfa_r = b.get("mfa", {})
        auto_r = b.get("auto", {})

        # Collect dimensions from any record that has them
        dims = sdpa_r.get("shape_dims") or mfa_r.get("shape_dims") or auto_r.get("shape_dims") or {}
        D = dims.get("D", "?")
        qL = dims.get("qL", "?")
        kL = dims.get("kL", "?")
        qLkL = f"{qL}/{kL}"

        sdpa_ms = sdpa_r.get("median_ms")
        mfa_ms = mfa_r.get("median_ms")
        auto_ms = auto_r.get("median_ms")

        mfa_unsupported = (mfa_r.get("supported") is False) or "error" in mfa_r
        any_error = "error" in sdpa_r or "error" in auto_r

        if any_error:
            verdict = "ERROR"
            counts["error"] += 1
        elif mfa_unsupported and sdpa_ms is not None:
            verdict = "MFA-unsupp→SDPA"
            counts["mfa-unsupported"] += 1
        elif mfa_ms is not None and sdpa_ms is not None:
            ratio = mfa_ms / sdpa_ms
            if ratio < 0.95:
                verdict = "MFA wins"
                counts["mfa-wins"] += 1
            elif ratio > 1.05:
                verdict = "SDPA wins"
                counts["sdpa-wins"] += 1
            else:
                verdict = "tied (±5%)"
                counts["tied"] += 1
        else:
            verdict = "INCOMPLETE"
            counts["error"] += 1

        sdpa_str = f"{sdpa_ms:>10.2f}" if sdpa_ms is not None else f"{'?':>10}"
        mfa_str = f"{mfa_ms:>10.2f}" if mfa_ms is not None else (f"{'unsup':>10}" if mfa_unsupported else f"{'?':>10}")
        auto_str = f"{auto_ms:>10.2f}" if auto_ms is not None else f"{'?':>10}"

        if mfa_ms and sdpa_ms:
            mfa_sdpa = f"{mfa_ms / sdpa_ms:>10.3f}x"
        else:
            mfa_sdpa = f"{'-':>10}"
        if auto_ms and sdpa_ms:
            auto_sdpa = f"{auto_ms / sdpa_ms:>11.3f}x"
        else:
            auto_sdpa = f"{'-':>11}"

        print(f"{shape:<22} {D:>4} {qLkL:>13} {sdpa_str} {mfa_str} {auto_str} "
              f"{mfa_sdpa} {auto_sdpa} {verdict:<14}")

        findings.append({
            "shape": shape, "D": D, "qL": qL, "kL": kL,
            "sdpa_ms": sdpa_ms, "mfa_ms": mfa_ms, "auto_ms": auto_ms,
            "verdict": verdict,
        })

    print()
    print(f"=== Summary ===")
    print(f"  MFA wins:           {counts['mfa-wins']:>2}")
    print(f"  SDPA wins:          {counts['sdpa-wins']:>2}")
    print(f"  Tied (±5%):         {counts['tied']:>2}")
    print(f"  MFA unsupported:    {counts['mfa-unsupported']:>2}")
    print(f"  Errors:             {counts['error']:>2}")
    print()
    print(f"=== Recommendations ===")
    print(f"  Routing rule (forward, M5+ NAX): SDPA when SDPA wins or tied;")
    print(f"  mlx-mfa for the {counts['mfa-wins']} MFA-winning carve-outs and")
    print(f"  the {counts['mfa-unsupported']} MFA-unsupported shapes (which already SDPA-fallback).")

    # Also print MFA-winning shapes for easy carve-out construction
    if counts["mfa-wins"] > 0:
        print()
        print(f"  MFA-winning shapes (need carve-outs in dispatch_policy):")
        for f in findings:
            if f["verdict"] == "MFA wins":
                print(f"    - {f['shape']}: D={f['D']}, qL={f['qL']}, kL={f['kL']}, "
                      f"mfa/sdpa={f['mfa_ms']/f['sdpa_ms']:.3f}x")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
