"""DAY-3 Block 1 arm — the July gate-remap arm with the per-row magnitude gates (2026-09-30).

REUSES benchmarks/bench_sparse_gate_remap.py of the mlx-mfa checkout unchanged: same cell
builder, same preflight (trace-terminal engagement, dtype, fp32 oracle cos, run-twice
determinism, distinct-path byteΔ), same timing (WARMUPS x DISPATCHES + SESSIONS medians).
Adds ONE refusal condition, required by the DAY-3 brief: the per-row magnitude gates of
tests/sparse_gates.py (finite, max-abs, per-row norm ratio, per-row relative L2 — cosine is
magnitude-blind) on BOTH arms' outputs vs the same fp32 oracle, thresholds = the repo's
unit-variance gates (_G16 / _GBF16 of tests/test_sparse_extended_envelope.py).

Forcing the NAX kernel at B·H outside the measured gate is done by the CALLER through the
public contract (env MFA_SPARSE_NAX_EXTENDED=1 on the process) — the bench's preflight then
refuses to time unless the public terminal really is ``v6nax_sparse``.

  python day3_gate_arm.py --mfa-root <mlx-mfa checkout> <bench args…>
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROW_GATES = {  # tests/test_sparse_extended_envelope.py _G16 / _GBF16 (row_rel = 2 x norm_tol)
    "fp16": {"max_abs": 1e-2, "norm_tol": 5e-3, "row_rel": 1e-2},
    "bf16": {"max_abs": 3e-2, "norm_tol": 2e-2, "row_rel": 4e-2},
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) < 2 or argv[0] != "--mfa-root":
        raise SystemExit("usage: day3_gate_arm.py --mfa-root <mlx-mfa checkout> <bench args...>")
    root = Path(argv[1]).resolve()
    sys.path.insert(0, str(root))                         # tests.sparse_gates importable
    bench = _load(root / "benchmarks" / "bench_sparse_gate_remap.py", "bench_sparse_gate_remap")
    from tests.sparse_gates import row_gate_report

    orig_preflight = bench.preflight

    def preflight_with_row_gates(args, q, block_mask, public, baseline, oracle):
        fp = orig_preflight(args, q, block_mask, public, baseline, oracle)   # July guards first
        ref = oracle()
        reports = {}
        for name, fn in (("public", public), ("baseline", baseline)):
            out = fn()
            bench.evaluate((out, ref))
            reports[name] = row_gate_report(out, ref)
        th = ROW_GATES[args.dtype]
        failures = []
        for name, rep in reports.items():
            if not rep["finite"]:
                failures.append(f"{name} non-finite")
            if rep["max_abs"] >= th["max_abs"]:
                failures.append(f"{name} max_abs={rep['max_abs']:.3g}>={th['max_abs']}")
            if rep["worst_row_norm_dev"] >= th["norm_tol"]:
                failures.append(f"{name} row_norm_dev={rep['worst_row_norm_dev']:.3g}>={th['norm_tol']}")
            if rep["worst_row_rel"] >= th["row_rel"]:
                failures.append(f"{name} row_rel={rep['worst_row_rel']:.3g}>={th['row_rel']}")
        if failures:
            raise RuntimeError("refusing sparse gate benchmark: per-row magnitude gates failed: "
                               + "; ".join(failures))
        fp["row_gates"] = {"thresholds": th, **reports}
        return fp

    bench.preflight = preflight_with_row_gates
    sys.argv = [str(root / "benchmarks" / "bench_sparse_gate_remap.py")] + argv[2:]
    bench.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
