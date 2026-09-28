"""Shared correctness gates for the block-sparse / auto_pad / SLA tests (review 2026-09,
DSP-14 / TST-04; decision pre-taken in the remediation brief).

A global cosine is MAGNITUDE-BLIND: the U1 defect (pad keys in the softmax denominator)
scaled whole rows (2-20 % on random inputs, x0.026 in the review's TST-01 repro) while the
global cosine stayed >= 0.999, and every Volet A/B gate passed.  These gates check each query row against an fp32 oracle:

  * shapes identical                                      (no silent broadcast)
  * max-abs error over all elements                       (absolute; scale-dependent)
  * per-row norm ratio |o_i| / |ref_i|, within 1 +- tol    (catches row scaling — U1)
  * per-row RELATIVE L2 error |o_i - ref_i| / |ref_i|     (scale-invariant: catches
    swapped / shifted / permuted rows and wrong keys at any input magnitude)
  * global cosine — a complement, never the only gate.

Thresholds are explicit at every call site.  Pre-checkpoint review (Phase B): with the
Volet tests' 0.1-scaled inputs, attention is near-uniform and several WRONG outputs
(Q ignored, rows shifted, 0.5% scaling) passed max-abs/norm/cos — the tests now use
unit-variance inputs and this relative per-row gate.
"""
from __future__ import annotations

import mlx.core as mx


def row_gate_report(out, ref, *, live_norm: float = 1e-3) -> dict:
    """Per-row magnitude statistics of ``out`` vs the fp32 oracle ``ref`` (same shape,
    rows on axis -2, features on axis -1).  Rows whose oracle norm is below
    ``live_norm`` (empty / near-zero rows) are excluded from the ratio statistic."""
    o = out.astype(mx.float32)
    r = ref.astype(mx.float32)
    diff = mx.abs(o - r)
    rn = mx.sqrt(mx.sum(r * r, axis=-1))
    on = mx.sqrt(mx.sum(o * o, axis=-1))
    live = rn > live_norm
    ratio_dev = mx.where(live, mx.abs(on / mx.maximum(rn, 1e-12) - 1.0), 0.0)
    row_rel = mx.where(live, mx.sqrt(mx.sum((o - r) ** 2, axis=-1)) / mx.maximum(rn, 1e-12), 0.0)
    both_zero = (mx.sum(o * o) == 0) & (mx.sum(r * r) == 0)     # cos of 0 vs 0 := 1
    cos = mx.where(both_zero, 1.0,
                   mx.sum(o * r) / mx.maximum(mx.sqrt(mx.sum(o * o) * mx.sum(r * r)), 1e-30))
    mx.eval(diff, ratio_dev, row_rel, cos)
    return {
        "max_abs": float(mx.max(diff)),
        "worst_row_norm_dev": float(mx.max(ratio_dev)),
        "worst_row_rel": float(mx.max(row_rel)),
        "cos": float(cos),
        "finite": bool(mx.all(mx.isfinite(o)).item()),
    }


def assert_row_gates(out, ref, *, max_abs: float, norm_tol: float,
                     row_rel: float | None = None, cos_min: float = 0.999,
                     label: str = "") -> dict:
    """``row_rel`` (per-row relative L2) defaults to 2 x ``norm_tol``."""
    assert tuple(out.shape) == tuple(ref.shape), (label, "shape", out.shape, ref.shape)
    rep = row_gate_report(out, ref)
    assert rep["finite"], (label, "non-finite output", rep)
    assert rep["max_abs"] < max_abs, (label, "max_abs", rep)
    assert rep["worst_row_norm_dev"] < norm_tol, (label, "per-row norm ratio", rep)
    tol_rel = 2 * norm_tol if row_rel is None else row_rel
    assert rep["worst_row_rel"] < tol_rel, (label, "per-row relative L2", rep)
    assert rep["cos"] > cos_min, (label, "global cosine (complement)", rep)
    return rep
