"""Shared correctness gates for the block-sparse / auto_pad / SLA tests (review 2026-09,
DSP-14 / TST-04; decision pre-taken in the remediation brief).

A global cosine is MAGNITUDE-BLIND: the U1 defect (pad keys in the softmax denominator)
scaled whole rows by up to x0.026 while the global cosine stayed >= 0.999, and every
Volet A/B gate passed.  These gates check each query row against an fp32 oracle:

  * max-abs error over all elements                      (catches local errors)
  * per-row norm ratio |o_i| / |ref_i|, within 1 +- tol   (catches row scaling — U1)
  * per-row max-abs error                                 (catches row-local errors)
  * global cosine — a complement, never the only gate.

Thresholds are explicit at every call site.
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
    both_zero = (mx.sum(o * o) == 0) & (mx.sum(r * r) == 0)     # cos of 0 vs 0 := 1
    cos = mx.where(both_zero, 1.0,
                   mx.sum(o * r) / mx.maximum(mx.sqrt(mx.sum(o * o) * mx.sum(r * r)), 1e-30))
    mx.eval(diff, ratio_dev, cos)
    return {
        "max_abs": float(mx.max(diff)),
        "worst_row_max_abs": float(mx.max(mx.max(diff, axis=-1))),
        "worst_row_norm_dev": float(mx.max(ratio_dev)),
        "cos": float(cos),
        "finite": bool(mx.all(mx.isfinite(o)).item()),
    }


def assert_row_gates(out, ref, *, max_abs: float, norm_tol: float,
                     cos_min: float = 0.999, label: str = "") -> dict:
    rep = row_gate_report(out, ref)
    assert rep["finite"], (label, "non-finite output", rep)
    assert rep["max_abs"] < max_abs, (label, "max_abs", rep)
    assert rep["worst_row_norm_dev"] < norm_tol, (label, "per-row norm ratio", rep)
    assert rep["cos"] > cos_min, (label, "global cosine (complement)", rep)
    return rep
