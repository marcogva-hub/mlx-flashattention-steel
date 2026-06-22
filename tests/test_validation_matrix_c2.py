"""Volet C2 — validation-matrix widening lock.

Round-3's validation matrix had completeness gaps: it covered `mfa_paged_kv_gather`
but missed the shared `_validate_paged_block_table` validator (used by
`flash_attention_paged` / paged-varlen / TQ) and lacked a Q/K/V
mutual-shape-compatibility column. This file locks the widened matrix — every
round-4 finding maps to a cell that turns silent-wrong/OOB into a loud raise:

  CX-02 / CC-01 (CRITICAL) paged batch-cardinality OOB (public + raw STEEL)
  CX-05 (HIGH)             paged float-metadata silent int32 truncation
  CX-04 (HIGH)             raw paged-varlen / TQ no metadata validation
  CX-03 (CRITICAL)         GNA Q/K/V batch/seq/head/head-dim OOB
  CC                       Hk=0 ZeroDivisionError -> clean ValueError

All cells assert a `ValueError` (mlx `std::invalid_argument` surfaces as
ValueError) is raised BEFORE dispatch. Valid-envelope outputs are unchanged
(proved separately by tests/test_oracle_envelope.py + the full suite).
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa
import mlx_mfa._ext as _ext

F16 = mx.float16
_H, _D, _BS, _NBLK = 4, 64, 16, 8


def _pool():
    mx.random.seed(0)
    a = mx.random.normal((_NBLK, _BS, _H, _D)).astype(F16)
    mx.eval(a)
    return a


def _q(B=1, N=1):
    mx.random.seed(1)
    a = mx.random.normal((B, _H, N, _D)).astype(F16)
    mx.eval(a)
    return a


# ── malformed-input cells: (id, callable) — each must raise ValueError ────────
_SCALE = 1.0 / math.sqrt(_D)


def _c(fn):
    return fn


_MALFORMED = {
    # CX-02/CC-01 — public flash_attention_paged batch cardinality
    "paged_pub_seqlens_short": _c(lambda: mlx_mfa.flash_attention_paged(
        _q(2), _pool(), _pool(),
        mx.array([[2, 5, 1, 0], [3, 4, 0, 1]], dtype=mx.int32),
        mx.array([48], dtype=mx.int32), scale=_SCALE, block_size=_BS)),
    "paged_pub_blocktable_short": _c(lambda: mlx_mfa.flash_attention_paged(
        _q(2), _pool(), _pool(),
        mx.array([[2, 5, 1, 0]], dtype=mx.int32),
        mx.array([48, 48], dtype=mx.int32), scale=_SCALE, block_size=_BS)),
    # CX-05 — public float metadata (silent int32 trunc -> reject)
    "paged_pub_float_blocktable": _c(lambda: mlx_mfa.flash_attention_paged(
        _q(1), _pool(), _pool(),
        mx.array([[2, 5, 1, 0]], dtype=mx.float32),
        mx.array([48], dtype=mx.int32), scale=_SCALE, block_size=_BS)),
    "paged_pub_float_seqlens": _c(lambda: mlx_mfa.flash_attention_paged(
        _q(1), _pool(), _pool(),
        mx.array([[2, 5, 1, 0]], dtype=mx.int32),
        mx.array([48], dtype=mx.float32), scale=_SCALE, block_size=_BS)),
    # CX-02 — raw paged STEEL batch cardinality
    "paged_raw_steel_bt_short": _c(lambda: _ext.mfa_paged_steel_forward(
        _q(2), _pool(), _pool(),
        mx.array([[2, 5, 1, 0]], dtype=mx.int32),
        mx.array([48, 48], dtype=mx.int32), _SCALE, False, -1, -1, _BS)),
    "paged_raw_steel_seq_short": _c(lambda: _ext.mfa_paged_steel_forward(
        _q(2), _pool(), _pool(),
        mx.array([[2, 5, 1, 0], [3, 4, 0, 1]], dtype=mx.int32),
        mx.array([48], dtype=mx.int32), _SCALE, False, -1, -1, _BS)),
    # CX-04 — raw paged-varlen metadata cardinality
    "paged_raw_varlen_seq_short": _c(lambda: _ext.mfa_paged_varlen_forward(
        _q(1, 2), _pool(), _pool(),
        mx.array([0, 1, 2], dtype=mx.int32), mx.array([0, 1, 2], dtype=mx.int32),
        mx.array([[2, 5], [3, 4]], dtype=mx.int32), mx.array([16], dtype=mx.int32),
        _SCALE, False, _BS)),
    # CX-03 — GNA Q/K/V mutual-shape-compat
    "gna_batch_mismatch": _c(lambda: mlx_mfa.flash_attention_gna(
        mx.random.normal((2, 2, 64, 128)).astype(F16),
        mx.random.normal((1, 2, 64, 128)).astype(F16),
        mx.random.normal((1, 2, 64, 128)).astype(F16),
        seq_shape=(4, 4, 4), window_size=(2, 2, 2), stride=(1, 1, 1))),
    "gna_kseq_ne_vseq": _c(lambda: mlx_mfa.flash_attention_gna(
        mx.random.normal((1, 2, 64, 128)).astype(F16),
        mx.random.normal((1, 2, 64, 128)).astype(F16),
        mx.random.normal((1, 2, 72, 128)).astype(F16),
        seq_shape=(4, 4, 4), window_size=(2, 2, 2), stride=(1, 1, 1))),
    "gna_head_mismatch": _c(lambda: mlx_mfa.flash_attention_gna(
        mx.random.normal((1, 4, 64, 128)).astype(F16),
        mx.random.normal((1, 2, 64, 128)).astype(F16),
        mx.random.normal((1, 2, 64, 128)).astype(F16),
        seq_shape=(4, 4, 4), window_size=(2, 2, 2), stride=(1, 1, 1))),
    # CC — Hk=0 clean ValueError (was ZeroDivisionError)
    "dense_hk_zero": _c(lambda: mlx_mfa.flash_attention(
        _q(1, 16), mx.zeros((1, 0, 16, 64), F16), mx.zeros((1, 0, 16, 64), F16))),
}


@pytest.mark.parametrize("cid", list(_MALFORMED.keys()))
def test_malformed_input_raises(cid):
    fn = _MALFORMED[cid]
    # mlx's std::invalid_argument host guards surface as Python ValueError.
    with pytest.raises(ValueError):
        out = fn()
        mx.eval(out[0] if isinstance(out, tuple) else out)


# ── valid-envelope cells: must NOT raise (boundary gate adds raises only) ─────
def _ok_paged():
    o = mlx_mfa.flash_attention_paged(
        _q(2), _pool(), _pool(),
        mx.array([[2, 5, 1, 0], [3, 4, 0, 1]], dtype=mx.int32),
        mx.array([48, 48], dtype=mx.int32), scale=_SCALE, block_size=_BS)
    mx.eval(o)
    return o


def _ok_gna():
    o = mlx_mfa.flash_attention_gna(
        mx.random.normal((1, 2, 64, 128)).astype(F16),
        mx.random.normal((1, 2, 64, 128)).astype(F16),
        mx.random.normal((1, 2, 64, 128)).astype(F16),
        seq_shape=(4, 4, 4), window_size=(2, 2, 2), stride=(1, 1, 1))
    mx.eval(o)
    return o


def _ok_raw_steel():
    o = _ext.mfa_paged_steel_forward(
        _q(2), _pool(), _pool(),
        mx.array([[2, 5, 1, 0], [3, 4, 0, 1]], dtype=mx.int32),
        mx.array([48, 48], dtype=mx.int32), _SCALE, False, -1, -1, _BS)
    mx.eval(o[0])
    return o


@pytest.mark.parametrize("cid,fn", [("paged", _ok_paged), ("gna", _ok_gna),
                                     ("raw_steel", _ok_raw_steel)])
def test_valid_envelope_still_runs(cid, fn):
    fn()  # must not raise — validation is a boundary gate, valid paths unchanged
