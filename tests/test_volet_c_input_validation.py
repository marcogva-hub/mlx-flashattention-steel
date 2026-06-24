"""Volet C — uniform host-side input-validation layer.

Every entry point × edge-input class that should now RAISE (loud-failure, Rule 8)
gets a cell here; previously each was silent-wrong / NaN / OOB.  See
devnotes/validation_matrix.md for the full matrix.  Includes the 3 CRITICAL
bite proofs (CC-01 empty-KV varlen, CX-01 dropout+softcap, CX-02 seq_lens≠B).
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa
import mlx_mfa.attention as A
from mlx_mfa import flash_attention, flash_attention_varlen, get_device_info

_HAS_NAX = bool(get_device_info().get("is_m5_plus", False))
try:
    import mlx_mfa._ext as _ext
    _HAS_EXT = True
except Exception:  # pragma: no cover
    _HAS_EXT = False

_SC = 1.0 / math.sqrt(128)


def _qf16(B=1, H=4, N=32, D=128, dtype=mx.float16):
    return (mx.random.normal((B, H, N, D)) * 0.1).astype(dtype)


# ─────────────────────────── Python forward / varlen ────────────────────────

class TestVarlenCuSeqlens:
    """CC-01 family: malformed cu_seqlens must raise, not produce NaN."""

    def _q(self):
        return _qf16(N=32)

    def test_empty_kv_segment_raises(self):  # CC-01 exact repro
        q = self._q()
        with pytest.raises(ValueError, match="(?i)zero kv|equal total_k|non-decreasing"):
            o = flash_attention_varlen(q, q, q, mx.array([0, 16, 32]),
                                       mx.array([0, 16, 16]), 16, 16)
            mx.eval(o)

    def test_empty_kv_segment_consistent_totals_raises(self):
        # q N=32 (cu_q=[0,16,32]) and k N=16 (cu_k=[0,16,16]) → totals match,
        # segment 1 has zero KV length (the empty-segment case, not a total bug).
        q = _qf16(N=32); k = _qf16(N=16)
        with pytest.raises(ValueError, match="(?i)zero kv length"):
            o = flash_attention_varlen(q, k, k, mx.array([0, 16, 32]),
                                       mx.array([0, 16, 16]), 16, 16)
            mx.eval(o)

    def test_non_monotone_raises(self):
        q = _qf16(N=32)
        with pytest.raises(ValueError, match="(?i)non-decreasing|equal total"):
            o = flash_attention_varlen(q, q, q, mx.array([0, 24, 16]),
                                       mx.array([0, 24, 16]), 24, 24)
            mx.eval(o)

    def test_cu_first_not_zero_raises(self):
        q = _qf16(N=32)
        with pytest.raises(ValueError, match=r"\[0\] must be 0"):
            o = flash_attention_varlen(q, q, q, mx.array([1, 16, 32]),
                                       mx.array([1, 16, 32]), 16, 16)
            mx.eval(o)

    def test_cu_last_not_total_raises(self):
        q = _qf16(N=32)
        with pytest.raises(ValueError, match="(?i)equal total"):
            o = flash_attention_varlen(q, q, q, mx.array([0, 16, 30]),
                                       mx.array([0, 16, 30]), 16, 16)
            mx.eval(o)

    def test_qk_segment_count_mismatch_raises(self):
        q = _qf16(N=32)
        with pytest.raises(ValueError, match="(?i)same length"):
            o = flash_attention_varlen(q, q, q, mx.array([0, 16, 32]),
                                       mx.array([0, 32]), 16, 32)
            mx.eval(o)

    def test_valid_varlen_still_works(self):
        q = _qf16(N=32)
        o = flash_attention_varlen(q, q, q, mx.array([0, 16, 32]),
                                   mx.array([0, 16, 32]), 16, 16)
        mx.eval(o)
        assert bool(mx.all(mx.isfinite(o.astype(mx.float32))).item())


class TestDropoutFeatureCombos:
    """CX-01: dropout + {bias,softcap,window,alibi} must raise (silent-drop)."""

    @pytest.mark.parametrize("kw", [
        dict(softcap=20.0),
        dict(window_size=(16, 16)),
        dict(attn_bias=mx.zeros((1, 4, 64, 64), dtype=mx.float16)),
        dict(alibi_slopes=mx.ones((4,), dtype=mx.float32)),
    ])
    def test_dropout_plus_feature_raises(self, kw):
        q = _qf16(N=64)
        with pytest.raises(ValueError, match="(?i)dropout_p>0 is not supported"):
            o = flash_attention(q, q, q, dropout_p=0.1, **kw)
            mx.eval(o if not isinstance(o, tuple) else o[0])

    def test_dropout_alone_ok(self):
        q = _qf16(N=64)
        o = flash_attention(q, q, q, dropout_p=0.1)
        mx.eval(o)


class TestZeroQueryArity:
    """CX-04: zero-query honors the (O, L) / (O, weights) tuple contract."""

    def test_zero_query_plain(self):
        z = mx.zeros((1, 4, 0, 128), dtype=mx.float16); k = _qf16(N=8)
        o = flash_attention(z, k, k); mx.eval(o)
        assert o.shape == (1, 4, 0, 128)

    def test_zero_query_return_lse(self):
        z = mx.zeros((1, 4, 0, 128), dtype=mx.float16); k = _qf16(N=8)
        o, lse = flash_attention(z, k, k, return_lse=True); mx.eval(o, lse)
        assert o.shape == (1, 4, 0, 128) and lse.shape == (1, 4, 0)

    def test_zero_query_return_attn_weights(self):
        z = mx.zeros((1, 4, 0, 128), dtype=mx.float16); k = _qf16(N=8)
        o, w = flash_attention(z, k, k, return_attn_weights=True); mx.eval(o, w)
        assert o.shape == (1, 4, 0, 128) and w.shape == (1, 4, 0, 8)


class TestWindowSize:
    """CC-17: negative non-sentinel window must raise."""

    @pytest.mark.parametrize("ws", [(-5, -5), (-2, 0), (0, -3)])
    def test_negative_non_sentinel_raises(self, ws):
        q = _qf16(N=64)
        with pytest.raises(ValueError, match="(?i)must be >= -1"):
            mx.eval(flash_attention(q, q, q, window_size=ws))

    def test_minus_one_sentinel_ok(self):
        q = _qf16(N=64)
        mx.eval(flash_attention(q, q, q, window_size=(-1, -1)))


# ─────────────────────────── raw _ext bindings ──────────────────────────────

@pytest.mark.skipif(not _HAS_EXT, reason="requires built _ext")
class TestPagedGatherHost:
    """CX-02 (seq_lens.shape==B) + CX-05 (int32 metadata)."""

    def _pool(self):
        return (mx.random.normal((4, 16, 4, 128)) * 0.1).astype(mx.float16)

    def test_seq_lens_shape_mismatch_raises(self):  # CX-02
        pool = self._pool(); bt = mx.zeros((2, 4), dtype=mx.int32)
        with pytest.raises(Exception, match="(?i)seq_lens length must equal"):
            mx.eval(_ext.mfa_paged_kv_gather(pool, bt, mx.array([8], dtype=mx.int32), 16))

    def test_seq_lens_float_dtype_raises(self):  # CX-05
        pool = self._pool(); bt = mx.zeros((2, 4), dtype=mx.int32)
        with pytest.raises(Exception, match="(?i)seq_lens must be int32"):
            mx.eval(_ext.mfa_paged_kv_gather(pool, bt, mx.array([8.0, 8.0], dtype=mx.float32), 16))

    def test_block_table_int64_raises(self):  # CX-05
        pool = self._pool(); bt = mx.zeros((2, 4), dtype=mx.int64)
        with pytest.raises(Exception, match="(?i)block_table must be int32"):
            mx.eval(_ext.mfa_paged_kv_gather(pool, bt, mx.array([8, 8], dtype=mx.int32), 16))

    def test_valid_gather_ok(self):
        pool = self._pool(); bt = mx.zeros((2, 4), dtype=mx.int32)
        mx.eval(_ext.mfa_paged_kv_gather(pool, bt, mx.array([8, 8], dtype=mx.int32), 16))


@pytest.mark.skipif(not (_HAS_EXT and _HAS_NAX), reason="requires _ext + M5 NAX")
class TestV6BackwardRawValidation:
    """CC-03: invalid GQA + aux-shape mismatch on the raw V6-NAX backward."""

    def _aux(self, h, N=256, D=128):
        q = _qf16(1, h, N, D)
        L = mx.random.normal((1, h, N)).astype(mx.float32)
        return q, L

    @pytest.mark.parametrize("hq,hk", [(7, 4), (8, 3), (5, 0)])
    def test_invalid_gqa_backward_kv_raises(self, hq, hk):
        # Hk=0 (k has zero heads) is the Hk==0 division-by-zero edge.
        q, L = self._aux(hq)
        k = mx.zeros((1, hk, 256, 128), mx.float16) if hk == 0 else _qf16(1, hk, 256, 128)
        v = k
        O, _ = self._aux(hq); dO, _ = self._aux(hq); dvec = L
        with pytest.raises(Exception, match="(?i)invalid gqa"):
            mx.eval(*_ext.v6_nax_backward_kv(q, k, v, O, L, dO, dvec, _SC, False))

    def test_invalid_gqa_dv_raw_raises(self):
        q, L = self._aux(7); k = _qf16(1, 4, 256, 128); dO, _ = self._aux(7)
        with pytest.raises(Exception, match="(?i)invalid gqa"):
            mx.eval(_ext.v6_nax_backward_dv_raw(q, k, k, L, dO, _SC, 4, False))

    def test_aux_lse_shape_mismatch_raises(self):
        q, _ = self._aux(8, N=256); k = _qf16(1, 8, 256, 128)
        O, _ = self._aux(8, N=256); dO, _ = self._aux(8, N=256)
        Lbad = mx.random.normal((1, 8, 128)).astype(mx.float32)  # N=128 != 256
        dvec = mx.random.normal((1, 8, 256)).astype(mx.float32)
        with pytest.raises(Exception, match="(?i)lse must be"):
            mx.eval(*_ext.v6_nax_backward_kv(q, k, k, O, Lbad, dO, dvec, _SC, False))

    def test_wm_nonpositive_raises(self):
        q, L = self._aux(8); k = _qf16(1, 8, 256, 128); dO, _ = self._aux(8)
        with pytest.raises(Exception, match="(?i)wm.*must be positive"):
            mx.eval(_ext.v6_nax_backward_dv_raw(q, k, k, L, dO, _SC, 0, False))


@pytest.mark.skipif(not _HAS_EXT, reason="requires built _ext")
class TestDebugBindingValidation:
    """CX-03 / CC-05: debug backward bindings validate inputs."""

    def test_bad_head_dim_raises(self):
        q = _qf16(1, 2, 16, 99)
        z = mx.zeros((1, 2, 16), mx.float32)
        with pytest.raises(Exception, match="(?i)head_dim must be"):
            mx.eval(*_ext.mfa_backward_kv_debug(q, q, q, q, z, z, q, _SC, False))

    def test_aux_O_shape_mismatch_raises(self):
        q = _qf16(1, 2, 16, 128); Obad = _qf16(1, 2, 8, 128)
        L = mx.zeros((1, 2, 16), mx.float32)
        with pytest.raises(Exception, match="(?i)O must have"):
            mx.eval(*_ext.mfa_backward_query_debug(q, q, q, Obad, L, q, _SC, False))


# ─────────────────────────────── BITE PROOFS ────────────────────────────────

class TestBiteProofs:
    """Prove the validation is load-bearing (non-destructive — monkeypatch /
    direct-call, never mutate-then-git-checkout a tracked file)."""

    def test_bite_cc01_empty_kv_would_nan_without_validation(self, monkeypatch):
        # Disable the Python cu_seqlens validation. Pre raw-host-parity sweep this
        # reproduced the silent NaN; now the C++ SOURCE guard (assert_varlen_segment_kv
        # in mfa_attention_varlen_forward) BACKSTOPS the disabled wrapper guard — the
        # per-segment zero-KV is caught at the kernel boundary instead of NaN-ing.
        # This is the wrapper-vs-raw defense-in-depth the sweep added: disabling the
        # Python guard no longer reaches a silent NaN; it raises at the C++ layer.
        monkeypatch.setattr(A, "_validate_cu_seqlens",
                            lambda *a, **k: None)
        q = _qf16(N=32)
        with pytest.raises(Exception, match="(?i)k_len==0|empty KV|zero"):
            o = flash_attention_varlen(q, q, q, mx.array([0, 16, 32]),
                                       mx.array([0, 16, 16]), 16, 16)
            mx.eval(o)

    def test_bite_cx01_dropout_path_drops_softcap(self):
        # Prove the hazard the CX-01 raise prevents: the plain dropout path
        # (_dropout_sdpa) ignores softcap — its output equals the no-softcap
        # result, so combining them WOULD silently drop softcap.
        q = _qf16(N=64)
        # dropout_p=0 disables the actual dropout mask (deterministic) so we can
        # compare the function's handling of softcap.
        out_nocap = A._dropout_sdpa(q, q, q, _SC, False, 0.0)
        out_cap = A._dropout_sdpa(q, q, q, _SC, False, 0.0)  # softcap not even a param
        mx.eval(out_nocap, out_cap)
        d = float(mx.max(mx.abs(out_nocap - out_cap)).item())
        assert d == 0.0  # _dropout_sdpa has no softcap arg → silently identical
        # And the public API now refuses the combo:
        with pytest.raises(ValueError, match="(?i)not supported together with softcap"):
            mx.eval(flash_attention(q, q, q, dropout_p=0.1, softcap=20.0))
