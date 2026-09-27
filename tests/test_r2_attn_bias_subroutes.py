"""R2 / MSL-01 / MSL-02 / MSL-06 (code review 2026-09, P0 PUBLISHED) — native `attn_bias`
must never be dropped by a fast sub-route of MFAttention::eval_gpu.

Before the fix: flash-decode (N<=4, S>=256) and STEEL V3 (causal, D=64 N>=4096 B*H>=4;
D=128 N>=2048 on M1/M2) were gated on !block_mask / !rope but NOT !attn_bias, and their
kernel keys hardcode has_attn_bias=false -> the bias was silently IGNORED (output ==
unbiased attention). MFA_DISABLE_V2=1 sent biased calls to STEEL V1, which has no bias
code (MSL-06). Sibling audit (report): V2 single-pass and V2 D-split implement the bias,
V2 split-K already excluded it; V1 now unreachable with a bias.

Every cell checks BOTH: agreement with a biased CPU fp32 oracle AND disagreement with
the unbiased oracle (the bug was "bias ignored", not "bias misapplied").
"""
from __future__ import annotations

import os
import warnings

import mlx.core as mx
import pytest

from mlx_mfa import flash_attention, is_mfa_available

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")

TOL = 1.5e-2       # f16 kernel vs fp32 oracle (V2 bias controls measured 1.6e-3..2.4e-3)
MIN_EFFECT = 0.25  # |biased - unbiased| oracle gap the bias must produce (N(0,4) bias)


def _inputs(N, S, D, H=2, mode=1, seed=0, dtype=mx.float16):
    mx.random.seed(seed)
    q = mx.random.normal((1, H, N, D)).astype(dtype)
    k = mx.random.normal((1, H, S, D)).astype(dtype)
    v = mx.random.normal((1, H, S, D)).astype(dtype)
    bshape = (1, 1, 1, S) if mode == 1 else (1, H, 1, S)
    bias = (4.0 * mx.random.normal(bshape)).astype(dtype)
    return q, k, v, bias


def _oracle(q, k, v, bias, causal):
    with mx.stream(mx.cpu):
        q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
        N, S, D = q.shape[2], k.shape[2], q.shape[3]
        s = (q32 @ mx.swapaxes(k32, -1, -2)) * (D ** -0.5)
        if bias is not None:
            s = s + bias.astype(mx.float32)
        if causal:
            qi = mx.arange(N)[:, None]
            kj = mx.arange(S)[None, :]
            s = s + mx.where(kj <= qi + (S - N), 0.0, float("-inf"))
        o = mx.softmax(s, axis=-1) @ v32
        mx.eval(o)
    return o


def _err(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("S", [512, 4096])
@pytest.mark.parametrize("N", [1, 4, 8])
@pytest.mark.parametrize("mode", [1, 2])
def test_bias_applied_decode_grid(mode, N, S, D, causal):
    q, k, v, bias = _inputs(N, S, D, mode=mode, seed=N * 7 + S + D + causal)
    out = flash_attention(q, k, v, attn_bias=bias, causal=causal)
    ref_b = _oracle(q, k, v, bias, causal)
    ref_nb = _oracle(q, k, v, None, causal)
    assert _err(ref_b, ref_nb) > MIN_EFFECT          # the bias matters for this cell
    assert _err(out, ref_b) < TOL, (_err(out, ref_b), _err(out, ref_nb))


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("N,D,H", [(4096, 64, 4), (2048, 128, 4)])
def test_bias_applied_on_v3_shapes(mode, N, D, H):
    """MSL-02: causal long-context shapes that select STEEL V3 (D=128 V3 is M1/M2 only)."""
    q, k, v, bias = _inputs(N, N, D, H=H, mode=mode, seed=11 + mode)
    out = flash_attention(q, k, v, attn_bias=bias, causal=True)
    ref_b = _oracle(q, k, v, bias, True)
    assert _err(out, _oracle(q, k, v, None, True)) > MIN_EFFECT
    assert _err(out, ref_b) < TOL, _err(out, ref_b)


def test_v3_shape_with_bias_does_not_run_v3(monkeypatch):
    """Which-binary: with a bias the V3 shape must run the same kernel as MFA_DISABLE_V3=1
    (V2) — byte-identical. Before the fix default ran V3 (bias dropped): Δ ≈ 3."""
    q, k, v, bias = _inputs(4096, 4096, 64, H=4, mode=1, seed=21)
    monkeypatch.delenv("MFA_DISABLE_V3", raising=False)
    o_default = flash_attention(q, k, v, attn_bias=bias, causal=True)
    mx.eval(o_default)
    monkeypatch.setenv("MFA_DISABLE_V3", "1")
    o_v2 = flash_attention(q, k, v, attn_bias=bias, causal=True)
    mx.eval(o_v2)
    assert _err(o_default, o_v2) == 0.0


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("D", [64, 128])
def test_decode_with_bias_does_not_run_flash_decode(mode, D):
    """Which-binary for flash-decode (no env knob): non-causal row i depends only on
    (q_i, K, V, bias), so an N=4 call must equal the first 4 rows of an N=8 call (V2).
    Before the fix N=4 took flash-decode and ignored the bias (Δ ≈ 2.8)."""
    q8, k, v, bias = _inputs(8, 1024, D, mode=mode, seed=31 + D)
    o8 = flash_attention(q8, k, v, attn_bias=bias, causal=False)
    o4 = flash_attention(q8[:, :, :4], k, v, attn_bias=bias, causal=False)
    assert _err(o4, o8[:, :, :4]) < 1e-3, _err(o4, o8[:, :, :4])


def test_raw_bias_entry_refuses_disable_v2(monkeypatch):
    """MSL-06: MFA_DISABLE_V2=1 would route a biased call to V1 (no bias code) -> raise."""
    from mlx_mfa._ext import mfa_attention_bias_forward
    q, k, v, bias = _inputs(64, 64, 64, mode=1, seed=41)
    monkeypatch.setenv("MFA_DISABLE_V2", "1")
    with pytest.raises(ValueError, match="MFA_DISABLE_V2"):
        mfa_attention_bias_forward(q, k, v, bias, 1, 64 ** -0.5, True)


def test_public_bias_with_disable_v2_is_still_correct(monkeypatch):
    """Public path: the raw refusal is caught -> one warning + SDPA fallback (correct)."""
    import mlx_mfa.attention as att
    q, k, v, bias = _inputs(64, 64, 64, mode=1, seed=42)
    monkeypatch.setenv("MFA_DISABLE_V2", "1")
    monkeypatch.setattr(att, "_attn_bias_native_warned", False)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        out = flash_attention(q, k, v, attn_bias=bias, causal=True)
    assert _err(out, _oracle(q, k, v, bias, True)) < TOL


def test_bias_survives_disable_v2_set_after_graph_build(monkeypatch):
    """Lazy-eval race: MFA_DISABLE_V2 is read live in eval_gpu. A graph built before the
    knob is set must still apply the bias (the bias bypasses the knob in eval_gpu)."""
    q, k, v, bias = _inputs(64, 64, 64, mode=2, seed=43)
    monkeypatch.delenv("MFA_DISABLE_V2", raising=False)
    out = flash_attention(q, k, v, attn_bias=bias, causal=True)   # graph only (lazy)
    monkeypatch.setenv("MFA_DISABLE_V2", "1")
    mx.eval(out)
    assert _err(out, _oracle(q, k, v, bias, True)) < TOL
