"""Tier-3 hygiene guards (CC-10/11/12, CC-26) — degenerate-input / misuse paths
now produce a defined result or a clear error instead of NaN / a cryptic raise."""
import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa.attention import flash_attention, flash_attention_sparse


def _qkv(N, S, D=64, H=4):
    f = lambda n: mx.random.normal((1, H, n, D)).astype(mx.float16)
    return f(N), f(S), f(S)


# ── CC-10: empty-KV / empty-query ──────────────────────────────────────────
def test_cc10_zero_queries_returns_empty():
    q, k, v = _qkv(0, 8)
    out = flash_attention(q, k, v, scale=0.125)
    mx.eval(out)
    assert out.shape == (1, 4, 0, 64)  # defined empty result, no NaN


def test_cc10_zero_keys_raises():
    q, k, v = _qkv(4, 0)
    with pytest.raises(ValueError, match="empty KV|zero keys|k_seq=0"):
        out = flash_attention(q, k, v, scale=0.125)
        mx.eval(out)


def test_cc10_normal_still_finite():
    q, k, v = _qkv(4, 8)
    out = flash_attention(q, k, v, scale=0.125)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item())


# ── CC-11: window_size shape validation ────────────────────────────────────
@pytest.mark.parametrize("bad", [128, (1, 2, 3), (5,), [1, 2, 3, 4]])
def test_cc11_bad_window_size_raises(bad):
    q, k, v = _qkv(8, 8)
    with pytest.raises(ValueError, match="window_size"):
        out = flash_attention(q, k, v, scale=0.125, window_size=bad)
        mx.eval(out)


def test_cc11_valid_window_size_ok():
    q, k, v = _qkv(64, 64, D=64, H=4)
    out = flash_attention(q, k, v, scale=0.125, causal=True, window_size=(32, 0))
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item())


# ── CC-12: sparse block_mask=None ──────────────────────────────────────────
def test_cc12_sparse_none_mask_raises_clear():
    q, k, v = _qkv(64, 64)
    with pytest.raises(ValueError, match="block_mask"):
        out = flash_attention_sparse(q, k, v, None, scale=0.125)
        mx.eval(out)


# ── CC-26: compile_metallib surfaces a compile failure (no silent sentinel) ─
def test_cc26_compile_failure_is_loud(capsys):
    import shutil, tempfile, os
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun not available")
    from mlx_mfa.compile_metallib import _compile_source_to_metallib
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "bad.metallib")
        ok = _compile_source_to_metallib("this is not valid metal source;", out)
    assert ok is False
    err = capsys.readouterr().err
    assert "COMPILE FAILED" in err or "COMPILE ERROR" in err  # diagnostic surfaced
