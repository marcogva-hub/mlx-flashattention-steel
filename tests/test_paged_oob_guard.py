"""Regression lock for the paged-KV out-of-bounds device-read fix (CC-02 CRITICAL,
CC-03 HIGH).

The paged kernels read a *physical* block id from the caller-supplied
``block_table`` and index the page pool with it.  Before the fix, an out-of-range
id (a stale/garbage entry, or a logical-block index past ``max_blocks``) was used
with no bounds check, producing an out-of-bounds device read (undefined; on Apple
GPU it silently absorbs to 0 but can leak adjacent memory — the audit observed a
non-finite leak in one layout).

Two-layer fix, both asserted here:
  * Layer 1 (in-kernel guard): out-of-range / ``-1`` block ids contribute zero —
    never index the pool out of bounds.  Direct ``_ext`` callers rely on this.
  * Layer 2 (host validation): the public ``flash_attention_paged*`` APIs raise a
    clear ``ValueError`` before dispatch.

``-1`` is a routine padding sentinel (unallocated page) and must stay valid
(contributes zero, no raise).

ENGAGED: uses ``has_nax()`` to ensure the real Metal kernels run (an SDPA fallback
would not exercise the paged gather/guard), forces the kernels via ``_ext`` /
the public paged wrappers, and checks against an independent fp32 oracle.
"""
import math
import numpy as np
import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa.attention import flash_attention_paged

pytestmark = pytest.mark.skipif(
    not mlx_mfa.has_nax(),
    reason="paged Metal kernels require NAX/_ext (has_nax() is False -> SDPA fallback)",
)

_EXT = pytest.importorskip("mlx_mfa._ext")


def _gather_oracle(pool, bt, sl, max_kv):
    """Independent reference: out[b,h,kv,:] = pool[phys, tok, h, :] for valid pages,
    else 0.  Mirrors the documented gather semantics (NOT the kernel)."""
    NB, BS, H, D = pool.shape
    B, MB = bt.shape
    out = np.zeros((B, H, max_kv, D), np.float32)
    for b in range(B):
        for kv in range(int(sl[b])):
            lb, tb = kv // BS, kv % BS
            if lb >= MB:
                continue
            phys = int(bt[b, lb])
            if phys < 0 or phys >= NB:
                continue
            out[b, :, kv, :] = pool[phys, tb, :, :].astype(np.float32)
    return out


def test_gather_valid_with_padding_matches_oracle():
    """Axis 1 + edge: valid pages incl. -1 padding gather correctly (engaged _ext)."""
    rng = np.random.default_rng(0)
    NB, BS, H, D = 8, 4, 2, 16
    pool = rng.standard_normal((NB, BS, H, D)).astype(np.float16)
    bt = np.array([[0, 2, -1, -1], [3, 5, 7, -1]], np.int32)   # -1 padding entries
    sl = np.array([6, 10], np.int32)
    out = np.asarray(
        _EXT.mfa_paged_kv_gather(mx.array(pool), mx.array(bt), mx.array(sl), 12)
        .astype(mx.float32)
    )
    ref = _gather_oracle(pool, bt, sl, 12)
    assert np.all(np.isfinite(out))
    assert np.abs(out - ref).max() < 1e-2


def test_gather_oob_phys_is_safe_zero():
    """Axis 2 (Layer 1): out-of-range phys via direct _ext -> defined zero, no OOB."""
    rng = np.random.default_rng(1)
    NB, BS, H, D = 8, 4, 2, 16
    pool = rng.standard_normal((NB, BS, H, D)).astype(np.float16)
    bt = np.array([[0, 99, -1, -1]], np.int32)   # 99 >= num_blocks=8
    out = np.asarray(
        _EXT.mfa_paged_kv_gather(mx.array(pool), mx.array(bt), mx.array(np.array([8], np.int32)), 8)
        .astype(mx.float32)
    )
    assert np.all(np.isfinite(out))
    # tokens served by the OOB physical block (kv 4..7 -> logical block 1 -> phys 99)
    # must be zero (guard), while tokens from block 0 (kv 0..3) are valid non-padding.
    assert np.all(out[0, :, 4:8, :] == 0)


def test_gather_logblk_overrun_is_safe():
    """Axis 2b (CC-03 secondary): seq_len overruns block_table columns -> safe zero."""
    rng = np.random.default_rng(2)
    NB, BS, H, D = 8, 4, 2, 16
    pool = rng.standard_normal((NB, BS, H, D)).astype(np.float16)
    bt = np.array([[0]], np.int32)               # max_blocks = 1
    out = np.asarray(
        _EXT.mfa_paged_kv_gather(mx.array(pool), mx.array(bt), mx.array(np.array([8], np.int32)), 8)
        .astype(mx.float32)
    )
    assert np.all(np.isfinite(out))
    assert np.all(out[0, :, 4:8, :] == 0)        # kv 4..7 -> logical block 1 >= max_blocks


@pytest.mark.parametrize("bad", [
    np.array([[0, 99, -1, -1]], np.int32),       # phys >= num_blocks
    np.array([[0, -5, -1, -1]], np.int32),       # phys < -1 (not the -1 sentinel)
])
def test_public_paged_raises_on_out_of_range(bad):
    """Axis 2c (Layer 2): public API raises ValueError before dispatch."""
    rng = np.random.default_rng(3)
    NB, BS, H, D = 8, 4, 2, 16
    q = mx.array(rng.standard_normal((1, 2, 4, D)).astype(np.float16))
    kp = mx.array(rng.standard_normal((NB, BS, H, D)).astype(np.float16))
    vp = mx.array(rng.standard_normal((NB, BS, H, D)).astype(np.float16))
    with pytest.raises(ValueError, match="block_table"):
        out = flash_attention_paged(q, kp, vp, mx.array(bad),
                                    mx.array(np.array([8], np.int32)), block_size=BS)
        mx.eval(out)


def test_public_paged_valid_with_padding_matches_oracle():
    """Axis 3: valid paged attention incl. -1 padding matches a fp32 gather+softmax oracle."""
    rng = np.random.default_rng(4)
    NB, BS, H, D = 8, 4, 2, 16
    q = mx.array(rng.standard_normal((1, 2, 4, D)).astype(np.float16))
    kp = mx.array(rng.standard_normal((NB, BS, H, D)).astype(np.float16))
    vp = mx.array(rng.standard_normal((NB, BS, H, D)).astype(np.float16))
    bt = np.array([[0, 1, -1, -1]], np.int32)
    sl = np.array([6], np.int32)
    out = np.asarray(
        flash_attention_paged(q, kp, vp, mx.array(bt), mx.array(sl),
                              block_size=BS, causal=False).astype(mx.float32)
    )
    Kg = _gather_oracle(np.asarray(kp.astype(mx.float32)), bt, sl, 6)
    Vg = _gather_oracle(np.asarray(vp.astype(mx.float32)), bt, sl, 6)
    qn = np.asarray(q.astype(mx.float32))
    sc = 1.0 / math.sqrt(D)
    ref = np.zeros_like(qn)
    for h in range(2):
        s = (qn[0, h] @ Kg[0, h].T) * sc
        m = s.max(1, keepdims=True)
        p = np.exp(s - m)
        p /= p.sum(1, keepdims=True)
        ref[0, h] = p @ Vg[0, h]
    assert np.all(np.isfinite(out))
    assert np.abs(out - ref).max() < 2e-2


def test_paged_steel_oob_is_finite():
    """Layer 1 on the fused PagedSteelForward kernel (the _ext.mfa_paged_steel_forward
    path): out-of-range phys -> finite output (no OOB read), not garbage/NaN."""
    rng = np.random.default_rng(5)
    NB, BS, H, D = 8, 4, 2, 16
    q = mx.array(rng.standard_normal((1, 2, 4, D)).astype(np.float16))
    kp = mx.array(rng.standard_normal((NB, BS, H, D)).astype(np.float16))
    vp = mx.array(rng.standard_normal((NB, BS, H, D)).astype(np.float16))
    bt = mx.array(np.array([[0, 99, -1, -1]], np.int32))
    o, _ = _EXT.mfa_paged_steel_forward(
        q, kp, vp, bt, mx.array(np.array([8], np.int32)),
        1.0 / math.sqrt(D), False, -1, -1, BS)
    mx.eval(o)
    assert np.all(np.isfinite(np.asarray(o.astype(mx.float32))))
