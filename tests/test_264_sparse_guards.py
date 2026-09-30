"""2.64 B1/B2/B3 — the guards that precede the sparse-envelope promotion.

B1 size guard: the sparse SDPA fallback never materialises an [.., N, S] mask larger
   than MFA_SPARSE_FALLBACK_MAX_BYTES (default 4 GiB); above it the call is routed to
   the V6NAX sparse kernel when the kernel can serve it, else refused LOUDLY (Rule 8).
   The 2.63 default at LongCat N=168,960 x H32 would have built a ~1.8 TB fp16 bias.
B2 bool fallback: every sparse SDPA leg (forward fallbacks AND the SDPA-vjp backward
   legs) receives a BOOL mask — never a float bias — and stays byte-identical to the
   2.63 float-bias operator (rebuilt below as the reference).
B3 density without an fp32 copy of the mask (peak memory).
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa import _dispatch_trace as dt
from mlx_mfa import flash_attention_sparse

nax = pytest.mark.skipif(not mlx_mfa.has_nax(), reason="M5+ sparse routing")


def _qkv(B, H, N, D=128, dtype=mx.float16, S=None, seed=0):
    S = N if S is None else S
    mx.random.seed(seed)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, S, D)).astype(dtype)
    v = mx.random.normal((B, H, S, D)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _ref_263(q, k, v, block_mask, tq, tk, causal):
    """The 2.63 float-bias operator (0/-inf bias, causal -inf add, empty rows -> 0)."""
    N, S = q.shape[2], k.shape[2]
    lead = tuple(block_mask.shape[:-2])
    nq, nk = block_mask.shape[-2:]
    fb = mx.where(block_mask, mx.array(0.0), mx.array(float("-inf")))
    fb = mx.broadcast_to(fb.reshape(*lead, nq, 1, nk, 1), (*lead, nq, tq, nk, tk))
    fb = fb.reshape(*lead, nq * tq, nk * tk)[..., :N, :S].astype(q.dtype)
    if causal:
        fb = fb + mx.triu(mx.full((N, S), float("-inf"), dtype=q.dtype), k=max(0, S - N) + 1)
    row_active = mx.any(fb > float("-inf"), axis=-1, keepdims=True)
    fb = mx.where(row_active, fb, mx.zeros_like(fb))
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), mask=fb)
    return mx.where(row_active, o, mx.zeros_like(o))


class _MaskSpy:
    def __init__(self, monkeypatch):
        self.dtypes = []
        real = mx.fast.scaled_dot_product_attention

        def spy(q, k, v, *a, mask=None, **kw):
            if mask is not None and not isinstance(mask, str):
                self.dtypes.append(mask.dtype)
            return real(q, k, v, *a, mask=mask, **kw)
        monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)


def _mask(nq, nk, heads=None, seed=3):
    mx.random.seed(seed)
    shape = (nq, nk) if heads is None else (heads, nq, nk)
    return mx.random.uniform(shape=shape) < 0.5


# ── B2: bool mask, byte-identical to the 2.63 float bias ─────────────────────────
@nax
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("per_head", [False, True])
def test_perhead_fallback_is_bool_and_byte_identical(monkeypatch, dtype, causal, per_head):
    # 32x16 STEEL geometry -> the M5 per-head SDPA fallback (the FlashVSR route)
    B, H, N = 1, 4, 512
    q, k, v = _qkv(B, H, N, dtype=dtype)
    m = _mask(N // 32, N // 16, heads=H if per_head else None)
    m = mx.concatenate([m[..., :1, :] & False, m[..., 1:, :]], axis=-2)   # an empty row
    spy = _MaskSpy(monkeypatch)
    o = flash_attention_sparse(q, k, v, m, causal=causal)
    mx.eval(o)
    assert spy.dtypes and all(d == mx.bool_ for d in spy.dtypes), spy.dtypes
    monkeypatch.undo()
    ref = _ref_263(q, k, v, m, 32, 16, causal)
    assert bool(mx.array_equal(o, ref)), "bool fallback is not byte-identical to 2.63"


@nax
@pytest.mark.parametrize("causal", [False, True])
def test_nax_route_backward_leg_is_bool_and_gradients_unchanged(monkeypatch, causal):
    """The SDPA-vjp backward of the NAX sparse wrapper: bool mask, same gradients."""
    B, H, N = 1, 12, 4096
    q, k, v = _qkv(B, H, N)
    nb = N // 32
    m = (mx.arange(nb)[:, None] - mx.arange(nb)[None, :]) % 8 == 0
    if causal:
        m = m & (mx.arange(nb)[None, :] <= mx.arange(nb)[:, None])

    def loss_new(q_, k_, v_):
        return mx.sum(flash_attention_sparse(q_, k_, v_, m, causal=causal).astype(mx.float32))

    def loss_ref(q_, k_, v_):
        return mx.sum(_ref_263(q_, k_, v_, m, 32, 32, causal).astype(mx.float32))

    spy = _MaskSpy(monkeypatch)
    g_new = mx.grad(loss_new, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_new)
    assert spy.dtypes and all(d == mx.bool_ for d in spy.dtypes), spy.dtypes
    monkeypatch.undo()
    g_ref = mx.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*g_ref)
    for a, b in zip(g_new, g_ref):
        assert bool(mx.array_equal(a, b)), "backward leg gradients changed"


# ── B1: size guard — arithmetic, NAX rescue, loud refusal ─────────────────────────
def test_fallback_mask_bytes_arithmetic_without_allocating():
    from mlx_mfa.attention import _sparse_fallback_mask_bytes
    # LongCat stage 3 T2V-temporal, per-head BT32 mask: [32, 5280, 5280] -> [32, N, N]
    m_shape = (32, 168960 // 32, 168960 // 32)
    n = _sparse_fallback_mask_bytes(m_shape, 168960, 168960)
    assert n == 32 * 168960 * 168960               # bool: 1 byte / element (~913 GB)
    assert n > 4 * 2**30
    # a head-shared 2-D mask stays head-shared (SDPA broadcasts it)
    assert _sparse_fallback_mask_bytes((64, 256), 2048, 8192) == 2048 * 8192


@nax
def test_oversize_fallback_routes_nax_when_the_kernel_can_serve_it(monkeypatch):
    monkeypatch.setenv("MFA_SPARSE_FALLBACK_MAX_BYTES", str(2**20))      # 1 MiB
    monkeypatch.setenv("MFA_SPARSE_NAX_LEGACY_POLICY", "1")              # force the fallback by policy
    q, k, v = _qkv(1, 4, 2048)
    nb = 2048 // 32
    m = mx.ones((nb, nb), dtype=mx.bool_)                                # density 1.0 -> policy says no
    with dt.capture() as tr:
        o = flash_attention_sparse(q, k, v, m)
        mx.eval(o)
    term = [t for t in tr if not t[1].startswith(dt.REENTRANT_PREFIX)][-1]
    assert term[0] == "v6nax_sparse" and "size guard" in term[1], term


@nax
def test_oversize_fallback_the_kernel_cannot_serve_is_refused(monkeypatch):
    monkeypatch.setenv("MFA_SPARSE_FALLBACK_MAX_BYTES", str(2**20))
    q, k, v = _qkv(1, 4, 2048)
    m = mx.ones((2048 // 32, 2048 // 16), dtype=mx.bool_)                # 32x16: not NAX-servable
    with pytest.raises(RuntimeError, match="MFA_SPARSE_FALLBACK_MAX_BYTES"):
        mx.eval(flash_attention_sparse(q, k, v, m))


# ── B3: density without an fp32 copy ─────────────────────────────────────────────
def test_mask_density_is_exact_and_copy_free():
    from mlx_mfa.lcsa_nax import mask_density
    nb = 4509                                         # N = 144,288 at BT32
    m = mx.zeros((40, nb, nb), dtype=mx.bool_)        # 813 MB bool
    m = mx.logical_or(m, (mx.arange(nb)[None, :] % 10 == 0)[None])
    mx.eval(m)
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    d = mask_density(m)
    peak = mx.get_peak_memory() - base
    expect = sum(1 for j in range(nb) if j % 10 == 0) / nb
    assert abs(d - expect) < 1e-12, (d, expect)
    assert peak < 64 * 2**20, f"density computation peaked at {peak / 2**20:.0f} MB (fp32 copy = 3.25 GB)"
    del m
