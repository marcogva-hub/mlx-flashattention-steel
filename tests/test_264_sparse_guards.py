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


@nax
def test_oversize_rescue_judges_the_callers_mask_geometry(monkeypatch):
    """The router splits masks to the STEEL geometry before the fallback; the rescue
    must judge the caller's 32x32 mask (here non-aligned N -> the auto_pad kernel)."""
    monkeypatch.setenv("MFA_SPARSE_FALLBACK_MAX_BYTES", str(2**20))
    monkeypatch.setenv("MFA_SPARSE_NAX_LEGACY_POLICY", "1")     # policy says no
    N = 4100
    q, k, v = _qkv(1, 4, N)
    nb = -(-N // 32)
    m = mx.ones((nb, nb), dtype=mx.bool_)
    with dt.capture() as tr:
        o = flash_attention_sparse(q, k, v, m)
        mx.eval(o)
    term = [t for t in tr if not t[1].startswith(dt.REENTRANT_PREFIX)][-1]
    assert term[0] == "v6nax_sparse" and "kv_valid_len" in term[1], term
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(128))
    assert float(mx.max(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32))).item()) < 1e-2


# ── 2.64 pre-merge review fixes ──────────────────────────────────────────────────
def _ref_masked(q, k, v, m32, causal=False, scale=None):
    """SDPA over the element-level operator (bool keep, empty rows -> 0) — independent ref."""
    N, S = q.shape[2], k.shape[2]
    keep = mx.repeat(mx.repeat(m32.astype(mx.bool_), 32, axis=-2), 32, axis=-1)[..., :N, :S]
    if causal:
        keep = keep & (mx.arange(S)[None, :] <= mx.arange(N)[:, None] + max(0, S - N))
    act = mx.any(keep, axis=-1, keepdims=True)
    keep = keep | ~act
    sc = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=keep)
    return mx.where(act, o, mx.zeros_like(o))


@nax
@pytest.mark.parametrize("scale", [0.0, -0.088, float("inf")])
def test_non_positive_or_non_finite_scale_stays_on_sdpa(monkeypatch, scale):
    """Review M2: the V6NAX kernel refuses such scales; the routes must not send them."""
    q, k, v = _qkv(1, 4, 4096)
    nb = 4096 // 32
    for m in (mx.ones((nb, nb), dtype=mx.bool_),                       # B6 (d=1.0)
              (mx.arange(nb)[:, None] - mx.arange(nb)[None, :]) % 40 == 0):  # law (d≈0.025)
        with dt.capture() as tr:
            o = flash_attention_sparse(q, k, v, m, scale=scale)
            mx.eval(o)
        own = [t for t in tr if not t[1].startswith(dt.REENTRANT_PREFIX)]
        assert own[-1][0] == "sdpa", own


@nax
def test_dispatcher_rescues_an_oversize_mask(monkeypatch):
    """Review M1: sparse_attention_dispatch had no B1 rescue (flash_attention_sparse did)."""
    from mlx_mfa.lcsa_nax import sparse_attention_dispatch
    monkeypatch.setenv("MFA_SPARSE_FALLBACK_MAX_BYTES", str(2**20))
    q, k, v = _qkv(1, 4, 4096)
    nb = 4096 // 32
    m = mx.ones((nb, nb), dtype=mx.bool_) & (mx.arange(nb)[None, :] <= mx.arange(nb)[:, None])
    with dt.capture() as tr:                     # causal: outside the causal cells -> fallback path
        o = sparse_attention_dispatch(q, k, v, m, block_tile=32, causal=True)
        mx.eval(o)
    own = [t for t in tr if not t[1].startswith(dt.REENTRANT_PREFIX)]
    assert own[-1][0] == "v6nax_sparse", own
    ref = _ref_masked(q, k, v, m, causal=True)
    assert float(mx.max(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32))).item()) < 1e-2


def test_dispatcher_never_routes_nax_off_m5(monkeypatch):
    """Review M3: the dispatcher's NAX route is M5-gated (the law widened it)."""
    import mlx_mfa.attention as att
    from mlx_mfa.lcsa_nax import sparse_attention_dispatch
    monkeypatch.setattr(att, "_get_is_m5_plus_cached", lambda: False)
    q, k, v = _qkv(1, 12, 4096)
    nb = 4096 // 32
    m = (mx.arange(nb)[:, None] - mx.arange(nb)[None, :]) % 10 == 0
    with dt.capture() as tr:
        mx.eval(sparse_attention_dispatch(q, k, v, m, block_tile=32))
    assert not any(t[0] == "v6nax_sparse" for t in tr), tr


@nax
def test_broadcast_shaped_mask_takes_sdpa(monkeypatch):
    """Review L1: a [1, NQ, NK] mask at H=12 broadcasts on SDPA; the kernel cannot index it."""
    q, k, v = _qkv(1, 12, 4096)
    nb = 4096 // 32
    m = ((mx.arange(nb)[:, None] - mx.arange(nb)[None, :]) % 10 == 0)[None]
    with dt.capture() as tr:
        o = flash_attention_sparse(q, k, v, m)
        mx.eval(o)
    own = [t for t in tr if not t[1].startswith(dt.REENTRANT_PREFIX)]
    assert own[-1][0] == "sdpa", own
    assert bool(mx.array_equal(o, _ref_masked(q, k, v, m)))


@nax
def test_rescued_call_logs_one_terminal(monkeypatch, capsys):
    """Review L3: a rescued call printed 'terminal=sdpa' then 'terminal=v6nax_sparse'."""
    from mlx_mfa import dispatch_policy
    monkeypatch.setenv("MFA_SPARSE_FALLBACK_MAX_BYTES", str(2**20))
    monkeypatch.setenv("MFA_SPARSE_NAX_LEGACY_POLICY", "1")         # policy -> fallback -> rescue
    q, k, v = _qkv(1, 4, 2048)
    m32 = mx.ones((2048 // 32, 2048 // 32), dtype=mx.bool_)
    dispatch_policy._set_verbose(True)
    try:
        capsys.readouterr()
        mx.eval(flash_attention_sparse(q, k, v, m32))
        lines = [l for l in capsys.readouterr().out.splitlines() if "terminal=" in l]
    finally:
        dispatch_policy._set_verbose(False)
    assert len(lines) == 1, lines


@nax
@pytest.mark.parametrize("case", ["rect_law", "quasi_dense", "rescue"])
def test_gradients_through_the_new_routes(monkeypatch, case):
    """Review M2 (tests): B5 rectangular, B6 quasi-dense and the B1 rescue run the SDPA-vjp
    backward of the SAME operator — gradients match an independent reference, and the
    backward is not size-guarded (the forward rescue at 1 MiB must still train)."""
    if case == "rescue":
        monkeypatch.setenv("MFA_SPARSE_FALLBACK_MAX_BYTES", str(2**20))
        monkeypatch.setenv("MFA_SPARSE_NAX_LEGACY_POLICY", "1")
    Nq, Nk = (2048, 8192) if case == "rect_law" else (2048, 2048)
    q, k, v = _qkv(1, 4, Nq, S=Nk)
    nq, nk = Nq // 32, Nk // 32
    if case == "rect_law":
        m = (mx.arange(nq)[:, None] - mx.arange(nk)[None, :]) % 10 == 0
    else:
        m = mx.ones((nq, nk), dtype=mx.bool_)
        m = mx.concatenate([m[:, :-1], mx.zeros((nq, 1), dtype=mx.bool_)], axis=-1)
    g = mx.random.normal(q.shape).astype(mx.float32)

    def L(fn):
        return lambda a, b, c: (fn(a, b, c).astype(mx.float32) * g).sum()
    with dt.capture() as tr:
        got = mx.grad(L(lambda a, b, c: flash_attention_sparse(a, b, c, m)), argnums=(0, 1, 2))(q, k, v)
        mx.eval(*got)
    assert any(t[0] == "v6nax_sparse" for t in tr), tr
    ref = mx.grad(L(lambda a, b, c: _ref_masked(a, b, c, m)), argnums=(0, 1, 2))(q, k, v)
    mx.eval(*ref)
    for a, b in zip(got, ref):
        assert bool(mx.array_equal(a, b)), float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
