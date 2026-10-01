"""U1 / U2 (review 2026-09, P0 unreleased) — `flash_attention_sparse(auto_pad=True)`.

U1: zero-padded KEYS (k = 0 -> score 0) entered the softmax denominator: every row whose
    active blocks include the ragged final block was scaled down (2-20 % on these random
    inputs; x0.026 in the review's TST-01 repro) while the global cosine stayed >= 0.999 —
    the Volet A gates were blind.  Causal cells do not discriminate (causality already
    hid the pads); the non-causal cells fail on the pre-fix route (checked by emulation).
U2: causal N != S padded Q and K separately, shifting the causal diagonal
    S-N -> Spad-Npad (queries saw future keys).

Fix (2.63.0): the V6NAX sparse kernel masks keys >= kv_valid_len element-wise; auto_pad
pads only when that kernel will run (predicate `_auto_pad_nax_route`, locked against the
router below), only for N == S (the kernel is square-only), and every other call is the
exact unpadded route.  Promoted repros: API-01, DSP-02, DOC-10, MSL-04, NEPB-01,
NEPF-01, STA-08, TST-01 (U1) and DSP-03, API-06, NEPF-02, NEPB-02 (U2) — all exited 1
before the fix, 0 after.  Gates: per-row magnitude (tests/sparse_gates.py).
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

import mlx_mfa.attention as att
from mlx_mfa import _dispatch_trace as dt
from mlx_mfa import flash_attention_sparse, is_mfa_available
from tests.sparse_gates import assert_row_gates

pytestmark = pytest.mark.skipif(
    not (is_mfa_available() and att._get_is_m5_plus_cached()),
    reason="MFA extension + M5+ required (the V6NAX sparse kernel is M5+ only)")

F16_GATES = dict(max_abs=1e-2, norm_tol=5e-3, cos_min=0.9999)   # observed row-norm dev ~4e-4


def _inputs(N, S, D, H, density, seed, dtype=mx.float16):
    mx.random.seed(seed)
    q = mx.random.normal((1, H, N, D)).astype(dtype)
    k = mx.random.normal((1, H, S, D)).astype(dtype)
    v = mx.random.normal((1, H, S, D)).astype(dtype)
    nq, nk = -(-N // 32), -(-S // 32)
    eye = mx.eye(nq, nk, dtype=mx.bool_)
    bm = eye | (mx.random.uniform(shape=(nq, nk)) < density)
    bm[..., -1] = True            # every row block touches the ragged final key block
    mx.eval(q, k, v, bm)
    return q, k, v, bm


def _oracle(q, k, v, bm, causal):
    N, S, D = q.shape[2], k.shape[2], q.shape[3]
    with mx.stream(mx.cpu):
        tok = mx.repeat(mx.repeat(bm, 32, axis=-2), 32, axis=-1)[..., :N, :S]
        s = (q.astype(mx.float32) @ mx.swapaxes(k.astype(mx.float32), -1, -2)) * D ** -0.5
        vis = tok
        if causal:
            vis = vis & (mx.arange(S)[None] <= mx.arange(N)[:, None] + max(0, S - N))
        s = mx.where(vis, s, float("-inf"))
        o = mx.softmax(s, axis=-1) @ v.astype(mx.float32)
        o = mx.where(mx.any(vis, axis=-1, keepdims=True), o, 0.0)
        mx.eval(o)
    return o


def _padded_route_ran(cap) -> bool:
    return any(r[0] == "v6nax_sparse" and "kv_valid" in r[1] for r in cap)


# ── U1: the padded kernel route (auto_pad, kv_valid_len) ─────────────────────────────
# 2.64: reached by the DEFAULT law (non-causal, e.g. sla_attention / Wan) or by the B1
# size-guard rescue (causal too).  Locked at kernel level on every cell; the public path
# must equal that kernel wherever the law routes it, and the oracle everywhere.
@pytest.mark.parametrize("N,D,H", [(4100, 128, 8), (4100, 128, 4), (4010, 64, 4),
                                   (4097, 128, 2), (2050, 128, 4)])   # 4097: 31 pad keys
@pytest.mark.parametrize("causal", [False, True])
def test_u1_extended_pad_keys_out_of_denominator(monkeypatch, N, D, H, causal):
    from mlx_mfa.attention import _make_sparse_nax_padded_vjp
    q, k, v, bm = _inputs(N, N, D, H, 0.1, seed=N + D + H)
    ref = _oracle(q, k, v, bm, causal)
    o = _make_sparse_nax_padded_vjp(1.0 / math.sqrt(D), causal)(q, k, v, bm)
    mx.eval(o)
    assert_row_gates(o, ref, **F16_GATES, label=f"kernel N={N} D={D} H={H} causal={causal}")
    # the rows the defect scaled: the final query block (and every row, above)
    assert_row_gates(o[..., -32:, :], ref[..., -32:, :], **F16_GATES, label="tail rows")
    with dt.capture() as cap:
        op = flash_attention_sparse(q, k, v, bm, causal=causal, auto_pad=True)
        mx.eval(op)
    from mlx_mfa.attention import _auto_pad_nax_route
    routed = _auto_pad_nax_route(q, k, v, bm, causal)       # the law, as the router applies it
    assert _padded_route_ran(cap) == routed, ([r[:2] for r in cap], routed)
    if routed:
        assert bool(mx.array_equal(op, o)), "public padded route != the padded kernel"
    assert_row_gates(op, ref, **F16_GATES, label=f"public N={N} D={D} H={H} causal={causal}")


def test_u1_auto_pad_equals_unpadded(monkeypatch):
    """Contract: auto_pad changes the route (speed), never the result."""
    q, k, v, bm = _inputs(4100, 4100, 128, 4, 0.1, seed=3)
    a = flash_attention_sparse(q, k, v, bm, auto_pad=True)
    b = flash_attention_sparse(q, k, v, bm, auto_pad=False)
    assert_row_gates(a, b, **F16_GATES, label="auto_pad vs unpadded")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["f16", "bf16"])
def test_u1_default_policy(dtype):
    """Default env: whichever route the policy picks for the padded shape, the result is
    exact (the predicate decides; the anti-drift lock keeps it honest)."""
    q, k, v, bm = _inputs(4100, 4100, 128, 12, 0.1, seed=4, dtype=dtype)
    o = flash_attention_sparse(q, k, v, bm, auto_pad=True)
    tol = F16_GATES if dtype == mx.float16 else dict(max_abs=3e-2, norm_tol=2e-2, cos_min=0.999)
    assert_row_gates(o, _oracle(q, k, v, bm, False), **tol, label=str(dtype))


def test_u1_gradients_match_oracle(monkeypatch):
    N, D = 4100, 64
    q, k, v, bm = _inputs(N, N, D, 2, 0.1, seed=5)
    mx.random.seed(6)
    g = mx.random.normal(q.shape).astype(mx.float32)
    with dt.capture() as cap:
        grads = mx.grad(lambda a, b, c: (flash_attention_sparse(a, b, c, bm, auto_pad=True)
                                         .astype(mx.float32) * g).sum(), argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
    assert _padded_route_ran(cap)
    with mx.stream(mx.cpu):
        tok = mx.repeat(mx.repeat(bm, 32, axis=-2), 32, axis=-1)[..., :N, :N]

        def f(a, b, c):
            s = mx.where(tok, (a @ mx.swapaxes(b, -1, -2)) * D ** -0.5, float("-inf"))
            return ((mx.softmax(s, axis=-1) @ c) * g).sum()
        ref = mx.grad(f, argnums=(0, 1, 2))(*(x.astype(mx.float32) for x in (q, k, v)))
        mx.eval(*ref)
    for name, x, y in zip(("dQ", "dK", "dV"), grads, ref):
        rel = float(mx.max(mx.abs(x.astype(mx.float32) - y))) / float(mx.max(mx.abs(y)))
        assert rel < 1e-2, (name, rel)


# ── U2: causal N != S — no padding, canonical offset, no future key ────────────────────
@pytest.mark.parametrize("N,S", [(4010, 4100), (90, 4100), (2050, 2080)])
@pytest.mark.parametrize("extended", [False, True])
def test_u2_causal_n_ne_s_sees_no_future_key(monkeypatch, N, S, extended):
    if extended:
        pass   # 2.64: the extended opt-in is a no-op (same default law either way)
    D, H = 64, 2
    q, k, v, bm = _inputs(N, S, D, H, 0.2, seed=N + S)
    bm = mx.ones_like(bm)                          # the causal mask alone decides visibility
    with dt.capture() as cap:
        o = flash_attention_sparse(q, k, v, bm, causal=True, auto_pad=True)
        mx.eval(o)
    assert not _padded_route_ran(cap)              # N != S is never padded
    assert_row_gates(o, _oracle(q, k, v, bm, True), **F16_GATES, label=f"N={N} S={S}")
    # perturb V only at FUTURE keys (j > i + S - N for EVERY row i -> j > N-1 + S-N = S-1:
    # none) — so perturb per row block instead: keys beyond row 0's horizon change only
    # rows that must not see them.
    off = max(0, S - N)
    rows = N // 2                                  # rows 0..N//2-1 see keys <= i + off
    horizon = off + rows                           # = the FIRST future key of the last row
    v2 = mx.concatenate([v[:, :, :horizon], v[:, :, horizon:] + 7.0], axis=2)
    o2 = flash_attention_sparse(q, k, v2, bm, causal=True, auto_pad=True)
    assert mx.array_equal(o[:, :, :rows], o2[:, :, :rows]), "a row saw a future key"


# ── anti-drift lock: the auto_pad predicate == the router's NAX decision ────────────────
@pytest.mark.parametrize("N,D,H,density,causal,extended", [
    (4096, 128, 12, 0.10, False, False), (4096, 128, 4, 0.10, False, False),
    (4096, 128, 1, 0.10, False, False), (8192, 64, 12, 0.20, False, False),
    (4096, 128, 12, 0.50, False, False), (4096, 128, 4, 0.05, True, False),
    (6144, 128, 16, 0.10, False, False), (6144, 128, 16, 0.10, False, True),
    (2048, 64, 4, 0.10, False, True), (4096, 128, 12, 0.95, False, True),
    (4096, 128, 12, 0.10, False, "v1"), (4096, 128, 12, 0.10, False, "scalar_fallback"),
    (4096, 128, 12, 0.35, False, False),           # density ceiling (0.30) as the reason
    (4096, 128, 1, 0.10, True, False),             # causal, rejected by policy
    (1024, 64, 4, 0.10, False, True),              # mask < 4096 B -> never NAX
    (4096, 128, 12, 0.10, False, "bf16"), (4096, 128, 12, 0.10, False, "no_hooks"),
    (4096, 64, 4, 0.10, False, "mask4d"),
])
def test_predicate_matches_router_on_aligned_shapes(monkeypatch, N, D, H, density, causal, extended):
    dtype = mx.float16
    if extended is True:
        pass   # 2.64: the extended opt-in is a no-op (predicate and router both see the default law)
    elif extended in ("v1", "scalar_fallback"):  # kernel override honoured by the router
        monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", extended)
    elif extended == "no_hooks":
        monkeypatch.setenv("MFA_DISABLE_AUTO_HOOKS", "1")
    elif extended == "bf16":
        dtype = mx.bfloat16
    q, k, v, bm = _inputs(N, N, D, H, density, seed=N + H, dtype=dtype)
    if extended == "mask4d":
        bm = mx.broadcast_to(bm, (1, H) + bm.shape)
    with dt.capture() as cap:
        mx.eval(flash_attention_sparse(q, k, v, bm, causal=causal))
    router_nax = any(r[0] == "v6nax_sparse" for r in cap)
    assert att._auto_pad_nax_route(q, k, v, bm, causal) == router_nax, [r[0] for r in cap]


# ── API-02 and its sibling: valid masks are never refused because of auto_pad ─────────
def test_api02_steel_geometry_mask_under_auto_pad():
    """API-02: a D=128 mask at the documented STEEL geometry (32x16) raised under
    auto_pad (the pad changed ceil(S/16)).  It is not 32-granular, so it is never
    padded: auto_pad=True == auto_pad=False."""
    N, D = 4100, 128
    mx.random.seed(8)
    q, k, v = ((mx.random.normal((1, 4, N, D)) * 0.1).astype(mx.float16) for _ in range(3))
    BQ, BK = att._steel_block_config(D)
    bm = mx.ones((-(-N // BQ), -(-N // BK)), dtype=mx.bool_)
    a = flash_attention_sparse(q, k, v, bm, auto_pad=True)
    b = flash_attention_sparse(q, k, v, bm, auto_pad=False)
    mx.eval(a, b)
    assert mx.array_equal(a, b)


@pytest.mark.parametrize("N,S", [(90, 90), (96, 4100), (4100, 4100)])
def test_extended_refusal_reads_ceil_counts(monkeypatch, N, S):
    """The extended path inferred BT as N // NQ: a valid 32-block mask with NQ | N
    (N=90 -> NQ=3 -> "BT=30") was refused.  Ceil-granular masks are accepted; a
    16-block mask is still refused loudly (2.64: by the default path's geometry check —
    the extended opt-in is retired)."""
    D = 64
    q, k, v, bm = _inputs(N, S, D, 2, 0.3, seed=9)
    ref = _oracle(q, k, v, bm, False)
    assert_row_gates(flash_attention_sparse(q, k, v, bm, auto_pad=True), ref, **F16_GATES)
    bm16 = mx.ones((-(-N // 16), -(-S // 16)), dtype=mx.bool_)
    with pytest.raises(ValueError, match="accepted geometry"):
        flash_attention_sparse(q, k, v, bm16)



# ── pre-checkpoint review (Phase B) ────────────────────────────────────────────────────
def test_auto_pad_honours_scalar_kernel_override(monkeypatch):
    """MFA_LCSA_KERNEL_VERSION=v1 makes the router use the scalar kernel; the padded
    route used to force v6nax_sparse anyway.  It is not padded now."""
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "v1")
    q, k, v, bm = _inputs(4100, 4100, 128, 4, 0.1, seed=12)
    with dt.capture() as cap:
        o = flash_attention_sparse(q, k, v, bm, auto_pad=True)
        mx.eval(o)
    assert not _padded_route_ran(cap), [r[:2] for r in cap]
    assert_row_gates(o, _oracle(q, k, v, bm, False), **F16_GATES)


@pytest.mark.parametrize("scale", [-0.1, float("inf")])
def test_auto_pad_never_changes_the_contract_for_odd_scales(monkeypatch, scale):
    """auto_pad=True raised where auto_pad=False computed (the kernel refuses a
    non-positive / non-finite scale): such calls are never padded."""
    q, k, v, bm = _inputs(4100, 4100, 64, 2, 0.1, seed=13)
    try:
        b = flash_attention_sparse(q, k, v, bm, scale=scale, auto_pad=False)
        mx.eval(b)
    except Exception as e:                     # the unpadded route's own contract
        with pytest.raises(type(e)):
            mx.eval(flash_attention_sparse(q, k, v, bm, scale=scale, auto_pad=True))
        return
    a = flash_attention_sparse(q, k, v, bm, scale=scale, auto_pad=True)
    mx.eval(a)
    assert mx.array_equal(mx.isnan(a), mx.isnan(b))
    assert mx.array_equal(mx.where(mx.isnan(a), 0, a), mx.where(mx.isnan(b), 0, b))


def test_raw_kv_valid_len_rejects_negative_values():
    from mlx_mfa import _ext
    q, k, v, bm = _inputs(4128, 4128, 64, 2, 0.1, seed=14)
    with pytest.raises(RuntimeError, match="kv_valid_len"):
        _ext.sparse_attention_forward(q, k, v, bm, block_tile=32, scale=0.125,
                                      kernel_version="v6nax_sparse", kv_valid_len=-5)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16], ids=["f16", "bf16"])
def test_u1_gqa_per_head_mask_forward_and_grad(monkeypatch, causal, dtype):
    """Coverage the review added by hand: GQA 8/2 with a 3-D per-head mask, forward
    and gradients, on the auto_pad kv_valid_len route."""
    N, D, Hq, Hk = 4097, 64, 8, 2
    mx.random.seed(15)
    q = mx.random.normal((1, Hq, N, D)).astype(dtype)
    k = mx.random.normal((1, Hk, N, D)).astype(dtype)
    v = mx.random.normal((1, Hk, N, D)).astype(dtype)
    nb = -(-N // 32)
    bm = (mx.random.uniform(shape=(Hq, nb, nb)) < 0.1) | mx.eye(nb, dtype=mx.bool_)
    mx.eval(q, k, v, bm)
    kr, vr = mx.repeat(k, Hq // Hk, axis=1), mx.repeat(v, Hq // Hk, axis=1)
    with dt.capture() as cap:
        o = flash_attention_sparse(q, k, v, bm, causal=causal, auto_pad=True)
        mx.eval(o)
    # 2.64: non-causal takes the padded route under the default law; causal stays on the
    # SDPA fallback by default (causal policy unchanged) — the padded kernel is then
    # locked directly (it serves the B1 size-guard rescue).
    assert _padded_route_ran(cap) or causal
    if causal:
        from mlx_mfa.attention import _make_sparse_nax_padded_vjp
        ok = _make_sparse_nax_padded_vjp(1.0 / math.sqrt(D), True)(q, k, v, bm)
        tol_k = F16_GATES if dtype == mx.float16 else dict(max_abs=3e-2, norm_tol=2e-2, cos_min=0.999)
        assert_row_gates(ok, _oracle(q, kr, vr, bm, True), **tol_k, label=f"kernel GQA causal {dtype}")
    tol = F16_GATES if dtype == mx.float16 else dict(max_abs=3e-2, norm_tol=2e-2, cos_min=0.999)
    assert_row_gates(o, _oracle(q, kr, vr, bm, causal), **tol, label=f"GQA 3-D {dtype} causal={causal}")
    if dtype == mx.float16:
        mx.random.seed(16)
        g = mx.random.normal(q.shape).astype(mx.float32)
        grads = mx.grad(lambda a, b, c: (flash_attention_sparse(a, b, c, bm, causal=causal, auto_pad=True)
                                         .astype(mx.float32) * g).sum(), argnums=(0, 1, 2))(q, k, v)
        mx.eval(*grads)
        with mx.stream(mx.cpu):
            tok = mx.repeat(mx.repeat(bm, 32, axis=-2), 32, axis=-1)[..., :N, :N]
            if causal:
                tok = tok & (mx.arange(N)[None] <= mx.arange(N)[:, None])

            def f(a, b, c):
                b, c = mx.repeat(b, Hq // Hk, axis=1), mx.repeat(c, Hq // Hk, axis=1)
                s = mx.where(tok, (a @ mx.swapaxes(b, -1, -2)) * D ** -0.5, float("-inf"))
                return ((mx.softmax(s, axis=-1) @ c) * g).sum()
            ref = mx.grad(f, argnums=(0, 1, 2))(*(x.astype(mx.float32) for x in (q, k, v)))
            mx.eval(*ref)
        for name, x, y in zip(("dQ", "dK", "dV"), grads, ref):
            rel = float(mx.max(mx.abs(x.astype(mx.float32) - y))) / float(mx.max(mx.abs(y)))
            assert rel < 1e-2, (name, rel)
