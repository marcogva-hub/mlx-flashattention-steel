"""RC 2.63.0 hardening of the sparse extended opt-in + SLA (maintainer decisions D1-D5,
2026-09-28, from the Phase B pre-checkpoint review).

D1  sla_attention(extended=None) = the extended path on M5+ only — it raised on pre-M5
    with the default arguments (auto-default principle).
D2  MFA_SPARSE_NAX_EXTENDED uses the strict 0/1 parser (was: "true"/"yes"/"on" enabled,
    "2" silently disabled); an invalid MFA_SPARSE_D_DENSE_CUTOFF is refused (was: silent
    0.85; nan / negative accepted).
D3  a non-32-block mask under the opt-in is refused with a message pointing to the
    default (non-extended) path, which accepts e.g. the 32x16 STEEL geometry.
D4  the opt-in is a context-local override (contextvar), not a process-wide os.environ
    mutation; extended=False forces it OFF even when the env var is set.
D5  both entry points (flash_attention_sparse, sparse_attention_dispatch) share the opt-in
    rules: refusals, 64-block masks expanded exactly to 32, masks below the kernel's
    4096-byte minimum and near-dense masks take the dense masked route (traced).
"""
from __future__ import annotations

import os
import threading

import mlx.core as mx
import pytest

import mlx_mfa.attention as att
import mlx_mfa.lcsa_nax as lcsa
from mlx_mfa import _dispatch_trace as dt
from mlx_mfa import is_mfa_available
from mlx_mfa import sla as S
from mlx_mfa.attention import flash_attention_sparse
from mlx_mfa.lcsa_nax import sparse_attention_dispatch
from tests.sparse_gates import assert_row_gates

_M5 = is_mfa_available() and att._get_is_m5_plus_cached()
m5only = pytest.mark.skipif(not _M5, reason="requires M5+ (V6NAX sparse) and the MFA extension")
G16 = dict(max_abs=1e-2, norm_tol=5e-3)


def _qkv(N, D, H=2, seed=0, dtype=mx.float16):
    mx.random.seed(seed)
    q, k, v = (mx.random.normal((1, H, N, D)).astype(dtype) for _ in range(3))
    mx.eval(q, k, v)
    return q, k, v


def _mask(nq, nk, density, seed=1):
    mx.random.seed(seed)
    m = (mx.random.uniform(shape=(nq, nk)) < density) | mx.eye(nq, nk, dtype=mx.bool_)
    mx.eval(m)
    return m


def _oracle(q, k, v, bm, bt):
    N, S_, D = q.shape[2], k.shape[2], q.shape[3]
    with mx.stream(mx.cpu):
        tok = mx.repeat(mx.repeat(bm, bt, axis=-2), bt, axis=-1)[..., :N, :S_]
        s = mx.where(tok, (q.astype(mx.float32) @ mx.swapaxes(k.astype(mx.float32), -1, -2))
                     * D ** -0.5, float("-inf"))
        o = mx.softmax(s, axis=-1) @ v.astype(mx.float32)
        o = mx.where(mx.any(tok, axis=-1, keepdims=True), o, 0.0)
        mx.eval(o)
    return o


# ── D2: strict parsing ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw", ["true", "yes", "on", " 1 ", "2", "enable"])
def test_extended_env_is_strict_bool(monkeypatch, raw):
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", raw)
    with pytest.raises(ValueError, match="must be '0' or '1'"):
        lcsa._sparse_extended_enabled()


@pytest.mark.parametrize("raw,want", [("1", True), ("0", False)])
def test_extended_env_valid_values(monkeypatch, raw, want):
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", raw)
    assert lcsa._sparse_extended_enabled() is want


@pytest.mark.parametrize("raw", ["abc", "0,85", "nan", "-1", "0", "inf"])
def test_dense_cutoff_invalid_is_refused(monkeypatch, raw):
    monkeypatch.setenv("MFA_SPARSE_D_DENSE_CUTOFF", raw)
    with pytest.raises(ValueError, match="MFA_SPARSE_D_DENSE_CUTOFF"):
        lcsa._d_dense_cutoff()


@pytest.mark.parametrize("raw,want", [("0.5", 0.5), ("1.01", 1.01)])
def test_dense_cutoff_valid(monkeypatch, raw, want):
    monkeypatch.setenv("MFA_SPARSE_D_DENSE_CUTOFF", raw)
    assert lcsa._d_dense_cutoff() == want


def test_new_bool_knob_registered():
    from mlx_mfa import _knobs
    assert "MFA_SPARSE_NAX_EXTENDED" in _knobs.BOOL_KNOBS


# ── D4: context-local override ─────────────────────────────────────────────────────────
def test_override_is_context_local_and_forces_off(monkeypatch):
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    with lcsa._extended_override(False):
        assert lcsa._sparse_extended_enabled() is False          # forces OFF over the env
    monkeypatch.delenv("MFA_SPARSE_NAX_EXTENDED")
    seen = {}
    with lcsa._extended_override(True):
        assert lcsa._sparse_extended_enabled() is True
        t = threading.Thread(target=lambda: seen.update(other=lcsa._sparse_extended_enabled()))
        t.start(); t.join()
        assert "MFA_SPARSE_NAX_EXTENDED" not in os.environ       # no process-wide mutation
    assert seen["other"] is False                                # other threads unaffected
    assert lcsa._sparse_extended_enabled() is False


# ── D1 / D4 through sla_attention ──────────────────────────────────────────────────────
@m5only
def test_sla_default_is_extended_on_m5(monkeypatch):
    q, k, v = _qkv(2048, 64, H=2)                                # B·H=2: outside the policy
    with dt.capture() as cap:
        mx.eval(S.sla_attention(q, k, v, topk_ratio=0.1))
    assert any(r[0] == "v6nax_sparse" for r in cap)              # extended route on M5


def _sla_reference(q, k, v, ratio, blkq=128, blkk=64):
    """CPU fp32 SLA (proj_l = identity) with the selection sla_attention itself makes —
    independent of the attention kernels."""
    L, D = q.shape[2], q.shape[3]
    sm = S._sla_block_map(q, k, ratio, blkq, blkk)
    with mx.stream(mx.cpu):
        q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
        tok = mx.repeat(mx.repeat(sm, blkq, axis=-2), blkk, axis=-1)[..., :L, :L]
        s_ = mx.where(tok, (q32 @ mx.swapaxes(k32, -1, -2)) * D ** -0.5, float("-inf"))
        o_s = mx.where(mx.any(tok, axis=-1, keepdims=True), mx.softmax(s_, axis=-1) @ v32, 0.0)
        fq, fk = mx.softmax(q32, axis=-1), mx.softmax(k32, axis=-1)
        o_l = (fq @ (mx.swapaxes(fk, -1, -2) @ v32)) / (
            mx.sum(fq * mx.sum(fk, axis=2, keepdims=True), axis=-1, keepdims=True) + 1e-5)
        o = o_s + o_l
        mx.eval(o)
    return o


@pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")
@pytest.mark.parametrize("L,ratio", [(2048, 0.1), (2000, 0.1), (256, 0.5), (2048, 0.001)])
def test_sla_default_on_pre_m5(monkeypatch, L, ratio):
    """D1 on pre-M5 — runs on ANY chip (a genuine pre-M5 run on M1-M4 CI; simulated on
    M5): the default must not raise and must match a CPU fp32 SLA reference.  ratio 0.001
    selects NO block (topk = 0): the sparse term is exactly 0, as in the reference
    (it went through the pre-M5 sparse kernel, whose empty rows gave NaN in simulation)."""
    monkeypatch.setattr(att, "_get_is_m5_plus_cached", lambda: False)
    q, k, v = _qkv(L, 64, H=2, seed=L)
    o = S.sla_attention(q, k, v, topk_ratio=ratio)
    mx.eval(o)
    assert_row_gates(o, _sla_reference(q, k, v, ratio), **G16, label=f"pre-M5 sla L={L}")


@m5only
def test_sla_extended_false_forces_off(monkeypatch):
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    q, k, v = _qkv(2048, 64, H=2)
    with dt.capture() as cap:
        mx.eval(S.sla_attention(q, k, v, topk_ratio=0.1, extended=False))
    assert not any(r[0] == "v6nax_sparse" for r in cap), [r[:2] for r in cap]


# ── D3: refusal message ────────────────────────────────────────────────────────────────
@m5only
def test_non_32_mask_refused_with_pointer_to_default_path(monkeypatch):
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    N, D = 4096, 128
    q, k, v = _qkv(N, D)
    BQ, BK = att._steel_block_config(D)
    bm = mx.ones((-(-N // BQ), -(-N // BK)), dtype=mx.bool_)    # documented 32x16 geometry
    with pytest.raises(ValueError, match="MFA_SPARSE_NAX_EXTENDED=0|default path"):
        flash_attention_sparse(q, k, v, bm)
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "0")
    mx.eval(flash_attention_sparse(q, k, v, bm))                 # the default path accepts it


# ── D5: both entry points share the rules ──────────────────────────────────────────────
@m5only
@pytest.mark.parametrize("entry", ["flash_attention_sparse", "sparse_attention_dispatch"])
def test_bt64_mask_expanded_under_extended(monkeypatch, entry):
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    N, D = 4096, 128
    q, k, v = _qkv(N, D, seed=2)
    bm64 = _mask(N // 64, N // 64, 0.1)
    fn = (lambda: flash_attention_sparse(q, k, v, bm64)) if entry == "flash_attention_sparse" \
        else (lambda: sparse_attention_dispatch(q, k, v, bm64, block_tile=64))
    with dt.capture() as cap:
        o = fn()
        mx.eval(o)
    assert any(r[0] == "v6nax_sparse" for r in cap), [r[:2] for r in cap]
    assert_row_gates(o, _oracle(q, k, v, bm64, 64), **G16, label=entry)


@m5only
@pytest.mark.parametrize("entry", ["flash_attention_sparse", "sparse_attention_dispatch"])
def test_dense_cutoff_applies_to_both_entries(monkeypatch, entry):
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    N, D = 4096, 128
    q, k, v = _qkv(N, D, seed=3)
    bm = mx.ones((N // 32, N // 32), dtype=mx.bool_)            # density 1.0 >= cutoff 0.85
    fn = (lambda: flash_attention_sparse(q, k, v, bm)) if entry == "flash_attention_sparse" \
        else (lambda: sparse_attention_dispatch(q, k, v, bm, block_tile=32))
    with dt.capture() as cap:
        o = fn()
        mx.eval(o)
    assert not any(r[0] == "v6nax_sparse" for r in cap), [r[:2] for r in cap]
    assert_row_gates(o, _oracle(q, k, v, bm, 32), **G16, label=entry)


@m5only
@pytest.mark.parametrize("entry", ["flash_attention_sparse", "sparse_attention_dispatch"])
def test_small_mask_takes_dense_route_under_extended(monkeypatch, entry):
    """Masks below the kernel's 4096-byte minimum: dense masked route, traced — not a
    silent STEEL fallback (flash_attention_sparse) nor a C++ raise (dispatcher)."""
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    N, D = 1024, 64                                              # 32 x 32 = 1024 B mask
    q, k, v = _qkv(N, D, seed=4)
    bm = _mask(N // 32, N // 32, 0.2)
    fn = (lambda: flash_attention_sparse(q, k, v, bm)) if entry == "flash_attention_sparse" \
        else (lambda: sparse_attention_dispatch(q, k, v, bm, block_tile=32))
    with dt.capture() as cap:
        o = fn()
        mx.eval(o)
    assert any(r[0] == "sdpa" and "4096" in r[1] for r in cap), [r[:2] for r in cap]
    assert_row_gates(o, _oracle(q, k, v, bm, 32), **G16, label=entry)


@m5only
@pytest.mark.parametrize("bad", ["D256", "BT16"])
def test_dispatcher_refusals_under_extended(monkeypatch, bad):
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    N = 4096
    if bad == "D256":
        q, k, v = _qkv(N, 256)
        bm, bt = _mask(N // 32, N // 32, 0.1), 32
        match = "head_dim must be 64 or 128"
    else:
        q, k, v = _qkv(N, 128)
        bm, bt = _mask(N // 16, N // 16, 0.1), 16
        match = "block tile must be 32"
    with pytest.raises(ValueError, match=match):
        sparse_attention_dispatch(q, k, v, bm, block_tile=bt)


# ── pre-RC review of dbd81f2 (siblings of D5) ─────────────────────────────────────────
@m5only
def test_flashvsr_bt16_unaffected_by_global_opt_in(monkeypatch):
    """The FlashVSR integration uses 16-block masks by design; with the opt-in set
    process-wide its dispatcher calls used to fall back to SDPA and, after D5, raised.
    16/non-32/64-block integration calls are scoped out of the opt-in (D4 override)."""
    import mlx.nn as nn
    from mlx_mfa.integrations.flashvsr_lcsa import patch_flashvsr_lcsa

    class _Attn(nn.Module):
        def __call__(self, Q, K, V):
            return mx.fast.scaled_dot_product_attention(Q, K, V, scale=Q.shape[-1] ** -0.5)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attn()

    N, D = 4096, 128
    q, k, v = _qkv(N, D, H=4, seed=6)
    bm16 = _mask(N // 16, N // 16, 0.02)
    model = _Model()
    model.attn.lcsa_block_mask = bm16
    model.attn.lcsa_block_tile = 16
    patch_flashvsr_lcsa(model, verbose=False)
    ref = model.attn(q, k, v)
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    out = model.attn(q, k, v)
    mx.eval(ref, out)
    assert mx.array_equal(out, ref)


@m5only
def test_small_mask_route_keeps_mask_shape_validation(monkeypatch):
    """The extended small-mask dense route returned before the 3-D/4-D head/batch
    checks: a [1, NQ, NK] mask with H=2 was silently broadcast (the default path
    raises).  Both paths now raise."""
    N, D = 1024, 64
    q, k, v = _qkv(N, D, H=2, seed=7)
    bm = _mask(N // 32, N // 32, 0.2)[None]                     # 3-D, shape[0]=1 != H=2
    with pytest.raises(ValueError, match="must equal H"):
        flash_attention_sparse(q, k, v, bm)
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    with pytest.raises(ValueError, match="must equal H"):
        flash_attention_sparse(q, k, v, bm)
