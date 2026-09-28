"""Volet B — SLA composition tests: component units + locks + parity vs a
torch-CPU fp32 reference derived LINE-BY-LINE from thu-ml/TurboDiffusion SLA
(SOT @ 73df7e1, SLA/{core,utils}.py). See devnotes/sla_mechanism.md.

The reference is a deliverable (a test tool), not a throwaway. Skips off M5+/NAX.
"""
import math
import numpy as np
import pytest
import mlx.core as mx
import torch
import torch.nn.functional as F

from mlx_mfa import _dispatch_trace as dt
from mlx_mfa import attention as A
from mlx_mfa import sla as S
from tests.sparse_gates import assert_row_gates

_M5 = bool(getattr(A, "_get_is_m5_plus_cached", lambda: False)())
m5only = pytest.mark.skipif(not _M5, reason="SLA sparse term is M5+/NAX only")
BT = 32


# ============================================================ torch-CPU fp32 reference
def _ref_mean_pool(x, blk):                                      # utils.py:43-52
    B, H, L, D = x.shape
    nb = (L + blk - 1) // blk
    full = L // blk
    parts = []
    if full > 0:
        parts.append(x[:, :, : full * blk, :].reshape(B, H, full, blk, D).mean(dim=3))
    if full < nb:
        parts.append(x[:, :, full * blk :, :].mean(dim=2, keepdim=True))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)


def _ref_block_map(q, k, topk_ratio, blkq, blkk):               # utils.py:55-67
    arg_k = k - k.mean(dim=-2, keepdim=True)                     # smooth-k (:56)
    pq = _ref_mean_pool(q, blkq)
    pk = _ref_mean_pool(arg_k, blkk)
    score = pq @ pk.transpose(-1, -2)                           # raw pooled QK^T (:59)
    K = score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    sm = torch.zeros_like(score, dtype=torch.int8)
    if topk > 0:
        lut = torch.topk(score, topk, dim=-1, sorted=False).indices  # per-row top-k (:63)
        sm.scatter_(-1, lut, 1)
    return sm.bool()


def _ref_linear(q, k, v, feature_map="softmax"):                # core.py:104-110
    fm = {"softmax": lambda x: torch.softmax(x, dim=-1),
          "elu": lambda x: F.elu(x) + 1, "relu": torch.relu}[feature_map]
    fq, fk = fm(q), fm(k)
    kvsum = fk.transpose(-1, -2) @ v
    ksum = fk.sum(dim=-2, keepdim=True)
    return (fq @ kvsum) / (1e-5 + (fq * ksum).sum(dim=-1, keepdim=True))


def _ref_sparse(q, k, v, sm, blkq, blkk, scale):                # kernel.py block-softmax
    B, H, L, D = q.shape
    em = sm.repeat_interleave(blkq, dim=-2).repeat_interleave(blkk, dim=-1)[..., :L, :L]
    s = (q @ k.transpose(-1, -2)) * scale
    s = s.masked_fill(~em, float("-inf"))
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)                            # all-False row → zero (v2.34 contract)
    return p @ v


def _ref_sla(q, k, v, topk_ratio, blkq, blkk, feature_map, Wl, bl, scale):  # core.py:114
    sm = _ref_block_map(q, k, topk_ratio, blkq, blkk)
    o_s = _ref_sparse(q, k, v, sm, blkq, blkk, scale)
    o_l = _ref_linear(q, k, v, feature_map)
    if Wl is not None:
        o_l = o_l @ Wl.transpose(-1, -2) + bl
    return o_s + o_l, sm, o_s, o_l


# ------------------------------------------------------------------------- utilities
def _mk(B, H, L, D, dt=mx.float16, seed=0):
    mx.random.seed(seed)
    f = lambda: (mx.random.normal((B, H, L, D)) * 0.1).astype(dt)
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    return q, k, v


def _to_t(a):
    return torch.from_numpy(np.asarray(a.astype(mx.float32))).to(torch.float32)


def _mx(t):
    """torch fp32 reference -> mx fp32 (for the shared per-row gates)."""
    return mx.array(t.detach().numpy()) if isinstance(t, torch.Tensor) else t


# Review 2026-09 (DSP-14 / TST-04 / TST-10, remediation B1): the gates here were a
# GLOBAL cosine >= 0.999 — magnitude-blind (U1 scaled whole rows, cos stayed >= 0.999).
# Per-row magnitude gates now (tests/sparse_gates.py); cosine is only a complement.
_G = dict(max_abs=1e-2, norm_tol=1e-2, cos_min=0.999)


# ===================================================================== component units
def test_component_selection_matches_ref():
    """Selection block-set is EXACTLY the thu-ml top-k (both fp32)."""
    q, k, v = _mk(1, 8, 4096, 128)
    sm_mlx = S._sla_block_map(q, k, 0.1, 128, 64)
    mx.eval(sm_mlx)
    sm_ref = _ref_block_map(_to_t(q), _to_t(k), 0.1, 128, 64)
    a = np.asarray(sm_mlx).astype(bool)
    b = sm_ref.numpy().astype(bool)
    assert a.shape == b.shape, f"{a.shape} vs {b.shape}"
    div = int((a != b).sum())
    assert div == 0, f"selection block-set diverged in {div}/{a.size} blocks"


def test_component_linear_matches_ref():
    """Linear term (softmax feature map, over ALL keys) matches the reference."""
    q, k, v = _mk(1, 8, 2048, 128)
    o_mlx = S._linear_term(q, k, v, "softmax")
    mx.eval(o_mlx)
    o_ref = _ref_linear(_to_t(q), _to_t(k), _to_t(v), "softmax")
    assert_row_gates(o_mlx, _mx(o_ref), **_G, label="linear term")


# ===================================================================== locks
@m5only
@pytest.mark.parametrize("L", [2048, 2050])
def test_lock_topk1_equals_dense(monkeypatch, L):
    """topk_ratio=1.0 → all blocks selected → o_sparse == dense attention.

    TST-10 (review 2026-09): at density 1.0 the D_DENSE_CUTOFF sent this to the dense
    SDPA route, so the lock compared SDPA with SDPA.  The cutoff is lifted to force the
    V6NAX block-skip kernel (asserted by trace), against an independent CPU fp32 dense
    reference; L=2050 also exercises the auto_pad kv_valid_len route (U1)."""
    monkeypatch.setenv("MFA_SPARSE_D_DENSE_CUTOFF", "1.01")
    q, k, v = _mk(1, 8, L, 128)
    scale = 1.0 / math.sqrt(128)
    sm = S._sla_block_map(q, k, 1.0, 128, 64)
    assert bool(mx.all(sm).item()), "topk_ratio=1.0 must select every block"
    with dt.capture() as cap:
        o = S.sla_attention(q, k, v, topk_ratio=1.0, proj_l=lambda x: x * 0.0)   # o = o_s only
        mx.eval(o)
    assert any(r[0] == "v6nax_sparse" for r in cap), [r[0] for r in cap]
    with mx.stream(mx.cpu):
        ref = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k.astype(mx.float32),
                                                   v.astype(mx.float32), scale=scale)
        mx.eval(ref)
    assert_row_gates(o, ref, **_G, label=f"topk=1 L={L}")


@m5only
def test_lock_projl_zero_equals_sparse():
    """proj_l ≡ 0 → o == o_sparse (the sum reduces correctly)."""
    q, k, v = _mk(1, 8, 2048, 128)
    scale = 1.0 / math.sqrt(128)
    o = S.sla_attention(q, k, v, topk_ratio=0.1, proj_l=lambda x: x * 0.0)
    # independent o_sparse via the ref path with the SAME selection
    sm = _ref_block_map(_to_t(q), _to_t(k), 0.1, 128, 64)
    o_s_ref = _ref_sparse(_to_t(q), _to_t(k), _to_t(v), sm, 128, 64, scale)
    mx.eval(o)
    assert_row_gates(o, _mx(o_s_ref), **_G, label="proj_l=0 -> o_sparse")


def test_causal_raises():
    q, k, v = _mk(1, 4, 256, 64)
    with pytest.raises(NotImplementedError, match="non-causal"):
        S.sla_attention(q, k, v, topk_ratio=0.1, causal=True)


# ===================================================================== parity (Phase 3)
@m5only
@pytest.mark.parametrize("B,Hq,Hk,L,D,seed", [
    (1, 8, 8, 2048, 128, 0),      # MHA
    (1, 8, 8, 4096, 128, 1),      # MHA, larger N
    (1, 8, 8, 2048, 64, 2),       # D=64
    (1, 8, 2, 2048, 128, 3),      # GQA (n_rep=4)
    (2, 4, 4, 2048, 128, 4),      # batch 2
])
def test_parity_mlx_vs_cpu_ref(B, Hq, Hk, L, D, seed):
    """Full SLA parity: per-component cos + block-set exactness vs the fp32 ref."""
    q, k, v = _mk(B, Hq, L, D, seed=seed)
    kk = (mx.random.normal((B, Hk, L, D)) * 0.1).astype(mx.float16)
    vv = (mx.random.normal((B, Hk, L, D)) * 0.1).astype(mx.float16)
    mx.eval(kk, vv)
    k, v = kk, vv
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(100 + seed)
    Wl = (mx.random.normal((D, D)) * 0.05).astype(mx.float16)
    bl = (mx.random.normal((D,)) * 0.01).astype(mx.float16)
    mx.eval(Wl, bl)
    projl = lambda x: x @ Wl.swapaxes(-1, -2) + bl

    o_mlx = S.sla_attention(q, k, v, topk_ratio=0.1, blkq=128, blkk=64, proj_l=projl, scale=scale)
    mx.eval(o_mlx)

    # ref on GQA-repeated K/V (SLA is MHA; GQA = repeat then MHA)
    n_rep = Hq // Hk
    kt = _to_t(k).repeat_interleave(n_rep, dim=1)
    vt = _to_t(v).repeat_interleave(n_rep, dim=1)
    o_ref, sm_ref, o_s_ref, o_l_ref = _ref_sla(
        _to_t(q), kt, vt, 0.1, 128, 64, "softmax", _to_t(Wl), _to_t(bl), scale)

    # block-set exactness
    sm_mlx = S._sla_block_map(q, S._repeat_kv(k, n_rep), 0.1, 128, 64)
    mx.eval(sm_mlx)
    div = int((np.asarray(sm_mlx).astype(bool) != sm_ref.numpy().astype(bool)).sum())
    assert div == 0, f"selection diverged in {div} blocks (shape {sm_ref.shape})"
    # final output parity
    assert_row_gates(o_mlx, _mx(o_ref), **_G, label=f"SLA parity B{B} Hq{Hq} Hk{Hk} L{L} D{D}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))


# ===================================================================== TST-07 (review 2026-09)
@m5only
@pytest.mark.parametrize("L,ratio", [(130, 0.5), (4099, 0.1), (73899, 0.1)])
def test_sla_nonaligned_L_sampled_rows(L, ratio):
    """sla_attention promises "L need not be block-aligned (auto_pad handles it)", but no
    test used a non-aligned L — and the sparse term inherited U1 (tail rows shrunk).
    Sparse term isolated (proj_l = 0); exact fp32 rows from the torch reference for the
    tail query block (where U1 hit), the first rows and random rows — a full [L, L]
    oracle at L=73 899 is ~22 GB, a row sample is exact and cheap."""
    B, H, D = 1, 2, 128
    q, k, v = _mk(B, H, L, D, seed=L)
    scale = 1.0 / math.sqrt(D)
    # ratio >= 1/NK so at least one key block is selected (L=130: 3 key blocks)
    o = S.sla_attention(q, k, v, topk_ratio=ratio, proj_l=lambda x: x * 0.0, scale=scale)
    mx.eval(o)
    qt, kt, vt = _to_t(q), _to_t(k), _to_t(v)
    sm = _ref_block_map(qt, kt, ratio, 128, 64)
    assert bool(sm.any()), "degenerate selection (no key block)"                         # [B, H, NQ, NK]
    rng = np.random.default_rng(L)
    rows = sorted(set(range(max(0, L - 128), L)) | set(range(min(32, L)))
                  | set(rng.integers(0, L, size=min(32, L)).tolist()))
    ref_rows = []
    for i in rows:
        keep = sm[:, :, i // 128, :].repeat_interleave(64, dim=-1)[..., :L]   # [B, H, L]
        s = (qt[:, :, i:i + 1, :] @ kt.transpose(-1, -2)).squeeze(-2) * scale
        p = torch.nan_to_num(torch.softmax(s.masked_fill(~keep, float("-inf")), -1), nan=0.0)
        ref_rows.append((p.unsqueeze(-2) @ vt).squeeze(-2))
    ref = mx.array(torch.stack(ref_rows, dim=2).numpy())               # [B, H, R, D]
    got = o[:, :, mx.array(rows), :]
    assert_row_gates(got, ref, **_G, label=f"sla L={L} ({len(rows)} rows)")
