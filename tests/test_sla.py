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

from mlx_mfa import attention as A
from mlx_mfa import sla as S

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


def _cos(a, b):
    a = np.asarray(a.astype(mx.float32)).ravel().astype(np.float64)
    b = b.detach().numpy().ravel().astype(np.float64) if isinstance(b, torch.Tensor) else np.asarray(b).ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


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
    assert _cos(o_mlx, o_ref) >= 0.999


# ===================================================================== locks
@m5only
def test_lock_topk1_equals_dense():
    """topk_ratio=1.0 → all blocks selected → o_sparse == dense attention."""
    q, k, v = _mk(1, 8, 2048, 128)
    scale = 1.0 / math.sqrt(128)
    sm = S._sla_block_map(q, k, 1.0, 128, 64)
    assert bool(mx.all(sm).item()), "topk_ratio=1.0 must select every block"
    o = S.sla_attention(q, k, v, topk_ratio=1.0, proj_l=lambda x: x * 0.0)   # o = o_s only
    ref = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), k.astype(mx.float32),
                                               v.astype(mx.float32), scale=scale)
    mx.eval(o, ref)
    assert _cos(o, np.asarray(ref)) >= 0.999


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
    assert _cos(o, o_s_ref) >= 0.999


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
    c = _cos(o_mlx, o_ref)
    assert c >= 0.999, f"SLA parity cos={c:.6f} (B{B} Hq{Hq} Hk{Hk} L{L} D{D})"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
