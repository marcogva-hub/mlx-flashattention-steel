"""Sparse-Linear Attention (SLA) — MLX composition (Volet B).

Faithful port of the thu-ml/TurboDiffusion SLA SEMANTICS (SOT @ 73df7e1,
`turbodiffusion/SLA/{core,utils}.py`) composed over the Volet A extended sparse
path. The mechanism (source-cited) is in `devnotes/sla_mechanism.md`:

    o = o_sparse + proj_l(o_linear)                              (core.py:114)
      o_sparse : block-softmax over the top-k K-blocks per Q-row (selection at
                 BLKQ×BLKK, expanded to BT=32) — flash_attention_sparse (Volet A)
      o_linear : linear attention over ALL keys, softmax feature map (core.py:104-110)
      proj_l   : learned Linear(D,D) from the checkpoint (None → identity, for tests)

Selection (utils.py:55-67): smooth-k (`k - mean_L(k)`), mean-pool q/k over
BLKQ/BLKK, RAW pooled QK^T (no softmax/scale), top-k PER Q-block-row. Production
`attention_type="sla"` uses BLKQ=128, BLKK=64 (asymmetric); the *default* is 64/64.

Scope: NON-CAUSAL MHA (the thu-ml semantics — video DiT is bidirectional); GQA via
KV-repeat. `causal=True` raises (causal linear attention needs a chunked scan, not
the O(N)-parallel form thu-ml uses — deferred, no SOT reference). Off-path intact:
nothing here runs unless `sla_attention` is called. fp16/bf16 inputs; the block
selection is computed in fp32 and is deterministic for a given device, but NOT bit-exact
against other implementations: the GPU fp32 matmul is close to TF32 precision, so
near-tied top-k scores can resolve differently from a CPU/fp64 reference (measured:
0.006 % of blocks at L=73 899 on unit-variance inputs; 0 at the tested 128/64 shapes).
The linear-term reductions run in fp32 (API-05).
Differentiable w.r.t. q/k/v (review 2026-09, NEPB-05): the top-k selection is
piecewise constant, so gradients flow through the sparse and linear terms only.
"""
from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx

from mlx_mfa.attention import flash_attention_sparse

BT = 32  # V6NAX sparse block tile (Volet A)


# --------------------------------------------------------------------------- helpers
def _mean_pool(x: mx.array, blk: int) -> mx.array:
    """Block-mean over the sequence axis (L = axis 2) → (B, H, ceil(L/blk), D).

    Matches thu-ml `mean_pool` (utils.py:43-52), ragged tail included."""
    B, H, L, D = x.shape
    nb = (L + blk - 1) // blk
    full = L // blk
    parts = []
    if full > 0:
        parts.append(x[:, :, : full * blk, :].reshape(B, H, full, blk, D).mean(axis=3))
    if full < nb:
        parts.append(x[:, :, full * blk :, :].mean(axis=2, keepdims=True))
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)


def _sla_block_map(q: mx.array, k: mx.array, topk_ratio: float, blkq: int, blkk: int) -> mx.array:
    """Top-k block selection (utils.py:55-67), computed in fp32 for a deterministic
    block set. Returns a per-(B,H) bool mask (B, H, NQ, NK), top-k K-blocks per Q-row."""
    qf = q.astype(mx.float32)
    kf = k.astype(mx.float32)
    arg_k = kf - mx.mean(kf, axis=2, keepdims=True)              # smooth-k (utils.py:56)
    pq = _mean_pool(qf, blkq)                                    # (B,H,NQ,D)
    pk = _mean_pool(arg_k, blkk)                                 # (B,H,NK,D)
    score = pq @ pk.swapaxes(-1, -2)                            # raw pooled QK^T (utils.py:59)
    NK = score.shape[-1]
    topk = min(NK, int(topk_ratio * NK))
    if topk <= 0:
        return mx.zeros(score.shape, dtype=mx.bool_)
    if topk >= NK:
        return mx.ones(score.shape, dtype=mx.bool_)
    # per-row top-k via dense descending ranks (rank < topk keeps the top-k columns)
    order = mx.argsort(-score, axis=-1)
    ranks = mx.argsort(order, axis=-1)
    return (ranks < topk).astype(mx.bool_)


def _expand_to_bt32(sparse_map: mx.array, blkq: int, blkk: int, nq32: int, nk32: int) -> mx.array:
    """Expand a (B,H,NQ,NK) selection at BLKQ×BLKK to the BT=32 grid, exactly
    (each block → (blkq/32)×(blkk/32) identical BT32 blocks), sliced to (nq32, nk32)."""
    rq, ck = blkq // BT, blkk // BT
    bm = mx.repeat(mx.repeat(sparse_map, rq, axis=-2), ck, axis=-1)
    return bm[..., :nq32, :nk32]


_FEATURE_MAPS = {
    "softmax": lambda x: mx.softmax(x, axis=-1),
    "elu": lambda x: (mx.where(x > 0, x, mx.exp(x) - 1.0) + 1.0),
    "relu": lambda x: mx.maximum(x, 0),
}


def _linear_term(q: mx.array, k: mx.array, v: mx.array, feature_map: str) -> mx.array:
    """Linear attention over ALL keys (core.py:104-110), softmax feature map (tied q/k).

    API-05 (review 2026-09): the L-reductions (kv = phi(k)^T v, ksum) ran in the INPUT
    dtype and overflowed fp16 (inf, no error; threshold ~1/L — at Wan L=144288 a value
    mean ~0.46 suffices).  They run in fp32 now (the thu-ml SOT uses bf16); the result
    is returned in the input dtype, so ``proj_l`` sees the same dtype as before.
    """
    fm = _FEATURE_MAPS[feature_map]
    q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
    fq = fm(q32)
    fk = fm(k32)
    kv = fk.swapaxes(-1, -2) @ v32                              # (B,H,D,Dv), contract L
    ksum = mx.sum(fk, axis=2, keepdims=True)                    # (B,H,1,D)
    num = fq @ kv                                               # (B,H,L,Dv)
    den = mx.sum(fq * ksum, axis=-1, keepdims=True) + 1e-5      # (B,H,L,1)
    return (num / den).astype(q.dtype)


def _repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    if n_rep == 1:
        return x
    B, Hk, L, D = x.shape
    return mx.broadcast_to(x[:, :, None], (B, Hk, n_rep, L, D)).reshape(B, Hk * n_rep, L, D)


# --------------------------------------------------------------------------- public API
def sla_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    topk_ratio: float,
    blkq: int = 128,
    blkk: int = 64,
    feature_map: str = "softmax",
    proj_l: Optional[Callable[[mx.array], mx.array]] = None,
    scale: Optional[float] = None,
    causal: bool = False,
    extended: Optional[bool] = None,
) -> mx.array:
    """Sparse-Linear Attention (thu-ml SLA semantics) over the Volet A sparse path.

    Args:
        q, k, v: (B, H, L, D) fp16/bf16. GQA allowed (Hk = Hq // n_rep); K/V are
                 repeated to Hq. L need not be block-aligned (auto_pad handles it).
        topk_ratio: fraction of K-blocks kept per Q-block-row (`sla_topk`, e.g. 0.1).
        blkq, blkk: SELECTION block sizes (production "sla" = 128, 64). Must be
                    multiples of 32.
        feature_map: linear-term feature map — 'softmax' (thu-ml default), 'elu', 'relu'.
        proj_l: learned projection applied to the linear term (the checkpoint's
                Linear(D,D)); None → identity (untrained/test).
        scale: sparse-term softmax scale (default 1/sqrt(D)).
        causal: NOT supported (thu-ml SLA is non-causal); raises.
        extended: run the sparse term on the Volet A extended path, so the block-skip
                  is reachable at B·H/N beyond the default gate.  None (default) = on
                  M5+ only (the extended path refuses pre-M5 chips); True forces it on
                  (raises pre-M5); False forces it off, even if MFA_SPARSE_NAX_EXTENDED=1.
                  The choice is context-local (no os.environ mutation).

    Returns:
        (B, H, L, D) = o_sparse + proj_l(o_linear).
    """
    if causal:
        raise NotImplementedError(
            "sla_attention: causal is not supported — thu-ml SLA is non-causal "
            "(bidirectional video DiT); causal linear attention needs a chunked-scan "
            "formulation outside the O(N)-parallel form. Deferred (no SOT reference).")
    if q.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError(f"sla_attention requires float16/bfloat16; got {q.dtype}")
    if blkq % BT or blkk % BT:
        raise ValueError(f"blkq/blkk must be multiples of {BT}; got {blkq}/{blkk}")
    B, H, L, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    n_rep = H // k.shape[1]
    kf = _repeat_kv(k, n_rep)
    vf = _repeat_kv(v, n_rep)

    # 1-2. selection at BLKQ×BLKK → expand to the BT32 grid the sparse kernel wants.
    sparse_map = _sla_block_map(q, kf, topk_ratio, blkq, blkk)
    nq32 = (L + BT - 1) // BT
    bm32 = _expand_to_bt32(sparse_map, blkq, blkk, nq32, nq32)

    # 3. sparse term — the Volet A extended path (block-skip, auto_pad).
    from mlx_mfa.attention import _get_is_m5_plus_cached
    from mlx_mfa.lcsa_nax import _extended_override
    use_extended = _get_is_m5_plus_cached() if extended is None else bool(extended)
    n_blocks_k = -(-L // blkk)
    if min(n_blocks_k, int(topk_ratio * n_blocks_k)) <= 0:
        # top-k selects NO block: the sparse term is exactly 0 (the reference's
        # nan_to_num contract).  Never sent to a sparse kernel — on the pre-M5 route
        # (opened by extended=None) all-empty rows came back NaN (RC review, D1).
        o_s = mx.zeros(q.shape[:-1] + (vf.shape[-1],), dtype=q.dtype)
    else:
        with _extended_override(use_extended):
            o_s = flash_attention_sparse(q, kf, vf, bm32, scale=scale, causal=False,
                                         auto_pad=True)

    # 4-5. linear term over ALL keys + learned projection.
    o_l = _linear_term(q, kf, vf, feature_map)
    if proj_l is not None:
        o_l = proj_l(o_l)

    # 6. recombination — plain sum (core.py:114).
    return (o_s + o_l).astype(q.dtype)
