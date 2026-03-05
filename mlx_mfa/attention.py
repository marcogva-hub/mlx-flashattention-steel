"""Flash Attention for MLX using Metal Flash Attention kernels.

Public surface:
    flash_attention(q, k, v, scale, causal, stream)  -- main entry point
    is_mfa_available()                               -- extension health check
    get_device_info()                                -- GPU family detection
    get_supported_configs()                          -- supported (D, dtype) set

Dispatch logic:
    flash_attention → validate inputs
                    → GQA tile if H_kv < H_q
                    → _can_use_mfa?
                      yes → _mfa_forward (STEEL kernel via custom_function)
                      no  → _fallback_sdpa (mx.fast.scaled_dot_product_attention)

Backward:
    _make_mfa_custom registers a custom vjp that re-materialises gradients via
    mx.vjp(_fallback_sdpa), bypassing the ccv C++ vjp path (which loses LSE).
    See _sever_lazy_graph() for the buffer-aliasing fix required in that path.
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import mlx.core as mx

_MFA_SUPPORTED_HDIMS = {64, 128, 256}
_MFA_SUPPORTED_DTYPES = {mx.float16, mx.bfloat16, mx.float32}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flash_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    causal: bool = False,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Compute scaled dot-product attention using Metal Flash Attention.

    Drop-in replacement for ``mx.fast.scaled_dot_product_attention``.

    The function dispatches to the Metal Flash Attention (MFA) kernel when:
    - ``head_dim`` is in ``{64, 128, 256}``
    - ``dtype`` is float16, bfloat16, or float32
    - all of q/k/v have the same ``head_dim``
    - the C++ extension (``mlx_mfa._ext``) is compiled and importable

    Falls back gracefully to ``mx.fast.scaled_dot_product_attention`` when
    any of the above conditions is unmet.

    Args:
        q: Query tensor of shape ``[batch, heads, seq_len, head_dim]``.
        k: Key tensor of shape ``[batch, heads, kv_len, head_dim]``.
        v: Value tensor of shape ``[batch, heads, kv_len, head_dim]``.
        scale: Attention scale factor. Defaults to ``1 / sqrt(head_dim)``.
        causal: Whether to apply causal (autoregressive) masking.
        stream: MLX stream for async execution. Defaults to the default GPU
            stream. Currently only honoured on the fallback path; the MFA
            kernel always uses the default GPU stream.

    Returns:
        Attention output of shape ``[batch, heads, seq_len, head_dim]``,
        in the same dtype as ``q``.

    Raises:
        ValueError: If any input is not a 4-D tensor, or if q/k/v have
            mismatched ``head_dim`` values.

    Example::

        import mlx.core as mx
        from mlx_mfa import flash_attention

        q = mx.random.normal((1, 8, 512, 128))
        k = mx.random.normal((1, 8, 512, 128))
        v = mx.random.normal((1, 8, 512, 128))
        out = flash_attention(q, k, v, causal=True)  # [1, 8, 512, 128]
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"flash_attention expects 4-D tensors [batch, heads, seq, head_dim]."
            f" Got q={q.ndim}D, k={k.ndim}D, v={v.ndim}D."
        )

    q_dim = q.shape[-1]
    k_dim = k.shape[-1]
    v_dim = v.shape[-1]
    if k_dim != q_dim or v_dim != q_dim:
        raise ValueError(
            f"q, k, v must all have the same head_dim. "
            f"Got q_dim={q_dim}, k_dim={k_dim}, v_dim={v_dim}."
        )

    head_dim = q_dim
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # --- Grouped Query Attention (GQA) validation ----------------------------
    # The STEEL kernel supports native GQA: gqa_factor = H_q / H_kv is set in
    # MFASteelParams and the shader maps Q head h → KV head h/gqa_factor.
    # No mx.repeat needed — K/V are passed with their original H_kv heads.
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if kv_heads != q_heads:
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"flash_attention GQA: q_heads ({q_heads}) must be divisible "
                f"by kv_heads ({kv_heads})."
            )

    if not _can_use_mfa(q, head_dim):
        return _fallback_sdpa(q, k, v, scale, causal, stream)

    return _mfa_forward(q, k, v, scale, causal, stream)


def is_mfa_available() -> bool:
    """Return True if the MFA C++ extension is compiled and loadable.

    When this returns False, :func:`flash_attention` silently falls back to
    ``mx.fast.scaled_dot_product_attention``.

    Example::

        from mlx_mfa import is_mfa_available
        if is_mfa_available():
            print("MFA kernel active")
    """
    return _ext_available()


def get_device_info() -> dict:
    """Return Metal GPU hardware information.

    When the C++ extension is not available, returns a dict with ``None``
    values for hardware fields.

    Returns:
        Dictionary with keys:

        - ``"device_name"`` (str | None): MTLDevice name, e.g. ``"Apple M2 Pro"``.
        - ``"gpu_family_gen"`` (int | None): Apple GPU family generation number.
          7 = M1/A15, 8 = M2/A16, 9 = M3/A17, 10 = M4.
        - ``"is_m3_plus"`` (bool | None): True for M3/M4 (uses different block
          params and ``preferAsyncCache`` vs ``preferAsyncLoad``).
        - ``"chip_name"`` (str | None): Inferred chip family, e.g. ``"M2"``.
        - ``"extension_available"`` (bool): Whether the C++ extension loaded.

    Example::

        from mlx_mfa import get_device_info
        info = get_device_info()
        print(info["device_name"])   # "Apple M2 Pro"
        print(info["chip_name"])     # "M2"
        print(info["is_m3_plus"])    # False
    """
    if not _ext_available():
        return {
            "device_name": None,
            "gpu_family_gen": None,
            "is_m3_plus": None,
            "chip_name": None,
            "extension_available": False,
        }

    from mlx_mfa._ext import get_device_info as _ext_get_device_info

    raw = _ext_get_device_info()

    # Map GPU silicon generation number → chip family name.
    #
    # get_architecture_gen() extracts the numeric part from the MLX
    # architecture string (e.g. "applegpu_g13s" → 13):
    #   13 → M1 family  (M1, M1 Pro, M1 Max, M1 Ultra)
    #   14 → M2 family  (M2, M2 Pro, M2 Max, M2 Ultra)
    #   15 → M3 family  (M3, M3 Pro, M3 Max, M3 Ultra)
    #   16 → M4 family  (M4, M4 Pro, M4 Max)
    #   17 → M5 family  (M5, M5 Pro, M5 Max — A19 / TBDR tensor ops)
    #
    # M3+ (gen >= 15) uses preferAsyncCache kernel params instead of
    # preferAsyncLoad, following the ccv blocking-parameter tables.
    # M5+ (gen >= 17) exposes the Metal 4 tensor API (MTLTensor /
    # cooperative tensors on A19+). Stub for future TensorOps kernels.
    _GEN_TO_CHIP = {
        13: "M1",
        14: "M2",
        15: "M3",
        16: "M4",
        17: "M5",
    }
    gen = raw.get("gpu_family_gen")
    chip = _GEN_TO_CHIP.get(gen, f"Apple-g{gen}") if gen is not None else None
    is_m3_plus = (gen >= 15) if gen is not None else None
    is_m5_plus = (gen >= 17) if gen is not None else None

    return {
        "device_name":         raw.get("device_name"),
        "gpu_family_gen":      gen,
        "is_m3_plus":          is_m3_plus,
        "is_m5_plus":          is_m5_plus,
        "chip_name":           chip,
        "extension_available": True,
    }


def get_supported_configs() -> dict:
    """Return the set of (head_dim, dtype) configurations supported by MFA.

    Returns:
        Dictionary with keys:
        - ``"head_dims"``: frozenset of supported integer head dimensions.
        - ``"dtypes"``: frozenset of supported MLX dtype values.
        - ``"extension_available"``: bool — whether the C++ extension loaded.

    Example::

        from mlx_mfa import get_supported_configs
        cfg = get_supported_configs()
        print(cfg["head_dims"])   # frozenset({64, 128, 256})
    """
    return {
        "head_dims": frozenset(_MFA_SUPPORTED_HDIMS),
        "dtypes": frozenset(_MFA_SUPPORTED_DTYPES),
        "extension_available": _ext_available(),
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Block-sparse block size lookup (mirrors select_steel_block_config in C++)
# ---------------------------------------------------------------------------

def _steel_block_config(head_dim: int) -> tuple[int, int]:
    """Return (BQ, BK) for the STEEL kernel at the given head_dim.

    Must stay in sync with select_steel_block_config() in mfa_steel_fwd.cpp.
    """
    if head_dim <= 64:
        return (32, 32)
    elif head_dim <= 128:
        return (32, 16)
    else:  # D=256
        return (32, 16)


# ---------------------------------------------------------------------------
# Block-sparse mask helpers
# ---------------------------------------------------------------------------

def make_causal_block_mask(seq_len: int, head_dim: int = 128) -> mx.array:
    """Block-causal mask: True where the K-block's last token index <= Q-block's first.

    Args:
        seq_len:  Sequence length.
        head_dim: Head dimension (determines BQ, BK tile sizes).

    Returns:
        bool array [NQ_tiles, NK_tiles].  True = compute this block.

    Example::

        mask = make_causal_block_mask(512)
        out = flash_attention_sparse(q, k, v, mask)
    """
    BQ, BK = _steel_block_config(head_dim)
    NQ = (seq_len + BQ - 1) // BQ
    NK = (seq_len + BK - 1) // BK
    rows = mx.arange(NQ, dtype=mx.int32)
    cols = mx.arange(NK, dtype=mx.int32)
    # Block (q, k) is active when the k-block's first token <= q-block's last token
    # i.e.  k_start <= q_end  ↔  k * BK <= (q+1) * BQ - 1
    q_end  = (rows + 1) * BQ - 1          # [NQ]
    k_start = cols * BK                    # [NK]
    mask = k_start[None, :] <= q_end[:, None]  # [NQ, NK]
    return mask


def make_sliding_window_mask(
    seq_len: int,
    window_size: int,
    head_dim: int = 128,
    causal: bool = False,
) -> mx.array:
    """Sliding-window block mask: each Q-block attends to K-blocks within
    +/- ``window_size`` tokens.

    Args:
        seq_len:     Sequence length.
        window_size: Number of tokens on each side of the query block's centre
                     that keys are visible from.
        head_dim:    Head dimension (determines BQ, BK tile sizes).
        causal:      If True, also apply causal masking (no future keys).

    Returns:
        bool array [NQ_tiles, NK_tiles].

    Example::

        # Each token sees 512 past + 512 future tokens
        mask = make_sliding_window_mask(4096, window_size=512)
        out  = flash_attention_sparse(q, k, v, mask)
    """
    BQ, BK = _steel_block_config(head_dim)
    NQ = (seq_len + BQ - 1) // BQ
    NK = (seq_len + BK - 1) // BK
    rows = mx.arange(NQ, dtype=mx.int32)
    cols = mx.arange(NK, dtype=mx.int32)

    q_centre = rows * BQ + BQ // 2   # centre token of Q-block [NQ]
    k_start  = cols * BK              # first token of K-block  [NK]
    k_end    = k_start + BK - 1       # last token of K-block   [NK]

    # K-block overlaps the [q_centre - window, q_centre + window] range
    in_window = (k_end[None, :] >= q_centre[:, None] - window_size) & \
                (k_start[None, :] <= q_centre[:, None] + window_size)

    if causal:
        q_end   = (rows + 1) * BQ - 1
        k_start2 = cols * BK
        in_window = in_window & (k_start2[None, :] <= q_end[:, None])

    return in_window


# ---------------------------------------------------------------------------
# Block-sparse forward
# ---------------------------------------------------------------------------

def flash_attention_sparse(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_mask: mx.array,
    scale: Optional[float] = None,
    causal: bool = False,
    stream: Optional[mx.Stream] = None,
    backward: str = "sdpa",
) -> mx.array:
    """Block-sparse Flash Attention.

    Only computes attention for (Q-tile, K-tile) pairs where
    ``block_mask[q_tile, k_tile] == True``.  Masked-out blocks
    contribute zero weight (equivalent to -inf before softmax).

    Args:
        q:          Query   [B, H, N, D].  f16 or bf16 only.
        k:          Key     [B, H, S, D].
        v:          Value   [B, H, S, D].
        block_mask: Boolean [NQ_tiles, NK_tiles].
                    ``NQ_tiles = ceil(N / BQ)``, ``NK_tiles = ceil(S / BK)``
                    where BQ, BK are from ``_steel_block_config(D)``.
                    Use :func:`make_causal_block_mask` or
                    :func:`make_sliding_window_mask` to generate.
        scale:      Attention scale (default: 1/sqrt(D)).
        causal:     Additional causal masking within the active blocks.
        stream:     Optional MLX stream.

    Returns:
        Output [B, H, N, D].

    Note — Backward pass:
        Gradients are computed via dense SDPA + float block bias, which is
        correct but does not benefit from sparsity (dense backward).
        A native sparse backward is planned for a future release.

    Example::

        mask = make_sliding_window_mask(4096, window_size=512)
        out  = flash_attention_sparse(q, k, v, mask)
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "flash_attention_sparse expects 4-D tensors [batch, heads, seq, head_dim]"
        )
    B, H, N, D = q.shape
    S = k.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if q.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError(
            "flash_attention_sparse requires float16 or bfloat16; "
            f"got {q.dtype}. For float32, use flash_attention() with a float mask."
        )
    if D not in _MFA_SUPPORTED_HDIMS:
        raise ValueError(
            f"flash_attention_sparse: head_dim must be in {_MFA_SUPPORTED_HDIMS}, "
            f"got {D}"
        )
    if block_mask.ndim != 2:
        raise ValueError(
            "block_mask must be 2-D [NQ_tiles, NK_tiles]; "
            f"got shape {list(block_mask.shape)}"
        )

    BQ, BK = _steel_block_config(D)
    NQ_expected = (N + BQ - 1) // BQ
    NK_expected = (S + BK - 1) // BK
    if block_mask.shape[0] != NQ_expected or block_mask.shape[1] != NK_expected:
        raise ValueError(
            f"block_mask shape {list(block_mask.shape)} does not match "
            f"expected [{NQ_expected}, {NK_expected}] "
            f"for seq_len={N}/{S}, head_dim={D} (BQ={BQ}, BK={BK})"
        )

    if not _ext_available():
        return _sparse_fallback_sdpa(q, k, v, block_mask, BQ, BK, scale, causal)

    impl = _make_mfa_sparse_custom(scale, causal, block_mask,
                                    head_dim=D, backward=backward)
    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    mask_uint8 = block_mask.astype(mx.uint8)
    mask_uint8 = mx.contiguous(mask_uint8)
    # _impl returns (O, L); public API returns only O.
    O, _L = impl(q, k, v, mask_uint8)
    return O


def _make_mfa_sparse_custom(
    scale: float,
    causal: bool,
    block_mask: mx.array,
    head_dim: int = 128,
    backward: str = "sdpa",
):
    """Build a custom_function wrapping the sparse STEEL kernel.

    The forward returns (O, L) where L is the logsumexp [B, H, N, float32].
    L is in log2 domain (STEEL convention: L = log2(e) * L_natural).

    backward options:
        "sdpa"         (default) — dense mx.fast.sdpa vjp; correct but O(N×S×D)
        "sdpa_sparse"  — tiled Python sparse backward using saved L;
                         O(nnz × BQ × BK × D), benefits large sparse configs
    """
    import numpy as _np

    BQ, BK = _steel_block_config(head_dim)

    @mx.custom_function
    def _impl(q, k, v, mask_uint8):
        # Returns (O, L) — L saved for backward via `output` parameter.
        from mlx_mfa._ext import mfa_attention_sparse_forward_with_lse as _fwd
        O, L = _fwd(q, k, v, mask_uint8, scale, causal)
        return O, L

    @_impl.vjp
    def _backward(primals, cotangents, outputs):
        q, k, v, mask_uint8 = primals
        dO, _dL = cotangents  # dL is zero (L not consumed downstream)
        O, L    = outputs

        if backward == "sdpa_sparse":
            # Tiled sparse backward using saved L — skips inactive tiles.
            D = q.shape[-1]
            bq, bk = _steel_block_config(D)
            block_mask_np = _np.array(block_mask.astype(mx.uint8))
            dQ, dK, dV = _sparse_backward_tiled(
                q, k, v, O, L, dO, block_mask_np, bq, bk, scale, causal
            )
            return dQ, dK, dV, mx.zeros_like(mask_uint8)

        # Dense SDPA backward (correct, no sparsity speedup).
        float_mask = _block_mask_to_float_bias(
            block_mask, q.shape[2], k.shape[2], scale_q_dtype=q.dtype
        )
        if causal:
            N, S = q.shape[2], k.shape[2]
            causal_m = mx.triu(
                mx.full((N, S), float("-inf"), dtype=q.dtype), k=S - N + 1
            )
            float_mask = float_mask + causal_m
        _, (dQ, dK, dV) = mx.vjp(
            lambda q, k, v: mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=float_mask
            ),
            [q, k, v],
            [dO],
        )
        return dQ, dK, dV, mx.zeros_like(mask_uint8)

    return _impl


def _block_mask_to_float_bias(
    block_mask: mx.array,
    seq_q: int,
    seq_k: int,
    scale_q_dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Expand a bool block_mask [NQ, NK] to a float additive bias [N, S].

    True  → 0.0     (include in attention)
    False → -inf    (mask out)
    """
    BQ, BK = block_mask.shape[0], block_mask.shape[1]
    # Create a full float mask of shape [NQ*BQ, NK*BK] then slice to [N, S]
    # block_mask is [NQ, NK] → repeat each element BQ/BK times
    # Expand: [NQ, 1, NK, 1] → [NQ, BQ, NK, BK] → [NQ*BQ, NK*BK]
    D = seq_q // block_mask.shape[0]   # BQ (approximate)
    BQ_actual = (seq_q + block_mask.shape[0] - 1) // block_mask.shape[0]
    BK_actual = (seq_k + block_mask.shape[1] - 1) // block_mask.shape[1]

    # float: True→0, False→-inf
    float_block = mx.where(block_mask, mx.array(0.0), mx.array(float("-inf")))
    # Repeat each block element to cover BQ query rows and BK key cols
    # Shape: [NQ, NK] → [NQ, 1, NK, 1] → [NQ, BQ, NK, BK] → [NQ*BQ, NK*BK]
    float_block = float_block[:, None, :, None]  # [NQ, 1, NK, 1]
    float_block = mx.broadcast_to(
        float_block,
        (block_mask.shape[0], BQ_actual, block_mask.shape[1], BK_actual)
    )
    # Reshape [NQ, BQ, NK, BK] → [NQ*BQ, NK*BK] via transpose + reshape
    NQ, _, NK, _ = float_block.shape
    float_block = mx.transpose(float_block, (0, 1, 2, 3))  # keep shape
    float_block = float_block.reshape(NQ * BQ_actual, NK * BK_actual)
    # Slice to actual [seq_q, seq_k]
    float_bias = float_block[:seq_q, :seq_k]
    return float_bias.astype(scale_q_dtype)


def _sparse_fallback_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_mask: mx.array,
    BQ: int,
    BK: int,
    scale: float,
    causal: bool,
) -> mx.array:
    """Dense SDPA fallback for flash_attention_sparse (used when C++ ext absent)."""
    N, S = q.shape[2], k.shape[2]
    float_bias = _block_mask_to_float_bias(block_mask, N, S, q.dtype)
    if causal:
        causal_m = mx.triu(
            mx.full((N, S), float("-inf"), dtype=q.dtype), k=S - N + 1
        )
        float_bias = float_bias + causal_m
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=float_bias
    )


# ---------------------------------------------------------------------------
# Sparse tiled backward (G.2-G.5)
# ---------------------------------------------------------------------------

def _sparse_backward_tiled(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    O: mx.array,
    L: mx.array,
    dO: mx.array,
    block_mask_np,      # numpy bool [NQ, NK] — pre-evaluated for Python loops
    BQ: int,
    BK: int,
    scale: float,
    causal: bool,
) -> tuple:
    """Tiled sparse backward using saved logsumexp L.

    Skips inactive tiles for O(nnz × BQ × BK × D) work vs O(N × S × D)
    for dense SDPA.  L is in log2 domain (STEEL kernel output convention):
        P_ij = exp2(scale_log2 * QK^T_ij - L_i)
             = exp(scale * QK^T_ij - L_natural_i)
    where L_natural = L_log2 * ln(2).

    Handles GQA (H_q != H_kv): K/V head index = q_head // gqa_factor.

    Returns (dQ, dK, dV) in the input dtype of q / k / v.
    """
    import math as _math
    import numpy as _np

    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    S = k.shape[2]
    NQ, NK = block_mask_np.shape

    LN2 = _math.log(2)  # convert L_log2 → L_natural: L_natural = L_log2 * ln(2)

    # D_scalar[b, h, i] = sum_d(dO[b,h,i,d] * O[b,h,i,d]) — query-row delta
    D_scalar = mx.sum(dO.astype(mx.float32) * O.astype(mx.float32), axis=-1)  # [B, H_q, N]

    # ── dQ: accumulate per Q-tile ───────────────────────────────────────────
    dQ_tiles = []
    for qi in range(NQ):
        qi_s = qi * BQ
        qi_e = min(qi_s + BQ, N)
        Q_qi  = q[:, :, qi_s:qi_e, :].astype(mx.float32)   # [B, H_q, bq, D]
        dO_qi = dO[:, :, qi_s:qi_e, :].astype(mx.float32)  # [B, H_q, bq, D]
        L_qi  = L[:, :, qi_s:qi_e] * LN2                    # [B, H_q, bq] natural log
        D_qi  = D_scalar[:, :, qi_s:qi_e]                   # [B, H_q, bq]

        contribs = []
        for kj in range(NK):
            if not block_mask_np[qi, kj]:
                continue
            kj_s = kj * BK
            kj_e = min(kj_s + BK, S)
            # GQA: select the KV-head for each Q-head
            # K has shape [B, H_kv, S, D]; broadcast over H_q via repeat
            # Use direct indexing for efficiency (no reshape needed):
            #   head_kv_idx = q_head_idx // gqa_factor — handled by taking
            #   K slice [B, H_kv, kj_s:kj_e, D] and repeating to H_q
            K_kj = k[:, :, kj_s:kj_e, :].astype(mx.float32)   # [B, H_kv, bk, D]
            # Expand H_kv → H_q if GQA
            if H_kv != H_q:
                ratio = H_q // H_kv
                K_kj = mx.repeat(K_kj, ratio, axis=1)          # [B, H_q, bk, D]

            # S_tile [B, H_q, bq, bk]
            S_tile = scale * mx.matmul(Q_qi, K_kj.swapaxes(-1, -2))
            if causal:
                row_ids = qi_s + mx.arange(qi_e - qi_s)         # [bq]
                col_ids = kj_s + mx.arange(kj_e - kj_s)         # [bk]
                causal_mask = col_ids[None, :] > row_ids[:, None]  # [bq, bk] bool
                S_tile = mx.where(causal_mask, mx.array(float("-inf")), S_tile)

            P_tile = mx.exp(S_tile - L_qi[:, :, :, None])       # [B, H_q, bq, bk]

            # dP [B, H_q, bq, bk]
            # need V_kj [B, H_q, bk, D]
            V_kj = v[:, :, kj_s:kj_e, :].astype(mx.float32)
            if H_kv != H_q:
                V_kj = mx.repeat(V_kj, ratio, axis=1)
            dP_tile = mx.matmul(dO_qi, V_kj.swapaxes(-1, -2))

            dS_tile = P_tile * (dP_tile - D_qi[:, :, :, None])  # [B, H_q, bq, bk]

            # dQ contribution: scale * dS @ K  [B, H_q, bq, D]
            contribs.append(scale * mx.matmul(dS_tile, K_kj))

        if contribs:
            dQ_qi = sum(contribs[1:], contribs[0]).astype(q.dtype)
        else:
            dQ_qi = mx.zeros((B, H_q, qi_e - qi_s, D), dtype=q.dtype)
        dQ_tiles.append(dQ_qi)

    dQ = mx.concatenate(dQ_tiles, axis=2)  # [B, H_q, N, D]

    # ── dK, dV: accumulate per K-tile ───────────────────────────────────────
    dK_tiles, dV_tiles = [], []
    for kj in range(NK):
        kj_s = kj * BK
        kj_e = min(kj_s + BK, S)
        K_kj = k[:, :, kj_s:kj_e, :].astype(mx.float32)   # [B, H_kv, bk, D]
        V_kj = v[:, :, kj_s:kj_e, :].astype(mx.float32)   # [B, H_kv, bk, D]

        dk_contribs, dv_contribs = [], []
        for qi in range(NQ):
            if not block_mask_np[qi, kj]:
                continue
            qi_s = qi * BQ
            qi_e = min(qi_s + BQ, N)
            Q_qi  = q[:, :, qi_s:qi_e, :].astype(mx.float32)   # [B, H_q, bq, D]
            dO_qi = dO[:, :, qi_s:qi_e, :].astype(mx.float32)
            L_qi  = L[:, :, qi_s:qi_e] * LN2                    # [B, H_q, bq]
            D_qi  = D_scalar[:, :, qi_s:qi_e]                   # [B, H_q, bq]

            # Expand K_kj/V_kj to H_q for GQA
            if H_kv != H_q:
                ratio = H_q // H_kv
                K_kj_h = mx.repeat(K_kj, ratio, axis=1)
                V_kj_h = mx.repeat(V_kj, ratio, axis=1)
            else:
                K_kj_h, V_kj_h = K_kj, V_kj

            S_tile = scale * mx.matmul(Q_qi, K_kj_h.swapaxes(-1, -2))
            if causal:
                row_ids = qi_s + mx.arange(qi_e - qi_s)
                col_ids = kj_s + mx.arange(kj_e - kj_s)
                causal_mask = col_ids[None, :] > row_ids[:, None]
                S_tile = mx.where(causal_mask, mx.array(float("-inf")), S_tile)

            P_tile  = mx.exp(S_tile - L_qi[:, :, :, None])      # [B, H_q, bq, bk]
            dP_tile = mx.matmul(dO_qi, V_kj_h.swapaxes(-1, -2)) # [B, H_q, bq, bk]
            dS_tile = P_tile * (dP_tile - D_qi[:, :, :, None])  # [B, H_q, bq, bk]

            # dV: P^T @ dO → [B, H_q, bk, D]; sum over H_q groups for GQA
            dV_contrib = mx.matmul(P_tile.swapaxes(-1, -2), dO_qi)  # [B, H_q, bk, D]
            # dK: scale * dS^T @ Q → [B, H_q, bk, D]; sum over H_q groups for GQA
            dK_contrib = scale * mx.matmul(dS_tile.swapaxes(-1, -2), Q_qi)  # [B, H_q, bk, D]

            if H_kv != H_q:
                # Collapse H_q → H_kv: sum over groups of (ratio) heads
                ratio = H_q // H_kv
                dV_contrib = dV_contrib.reshape(B, H_kv, ratio, kj_e - kj_s, D)
                dV_contrib = mx.sum(dV_contrib, axis=2)          # [B, H_kv, bk, D]
                dK_contrib = dK_contrib.reshape(B, H_kv, ratio, kj_e - kj_s, D)
                dK_contrib = mx.sum(dK_contrib, axis=2)          # [B, H_kv, bk, D]

            dv_contribs.append(dV_contrib)
            dk_contribs.append(dK_contrib)

        if dk_contribs:
            dK_kj = sum(dk_contribs[1:], dk_contribs[0]).astype(k.dtype)
            dV_kj = sum(dv_contribs[1:], dv_contribs[0]).astype(v.dtype)
        else:
            dK_kj = mx.zeros((B, H_kv, kj_e - kj_s, D), dtype=k.dtype)
            dV_kj = mx.zeros((B, H_kv, kj_e - kj_s, D), dtype=v.dtype)
        dK_tiles.append(dK_kj)
        dV_tiles.append(dV_kj)

    dK = mx.concatenate(dK_tiles, axis=2)  # [B, H_kv, S, D]
    dV = mx.concatenate(dV_tiles, axis=2)  # [B, H_kv, S, D]

    return dQ, dK, dV


# Internal helpers
# ---------------------------------------------------------------------------


def _can_use_mfa(q: mx.array, head_dim: int) -> bool:
    """Return True iff the MFA kernel can be dispatched for these inputs."""
    if head_dim not in _MFA_SUPPORTED_HDIMS:
        return False
    if q.dtype not in _MFA_SUPPORTED_DTYPES:
        return False
    if not _ext_available():
        return False
    return True


def _ext_available() -> bool:
    """Return True iff the C++ extension module is importable."""
    try:
        from mlx_mfa._ext import mfa_attention_forward  # noqa: F401
        return True
    except ImportError:
        return False


def _sever_lazy_graph(arr: mx.array) -> mx.array:
    """Return a copy of *arr* with no lazy-graph ancestry.

    **Why this is needed — buffer aliasing in Metal:**

    Inside a ``mx.custom_function`` vjp, the ``cotangent`` argument is often
    ``ones_like(O_fwd)`` — a lazy node that inherits the same buffer-ancestry
    as the first forward pass output ``O_fwd``.  When the backward then calls
    ``mfa_forward_with_lse`` a second time (gradient checkpointing), MLX may
    schedule both forward dispatches in the *same* Metal command encoder.  The
    Metal allocator can then alias ``O_r``'s output buffer with the freed
    ``O_fwd`` buffer; since ``L_r`` is written alongside ``O_r`` in one atomic
    kernel dispatch, this corrupts ``L_r`` and produces wrong or overflowed
    gradients.

    **The fix:**  ``arr + mx.zeros_like(arr)`` routes through an elementwise-
    add kernel that writes to a *fresh, independent* output buffer.  This new
    buffer has no shared ancestry with ``O_fwd``, so the allocator cannot
    alias it with ``O_r`` — the second forward runs cleanly.

    **Alternatives tested (Phase 4.1.1):**

    +-----------------------------------------+--------+
    | Approach                                | Works? |
    +=========================================+========+
    | ``arr + mx.zeros_like(arr)``            | ✓      |
    | numpy round-trip (f32 cast)             | ✓      |
    | ``mx.contiguous(arr)`` (after eval)     | ✓      |
    | ``mx.array(arr)``                       | ✗      |
    | ``mx.stop_gradient(arr)``               | ✗      |
    +-----------------------------------------+--------+

    The pure-MLX add is preferred: no CPU round-trip, no bfloat16 numpy issue.
    """
    return arr + mx.zeros_like(arr)


@functools.lru_cache(maxsize=32)
def _make_mfa_custom(scale: float, causal: bool):
    """Return a custom-vjp MFA forward function for the given (scale, causal).

    ``lru_cache`` ensures the same Python function object (with its registered
    backward) is reused for identical hyperparameters, avoiding repeated
    ``mx.custom_function`` decoration overhead.

    Design note — why not use the C++ Primitive vjp?
    ─────────────────────────────────────────────────
    ``mfa_attention_forward`` returns only ``outputs[0]`` (O).  MLX's autograd
    therefore prunes ``outputs[1]`` (L / logsumexp) from the computation graph.
    When MLX later calls ``MFAttention::vjp(..., outputs)``, ``outputs`` has
    size 1.  Accessing ``outputs[1]`` in C++ is undefined behaviour and returns
    garbage, corrupting every P / dS / dQ computation.

    The Python ``custom_function`` completely bypasses that path.  The backward
    re-materialises O by re-running the SDPA fallback, then uses MLX's
    native SDPA backward via ``mx.vjp``.  This is simpler and more
    maintainable than the ccv backward kernels while producing identical
    gradients.
    """
    from mlx_mfa._ext import mfa_forward_with_lse

    @mx.custom_function
    def _impl(q, k, v):
        # Forward: STEEL kernel.  L is not stored (only O is needed).
        O, _ = mfa_forward_with_lse(q, k, v, scale, causal)
        return O

    @_impl.vjp
    def _backward(primals, cotangent, output):
        # mx.custom_function vjp signature:
        #   primals   - tuple of all forward inputs (q, k, v)
        #   cotangent - gradient w.r.t. the output O  (i.e. dO)
        #   output    - forward output O (unused; gradients are computed fresh)
        q, k, v = primals

        # Use MLX's native SDPA backward via mx.vjp.
        # mx.vjp(fn, primals, cotangents) returns (out, grads) where
        # grads are the VJP of fn at primals with cotangents.
        _, (dQ, dK, dV) = mx.vjp(
            lambda q, k, v: _fallback_sdpa(q, k, v, scale, causal),
            [q, k, v],
            [cotangent],
        )
        return dQ, dK, dV

    return _impl


def _mfa_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Dispatch through the MFA custom-vjp path.

    Ensures inputs are contiguous before passing to the Metal kernel.
    The ``stream`` argument is accepted for API compatibility but the
    custom-vjp path always uses the default GPU stream.
    """
    # Metal kernels require BHND row-major layout (leading dim = D).
    # mx.contiguous() is a no-op when the array is already contiguous.
    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    impl = _make_mfa_custom(scale, causal)
    return impl(q, k, v)


def _fallback_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Fallback to ``mx.fast.scaled_dot_product_attention``."""
    mask = None
    if causal:
        N, S = q.shape[2], k.shape[2]
        mask = mx.triu(
            mx.full((N, S), float("-inf"), dtype=q.dtype),
            k=S - N + 1,
        )
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask,
    )
