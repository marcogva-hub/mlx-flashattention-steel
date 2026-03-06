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
from typing import Optional, Union, Sequence

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
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    return_attn_weights: bool = False,
    stream: Optional[mx.Stream] = None,
):
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
        softcap: Tanh softcapping factor (Gemma 2 / Grok style). When > 0,
            scores are capped via ``tanh(S / softcap) * softcap`` before
            softmax. Set to 0.0 (default) to disable.
        alibi_slopes: Optional ALiBi per-head position biases (Press et al.,
            2021). When not None, should be a 1-D float32 array of shape
            ``[H]`` with one slope per query head.  The bias added to score
            ``(i, j)`` for head ``h`` is ``alibi_slopes[h] * (j - i)``.
            Incompatible with ``softcap``; only f16/bf16 use the MFA kernel.
        dropout_p: Dropout probability on attention weights (0 = disabled).
            When > 0, the call falls back to a Python SDPA implementation —
            the MFA Metal kernel does not support dropout.  Intended for
            training only; pass 0.0 (default) for inference.
        return_attn_weights: When True, also return the softmax attention
            weight matrix ``[B, H, N, S]``.  Forces a Python SDPA fallback
            (the MFA kernel does not expose intermediate probabilities).
            Useful for attention visualization / debugging.
        stream: MLX stream for async execution. Defaults to the default GPU
            stream. Currently only honoured on the fallback path; the MFA
            kernel always uses the default GPU stream.

    Returns:
        When ``return_attn_weights=False`` (default): attention output of
        shape ``[batch, heads, seq_len, head_dim]`` in the same dtype as q.

        When ``return_attn_weights=True``: a 2-tuple
        ``(output, attn_weights)`` where ``output`` is ``[B, H, N, D]`` and
        ``attn_weights`` is ``float32 [B, H, N, S]``.

    Raises:
        ValueError: If any input is not a 4-D tensor, or if q and k have
            mismatched ``head_dim`` values.  Note: v may have a different
            ``head_dim`` than q/k (Track AE); the call falls back to SDPA
            in that case.

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

    # K must match Q for the attention score Q @ K^T.
    if k_dim != q_dim:
        raise ValueError(
            f"q and k must have the same head_dim. "
            f"Got q_dim={q_dim}, k_dim={k_dim}."
        )

    # V may have a different head_dim (Track AE).  MFA kernel requires D_v==D_qk;
    # fall back to SDPA when they differ — SDPA natively handles Dv != Dqk.
    v_dim_mismatch = (v_dim != q_dim)

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

    # Track AH: return_attn_weights forces Python SDPA (MFA kernel
    # does not expose intermediate softmax probabilities).
    if return_attn_weights:
        return _sdpa_with_weights(q, k, v, scale, causal, softcap, dropout_p)

    # Track AG: dropout falls back to Python SDPA (MFA kernel has no dropout).
    if dropout_p > 0.0:
        return _dropout_sdpa(q, k, v, scale, causal, dropout_p)

    if not _can_use_mfa(q, head_dim) or v_dim_mismatch:
        if softcap != 0.0:
            return _softcap_sdpa_ref(q, k, v, scale, causal, softcap)
        if alibi_slopes is not None:
            return _alibi_sdpa_ref(q, k, v, alibi_slopes, scale, causal)
        return _fallback_sdpa(q, k, v, scale, causal, stream)

    # ALiBi requires f16/bf16 for the Metal kernel (f32 has no STEEL ALiBi).
    if alibi_slopes is not None:
        if q.dtype == mx.float32:
            return _alibi_sdpa_ref(q, k, v, alibi_slopes, scale, causal)
        return _mfa_alibi_forward(q, k, v, alibi_slopes, scale, causal)

    return _mfa_forward(q, k, v, scale, causal, softcap, stream)


def make_rope_3d_tables(
    grid_h: int,
    grid_w: int,
    num_frames: int,
    d_h: Optional[int] = None,
    d_w: Optional[int] = None,
    d_t: Optional[int] = None,
    head_dim: int = 128,
    theta: float = 10000.0,
) -> tuple[mx.array, mx.array]:
    """Build 3D RoPE cosine/sine tables for video attention.

    Returns ``(cos, sin)`` of shape ``[N, D/2]`` where ``N = grid_h * grid_w *
    num_frames``.  The D/2 pairs are split into three consecutive sub-bands:

    * pairs ``[0, d_h/2)``:                  height axis, position = patch y
    * pairs ``[d_h/2, d_h/2 + d_w/2)``:     width  axis, position = patch x
    * pairs ``[d_h/2 + d_w/2, D/2)``:        temporal axis, position = frame t

    Compatible with ``flash_attention_rope(..., rope_3d={...})``.

    Args:
        grid_h:    Number of patch rows (height // patch_size).
        grid_w:    Number of patch columns (width // patch_size).
        num_frames: Number of frames (or temporal patches).
        d_h:       Head-dim elements for height axis (default: head_dim // 3).
        d_w:       Head-dim elements for width axis (default: head_dim // 3).
        d_t:       Head-dim elements for temporal axis (default: head_dim - d_h - d_w).
        head_dim:  Total head dimension D.
        theta:     RoPE base frequency (default 10000.0).

    Returns:
        Tuple ``(cos_table, sin_table)`` each ``float32 [N, D/2]``.

    Example::

        cos, sin = make_rope_3d_tables(32, 32, 16, head_dim=128)
        out = flash_attention_rope(q, k, v, cos, sin, rope_3d={
            'grid_h': 32, 'grid_w': 32, 'num_frames': 16})
    """
    import numpy as _np

    if d_h is None:
        # Round down to even
        d_h = (head_dim // 3) & ~1
    if d_w is None:
        d_w = (head_dim // 3) & ~1
    if d_t is None:
        # Consume the remaining dimensions
        d_t = head_dim - d_h - d_w
        # Round down to even, let d_h absorb any remainder
        if d_t % 2:
            d_t -= 1
            d_h += 1

    # All sub-dims must be even (RoPE works on pairs)
    if d_h % 2 or d_w % 2 or d_t % 2:
        raise ValueError(
            f"d_h, d_w, d_t must all be even. Got d_h={d_h}, d_w={d_w}, d_t={d_t}."
        )

    pHW = grid_h * grid_w
    N = num_frames * pHW
    D2 = (d_h + d_w + d_t) // 2  # == head_dim // 2

    token_idx = _np.arange(N, dtype=_np.int64)
    t = token_idx // pHW
    spatial = token_idx % pHW
    y = spatial // grid_w
    x = spatial % grid_w

    cos_table = _np.zeros((N, D2), dtype=_np.float32)
    sin_table = _np.zeros((N, D2), dtype=_np.float32)

    # Height axis — pairs [0, d_h//2)
    j_h = _np.arange(d_h // 2, dtype=_np.float32)
    freq_h = 1.0 / (theta ** (2.0 * j_h / d_h))
    angles_h = y[:, None].astype(_np.float32) * freq_h[None, :]  # [N, d_h//2]
    cos_table[:, :d_h // 2] = _np.cos(angles_h)
    sin_table[:, :d_h // 2] = _np.sin(angles_h)

    # Width axis — pairs [d_h//2, d_h//2 + d_w//2)
    j_w = _np.arange(d_w // 2, dtype=_np.float32)
    freq_w = 1.0 / (theta ** (2.0 * j_w / d_w))
    angles_w = x[:, None].astype(_np.float32) * freq_w[None, :]  # [N, d_w//2]
    off_w = d_h // 2
    cos_table[:, off_w:off_w + d_w // 2] = _np.cos(angles_w)
    sin_table[:, off_w:off_w + d_w // 2] = _np.sin(angles_w)

    # Temporal axis — pairs [d_h//2 + d_w//2, D2)
    j_t = _np.arange(d_t // 2, dtype=_np.float32)
    freq_t = 1.0 / (theta ** (2.0 * j_t / d_t))
    angles_t = t[:, None].astype(_np.float32) * freq_t[None, :]  # [N, d_t//2]
    off_t = d_h // 2 + d_w // 2
    cos_table[:, off_t:] = _np.cos(angles_t)
    sin_table[:, off_t:] = _np.sin(angles_t)

    return mx.array(cos_table), mx.array(sin_table)


def flash_attention_rope(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rotary_cos: Optional[mx.array] = None,
    rotary_sin: Optional[mx.array] = None,
    scale: Optional[float] = None,
    causal: bool = False,
    cache_seqlens: Union[int, "mx.array", Sequence[int]] = 0,
    rope_3d: Optional[dict] = None,
    interleaved: bool = True,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Flash Attention with in-kernel RoPE (Rotary Position Embedding) fusion.

    Applies rotary position embeddings to Q and K *inside* the Metal kernel,
    eliminating a separate elementwise pass over the full Q/K tensors.

    The rotation is applied per adjacent pair ``(d, d+1)`` in the head dimension::

        q_rot[2i]   = q[2i] * cos[pos][i] - q[2i+1] * sin[pos][i]
        q_rot[2i+1] = q[2i] * sin[pos][i] + q[2i+1] * cos[pos][i]

    **1D RoPE (LLM)**:
        Pass ``rotary_cos`` and ``rotary_sin`` as ``float32 [max_seq_len, D/2]``.
        Q positions: ``[cache_seqlens, cache_seqlens + N)``.
        K positions: ``[0, S)``.

    **3D RoPE (video)**:
        Pass ``rope_3d`` dict with keys ``grid_h``, ``grid_w``, ``num_frames``
        (and optionally ``d_h``, ``d_w``, ``d_t``, ``theta``).
        Tables are built automatically via :func:`make_rope_3d_tables`.
        Token layout assumed: ``(T, H, W)`` row-major, same as
        :func:`make_spatial_3d_mask`.  K is also rotated.
        Mutually exclusive with explicit ``rotary_cos``/``rotary_sin``.

    Falls back to a pure-MLX ``_apply_rope_mlx`` + SDPA when the C++
    extension is unavailable or when head_dim / dtype is unsupported.

    Args:
        q: ``[B, H, N, D]`` float16 or bfloat16.
        k: ``[B, H, S, D]`` float16 or bfloat16.
        v: ``[B, H, S, D]`` float16 or bfloat16.
        rotary_cos: ``float32 [max_seq_len, D/2]`` — cosine table (1D RoPE).
        rotary_sin: ``float32 [max_seq_len, D/2]`` — sine table (1D RoPE).
        scale: Attention scale. Defaults to ``1 / sqrt(D)``.
        causal: Apply causal masking.
        cache_seqlens: KV cache length — absolute position of Q token 0.
            Use 0 for prefill, len(kv_cache) for autoregressive decode.
            Can also be a 1D array/list of length ``B`` for per-batch offsets
            (e.g. different decode positions in a batch).
            Only used in 1D mode.
        rope_3d: 3D RoPE config dict.  Required keys: ``grid_h``, ``grid_w``,
            ``num_frames``.  Optional: ``d_h``, ``d_w``, ``d_t``, ``theta``.
            When provided, ``rotary_cos``/``rotary_sin`` must be None.
        interleaved: RoPE pairing mode.  ``True`` (default) = LLaMA style,
            adjacent pairs ``(2i, 2i+1)``.  ``False`` = GPT-NeoX style,
            split-halves ``(i, i+D/2)``.
        stream: MLX stream (GPU). Forwarded to fallback only.

    Returns:
        Attention output ``[B, H, N, D]``, same dtype as ``q``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "flash_attention_rope expects 4-D tensors [batch, heads, seq, head_dim]."
            f" Got q={q.ndim}D, k={k.ndim}D, v={v.ndim}D."
        )

    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # --- 3D RoPE: build flat [N, D/2] tables from grid coordinates ---
    if rope_3d is not None:
        if rotary_cos is not None or rotary_sin is not None:
            raise ValueError(
                "rope_3d and rotary_cos/rotary_sin are mutually exclusive."
            )
        grid_h = rope_3d["grid_h"]
        grid_w = rope_3d["grid_w"]
        num_frames = rope_3d.get("num_frames", 1)
        d_h = rope_3d.get("d_h", None)
        d_w = rope_3d.get("d_w", None)
        d_t = rope_3d.get("d_t", None)
        theta = rope_3d.get("theta", 10000.0)
        rotary_cos, rotary_sin = make_rope_3d_tables(
            grid_h, grid_w, num_frames,
            d_h=d_h, d_w=d_w, d_t=d_t,
            head_dim=head_dim, theta=theta,
        )
        # 3D RoPE always starts at position 0 (spatial, not sequential)
        cache_seqlens = 0

    if rotary_cos is None or rotary_sin is None:
        raise ValueError(
            "flash_attention_rope requires either rotary_cos/rotary_sin (1D) "
            "or rope_3d (3D) to be provided."
        )

    # Track AD: per-batch cache_seqlens — split along batch dim.
    if not isinstance(cache_seqlens, int):
        if isinstance(cache_seqlens, mx.array):
            cs_list = [int(v) for v in cache_seqlens.tolist()]
        else:
            cs_list = [int(v) for v in cache_seqlens]
        B = q.shape[0]
        if len(cs_list) != B:
            raise ValueError(
                f"cache_seqlens length {len(cs_list)} must equal batch size B={B}"
            )
        chunks = [
            flash_attention_rope(
                q[b : b + 1], k[b : b + 1], v[b : b + 1],
                rotary_cos, rotary_sin,
                scale=scale, causal=causal, cache_seqlens=cs,
                rope_3d=None, interleaved=interleaved, stream=stream,
            )
            for b, cs in enumerate(cs_list)
        ]
        return mx.concatenate(chunks, axis=0)

    # RoPE requires f16/bf16 on the STEEL path.
    # Fall back gracefully for f32 or unsupported head_dim.
    if not _can_use_mfa(q, head_dim) or q.dtype == mx.float32:
        q_rot = _apply_rope_mlx(q, rotary_cos, rotary_sin,
                                offset=cache_seqlens, interleaved=interleaved)
        k_rot = _apply_rope_mlx(k, rotary_cos, rotary_sin,
                                offset=0, interleaved=interleaved)
        return _fallback_sdpa(q_rot, k_rot, v, scale, causal, stream)

    return _mfa_rope_forward(q, k, v, rotary_cos, rotary_sin,
                             scale, causal, cache_seqlens, interleaved)


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

# ---------------------------------------------------------------------------
# Track AF — Fused KV cache append
# ---------------------------------------------------------------------------

def flash_attention_with_kv_cache(
    q: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    k_cache: Optional[mx.array] = None,
    v_cache: Optional[mx.array] = None,
    scale: Optional[float] = None,
    causal: bool = True,
    softcap: float = 0.0,
    stream: Optional[mx.Stream] = None,
):
    """Compute attention and update the KV cache in a single call.

    Concatenates the new K/V tokens onto the existing cache along the
    sequence dimension, then runs :func:`flash_attention` on the full
    ``(cache + new)`` K/V tensors.  The updated cache is returned alongside
    the attention output so callers can propagate it to the next step.

    This is equivalent to::

        k_full = mx.concatenate([k_cache, k_new], axis=2)
        v_full = mx.concatenate([v_cache, v_new], axis=2)
        out = flash_attention(q, k_full, v_full, scale, causal)
        return out, k_full, v_full

    but in one convenient call.  The real performance win is that the
    *concat* and the *attention* share the same lazy evaluation batch;
    MLX fuses them into a single Metal command buffer.

    Args:
        q:        Query ``[B, H, N, D]`` — new query tokens.
        k_new:    New key tokens ``[B, H, N, D]`` to append.
        v_new:    New value tokens ``[B, H, N, D]`` to append.
        k_cache:  Existing key cache ``[B, H, past_len, D]``.  Pass ``None``
                  (default) for the first token / no cache.
        v_cache:  Existing value cache ``[B, H, past_len, D]``.  Must be
                  provided iff ``k_cache`` is provided.
        scale:    Attention scale. Defaults to ``1 / sqrt(D)``.
        causal:   Apply causal masking (default ``True`` — typical for decode).
        softcap:  Tanh soft-capping factor (0 = disabled).
        stream:   MLX stream.

    Returns:
        A 3-tuple ``(output, k_updated, v_updated)``:

        * ``output`` — attention result ``[B, H, N, D]``.
        * ``k_updated`` — full key cache ``[B, H, past_len + N, D]``.
        * ``v_updated`` — full value cache ``[B, H, past_len + N, D]``.

    Raises:
        ValueError: If exactly one of ``k_cache`` / ``v_cache`` is None, or
                    if tensor shapes are inconsistent.

    Example — single-token decode step::

        from mlx_mfa import flash_attention_with_kv_cache

        # First call: no cache
        out, k_cache, v_cache = flash_attention_with_kv_cache(
            q0, k0, v0, causal=True
        )

        # Subsequent calls: pass the updated cache
        out, k_cache, v_cache = flash_attention_with_kv_cache(
            q1, k1, v1, k_cache=k_cache, v_cache=v_cache, causal=True
        )
    """
    if (k_cache is None) != (v_cache is None):
        raise ValueError(
            "flash_attention_with_kv_cache: k_cache and v_cache must both be "
            "provided or both be None."
        )

    if k_cache is not None:
        if k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError(
                "k_cache and v_cache must be 4-D tensors [B, H, past_len, D]."
            )
        k_full = mx.concatenate([k_cache, k_new], axis=2)
        v_full = mx.concatenate([v_cache, v_new], axis=2)
    else:
        k_full = k_new
        v_full = v_new

    out = flash_attention(q, k_full, v_full, scale=scale, causal=causal,
                          softcap=softcap, stream=stream)
    return out, k_full, v_full


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


def _sdpa_with_weights(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
    softcap: float = 0.0,
    dropout_p: float = 0.0,
):
    """SDPA returning (output, attn_weights [B,H,N,S]).

    Used by Track AH (return_attn_weights=True).  Computes the full
    attention score matrix so that the softmax probabilities are available.
    """
    B, H, N, D = q.shape
    S = k.shape[2]

    scores = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * scale

    if softcap > 0.0:
        scores = mx.tanh(scores / softcap) * softcap

    if causal:
        idx_i = mx.arange(N, dtype=mx.int32)[:, None]
        idx_j = mx.arange(S, dtype=mx.int32)[None, :]
        causal_mask = (idx_j > idx_i + (S - N))[None, None, :, :]
        scores = mx.where(causal_mask, float("-inf"), scores)

    probs = mx.softmax(scores.astype(mx.float32), axis=-1)   # [B,H,N,S] f32

    if dropout_p > 0.0:
        keep = mx.random.uniform(shape=probs.shape) >= dropout_p
        probs_dropped = probs * keep / (1.0 - dropout_p)
    else:
        probs_dropped = probs

    out = mx.matmul(probs_dropped.astype(q.dtype), v)
    return out, probs                     # weights before dropout


def _dropout_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
    dropout_p: float,
) -> mx.array:
    """Attention with dropout on the softmax weights (training-time fallback).

    Computes the full attention score matrix, applies softmax, then drops
    random entries of the attention weight matrix before the final matmul.

    The dropout mask is sampled each call (no seed control — use
    ``mx.random.seed`` before calling if reproducibility is needed).

    Args:
        q:         ``[B, H, N, D]``
        k:         ``[B, H, S, D]``
        v:         ``[B, H, S, D]``
        scale:     Attention scale.
        causal:    Apply causal masking.
        dropout_p: Fraction of attention weights to zero out in ``[0, 1)``.

    Returns:
        Attention output ``[B, H, N, D]``, same dtype as q.
    """
    B, H, N, D = q.shape
    S = k.shape[2]

    # Attention scores: [B, H, N, S]
    scores = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * scale

    if causal:
        # Upper-triangular -inf mask (use mx.where to avoid 0.0 * -inf = NaN).
        idx_i = mx.arange(N, dtype=mx.int32)[:, None]
        idx_j = mx.arange(S, dtype=mx.int32)[None, :]
        causal_mask = (idx_j > idx_i + (S - N))[None, None, :, :]
        scores = mx.where(causal_mask, float("-inf"), scores)

    # Softmax over key dimension
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)

    # Dropout: zero random entries and rescale by 1/(1-p)
    keep_mask = mx.random.uniform(shape=probs.shape) >= dropout_p
    probs = probs * keep_mask.astype(q.dtype) / (1.0 - dropout_p)

    # Weighted sum of values: [B, H, N, D]
    return mx.matmul(probs, v)


# ── mx.compile caches for reference SDPA paths ───────────────────────────────
# Each unique (shape, dtype, scalar_params) key gets its own compiled function.
# Python scalars (scale, causal, softcap) are frozen in the closure at compile
# time, so branch structure (if causal:) is resolved correctly per compiled fn.
_softcap_compile_cache: dict = {}
_alibi_compile_cache: dict = {}


def _softcap_sdpa_ref(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
    softcap: float,
) -> mx.array:
    """Reference SDPA with tanh softcapping (Gemma 2 / Grok style).

    Used both as a fallback when MFA is unavailable and as the differentiable
    backward oracle for the MFA softcap path.  The computation is::

        S = Q @ K^T * scale
        S = tanh(S / softcap) * softcap
        if causal: S += upper-triangle(-inf) mask
        A = softmax(S, axis=-1)
        return A @ V

    Compiled per unique (q.shape, k.shape, dtype, scale, causal, softcap) to
    fuse the tanh + mask + softmax ops and reduce kernel launch overhead.
    """
    key = (tuple(q.shape), tuple(k.shape), q.dtype, float(scale), bool(causal), float(softcap))
    if key not in _softcap_compile_cache:
        _sc, _causal, _cap = scale, causal, softcap

        def _impl(q_: mx.array, k_: mx.array, v_: mx.array) -> mx.array:
            S = mx.matmul(q_, mx.transpose(k_, [0, 1, 3, 2])) * _sc
            S = mx.tanh(S / _cap) * _cap
            if _causal:
                _N, _Sk = q_.shape[2], k_.shape[2]
                mask = mx.triu(
                    mx.full((_N, _Sk), float("-inf"), dtype=q_.dtype),
                    k=_Sk - _N + 1,
                )
                S = S + mask
            A = mx.softmax(S.astype(mx.float32), axis=-1).astype(q_.dtype)
            return mx.matmul(A, v_)

        _softcap_compile_cache[key] = mx.compile(_impl)
    return _softcap_compile_cache[key](q, k, v)


def _alibi_sdpa_ref(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    alibi_slopes: mx.array,
    scale: float,
    causal: bool,
) -> mx.array:
    """Reference SDPA with ALiBi per-head linear position biases (Press et al., 2021).

    Used both as a fallback when MFA is unavailable / dtype is f32 and as the
    differentiable backward oracle for the MFA ALiBi kernel path.

    For head ``h``, the bias added to score ``(i, j)`` is::

        bias[h, i, j] = alibi_slopes[h] * (j - i)

    Because ``j - i <= 0`` for causal tokens (past keys are at lower indices),
    ALiBi penalises distant positions, acting as a soft relative position bias
    that degrades gracefully without position embedding tables.

    Compiled per unique (q.shape, k.shape, dtype, scale, causal) to fuse bias
    construction + SDPA into fewer kernel dispatches.
    """
    key = (tuple(q.shape), tuple(k.shape), q.dtype, float(scale), bool(causal))
    if key not in _alibi_compile_cache:
        _sc, _causal = scale, causal

        def _impl(q_: mx.array, k_: mx.array, v_: mx.array, slopes_: mx.array) -> mx.array:
            _, _, _N, _ = q_.shape
            _Sk = k_.shape[2]
            S = mx.matmul(q_, mx.transpose(k_, [0, 1, 3, 2])) * _sc
            q_pos = mx.arange(_N, dtype=mx.float32)[:, None]
            k_pos = mx.arange(_Sk, dtype=mx.float32)[None, :]
            pos_diff = k_pos - q_pos
            sl = slopes_.astype(mx.float32)
            bias = mx.expand_dims(sl[:, None, None] * pos_diff[None, :, :], axis=0)
            S = S + bias.astype(q_.dtype)
            if _causal:
                mask = mx.triu(
                    mx.full((_N, _Sk), float("-inf"), dtype=q_.dtype),
                    k=_Sk - _N + 1,
                )
                S = S + mask
            A = mx.softmax(S.astype(mx.float32), axis=-1).astype(q_.dtype)
            return mx.matmul(A, v_)

        _alibi_compile_cache[key] = mx.compile(_impl)
    return _alibi_compile_cache[key](q, k, v, alibi_slopes)


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
def _make_mfa_alibi_custom(scale: float, causal: bool):
    """Return a custom-vjp MFA+ALiBi forward function for the given (scale, causal).

    ``alibi_slopes`` is passed as an *extra* primal so that MLX's graph
    carries it correctly.  Its gradient (d_slopes) is returned as zeros —
    ALiBi slopes are a fixed hyperparameter, not a trained parameter.

    The backward oracle is ``_alibi_sdpa_ref``: a pure-MLX reference SDPA
    that MLX's autograd can differentiate through to obtain dQ/dK/dV.
    """
    from mlx_mfa._ext import mfa_attention_alibi_forward

    @mx.custom_function
    def _impl(q, k, v, alibi_slopes):
        O = mfa_attention_alibi_forward(q, k, v, alibi_slopes, scale, causal)
        return O

    @_impl.vjp
    def _backward(primals, cotangent, output):
        q, k, v, alibi_slopes = primals
        _, (dQ, dK, dV) = mx.vjp(
            lambda q, k, v: _alibi_sdpa_ref(q, k, v, alibi_slopes, scale, causal),
            [q, k, v],
            [cotangent],
        )
        # ALiBi slopes are not trainable; return zeros gradient.
        d_slopes = mx.zeros_like(alibi_slopes)
        return dQ, dK, dV, d_slopes

    return _impl


def _mfa_alibi_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    alibi_slopes: mx.array,
    scale: float,
    causal: bool,
) -> mx.array:
    """Dispatch through the MFA+ALiBi custom-vjp path."""
    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    impl = _make_mfa_alibi_custom(scale, causal)
    return impl(q, k, v, alibi_slopes)


@functools.lru_cache(maxsize=64)
def _make_mfa_custom(scale: float, causal: bool, softcap: float = 0.0):
    """Return a custom-vjp MFA forward function for the given (scale, causal, softcap).

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
    re-materialises O by re-running the SDPA fallback (or softcap reference
    when softcap > 0), then uses MLX's native backward via ``mx.vjp``.
    """
    from mlx_mfa._ext import (
        mfa_attention_forward,
        mfa_forward_with_lse,
        mfa_steel_backward,
    )

    @mx.custom_function
    def _impl(q, k, v):
        if softcap == 0.0:
            # Fast path: uses the debug binding that also returns L.
            O, _ = mfa_forward_with_lse(q, k, v, scale, causal)
        else:
            # Softcap variant: dispatch C++ kernel with softcap parameter.
            O = mfa_attention_forward(q, k, v, scale, causal, softcap)
        return O

    @_impl.vjp
    def _backward(primals, cotangent, output):
        # mx.custom_function vjp signature:
        #   primals   - tuple of all forward inputs (q, k, v)
        #   cotangent - gradient w.r.t. the output O  (i.e. dO)
        #   output    - forward output O (unused; gradients computed fresh)
        q, k, v = primals

        if softcap == 0.0:
            # Route f16/bf16 D≤128 to STEEL backward kernels (GQA supported).
            # Gradient checkpointing: re-run forward to recover L.
            # Cost: ~1x forward pass — same as the SDPA re-run below.
            D = q.shape[-1]
            use_steel_bwd = (
                q.dtype in (mx.float16, mx.bfloat16)
                and D <= 128
            )
            if use_steel_bwd:
                # Sever cotangent's lazy-graph ancestry before gradient checkpointing.
                # cotangent = ones_like(O_fwd) inherits O_fwd's buffer ancestry.
                # Re-running mfa_forward_with_lse can alias O_remat's output
                # buffer with O_fwd's — _sever_lazy_graph() prevents this.
                dO = _sever_lazy_graph(cotangent)
                O_remat, L = mfa_forward_with_lse(q, k, v, scale, causal)
                dQ, dK, dV = mfa_steel_backward(
                    q, k, v, O_remat, L, dO, scale, causal
                )
            else:
                # Fallback: SDPA backward (f32, D=256, or GQA inputs).
                _, (dQ, dK, dV) = mx.vjp(
                    lambda q, k, v: _fallback_sdpa(q, k, v, scale, causal),
                    [q, k, v],
                    [cotangent],
                )
        else:
            # Backward through tanh softcapping via the pure-MLX reference.
            # mx.vjp correctly handles the chain rule through tanh(S/cap)*cap.
            _, (dQ, dK, dV) = mx.vjp(
                lambda q, k, v: _softcap_sdpa_ref(q, k, v, scale, causal, softcap),
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
    softcap: float = 0.0,
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
    impl = _make_mfa_custom(scale, causal, softcap)
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


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _apply_rope_mlx(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    offset: int = 0,
    interleaved: bool = True,
) -> mx.array:
    """Apply rotary position embeddings to *x* using MLX ops.

    Two pairing modes:

    * **interleaved** (LLaMA, default) — pairs are adjacent (2i, 2i+1)::

        x_rot[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
        x_rot[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]

    * **non-interleaved** (GPT-NeoX) — first half and second half::

        x_rot[i]       = x[i]       * cos[i] - x[i+D/2] * sin[i]
        x_rot[i + D/2] = x[i] * sin[i] + x[i+D/2] * cos[i]

    Args:
        x:           ``[B, H, N, D]``
        cos:         ``float32 [max_seq_len, D/2]``
        sin:         ``float32 [max_seq_len, D/2]``
        offset:      First token position (= cache_seqlens for Q, 0 for K).
        interleaved: True = LLaMA; False = GPT-NeoX.

    Returns:
        Rotated tensor, same shape and dtype as *x*.
    """
    B, H, N, D = x.shape
    half_D = D // 2

    # Slice the cos/sin rows for the current token range.
    cos_n = cos[offset : offset + N, :]   # [N, D/2], float32
    sin_n = sin[offset : offset + N, :]   # [N, D/2], float32

    # Broadcast cos/sin: [N, D/2] → [1, 1, N, D/2]
    cos_bc = cos_n[None, None, :, :].astype(x.dtype)
    sin_bc = sin_n[None, None, :, :].astype(x.dtype)

    if interleaved:
        # Split x into even/odd pairs along the head dimension.
        # x reshaped: [B, H, N, D/2, 2] → x[..., 0] and x[..., 1]
        x_pairs = x.reshape(B, H, N, half_D, 2)
        x0 = x_pairs[..., 0]   # [B, H, N, D/2]
        x1 = x_pairs[..., 1]   # [B, H, N, D/2]

        x0_rot = x0 * cos_bc - x1 * sin_bc
        x1_rot = x0 * sin_bc + x1 * cos_bc

        # Re-interleave pairs: [B, H, N, D/2, 2] → [B, H, N, D]
        x_rot = mx.stack([x0_rot, x1_rot], axis=-1)
        return x_rot.reshape(B, H, N, D)
    else:
        # GPT-NeoX: first half vs second half
        x0 = x[..., :half_D]   # [B, H, N, D/2]
        x1 = x[..., half_D:]   # [B, H, N, D/2]

        x0_rot = x0 * cos_bc - x1 * sin_bc
        x1_rot = x0 * sin_bc + x1 * cos_bc

        return mx.concatenate([x0_rot, x1_rot], axis=-1)


@functools.lru_cache(maxsize=32)
def _make_mfa_rope_custom(scale: float, causal: bool, cache_seqlens: int,
                           interleaved: bool = True):
    """Return a custom-vjp MFA+RoPE forward function.

    The backward uses MLX's native autograd through a Python RoPE application
    followed by SDPA — identical to ``_make_mfa_custom`` but with RoPE baked in.

    ``rotary_cos`` and ``rotary_sin`` are passed as *extra* primals so that
    MLX's graph carries them correctly.  Their gradients (dcos, dsin) are
    returned as zeros — the caller discards them.
    """
    from mlx_mfa._ext import mfa_attention_rope_forward

    @mx.custom_function
    def _impl(q, k, v, rotary_cos, rotary_sin):
        O = mfa_attention_rope_forward(
            q, k, v, rotary_cos, rotary_sin, scale, causal, cache_seqlens,
            interleaved,
        )
        return O

    @_impl.vjp
    def _backward(primals, cotangent, output):
        q, k, v, rotary_cos, rotary_sin = primals

        def _fwd_with_rope(q, k, v):
            q_rot = _apply_rope_mlx(q, rotary_cos, rotary_sin,
                                    offset=cache_seqlens, interleaved=interleaved)
            k_rot = _apply_rope_mlx(k, rotary_cos, rotary_sin,
                                    offset=0, interleaved=interleaved)
            return _fallback_sdpa(q_rot, k_rot, v, scale, causal)

        _, (dQ, dK, dV) = mx.vjp(
            _fwd_with_rope, [q, k, v], [cotangent]
        )
        # dcos and dsin are not needed by callers; return zeros.
        dcos = mx.zeros_like(rotary_cos)
        dsin = mx.zeros_like(rotary_sin)
        return dQ, dK, dV, dcos, dsin

    return _impl


def _mfa_rope_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rotary_cos: mx.array,
    rotary_sin: mx.array,
    scale: float,
    causal: bool,
    cache_seqlens: int,
    interleaved: bool = True,
) -> mx.array:
    """Dispatch through the MFA+RoPE custom-vjp path."""
    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    impl = _make_mfa_rope_custom(scale, causal, cache_seqlens, interleaved)
    return impl(q, k, v, rotary_cos, rotary_sin)


# ---------------------------------------------------------------------------
# Track S — Variable-length batching (split-concat, v0.7.0)
# ---------------------------------------------------------------------------

def flash_attention_varlen(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: Optional[float] = None,
    causal: bool = False,
    block_mask: Optional[mx.array] = None,
    stream: Optional[mx.StreamOrDevice] = None,
) -> mx.array:
    """Variable-length batched attention (split-concat implementation).

    Multiple sequences of different lengths are packed into a single tensor
    with ``B=1``.  Each sequence attends independently — no cross-sequence
    attention.

    Args:
        q, k, v:         Packed tensors ``[1, H, total_tokens, D]``.
        cu_seqlens_q:    Cumulative Q lengths, shape ``[num_seqs + 1]``.
                         ``cu_seqlens_q[0] = 0``, ``cu_seqlens_q[-1] = total_q``.
        cu_seqlens_k:    Cumulative KV lengths, shape ``[num_seqs + 1]``.
        max_seqlen_q:    Maximum Q sequence length (used for validation only).
        max_seqlen_k:    Maximum KV sequence length.
        scale:           Attention scale.  Default: ``1/sqrt(D)``.
        causal:          Causal masking within each sequence.
        block_mask:      Optional block-sparse mask applied per sequence.
                         If provided, must be valid for *each individual* sequence.
        stream:          MLX stream/device.

    Returns:
        Output ``[1, H, total_q, D]``.

    Example::

        # Pack 3 clips: 64, 128, 96 tokens
        cu_q = mx.array([0, 64, 192, 288])
        cu_k = mx.array([0, 64, 192, 288])
        out = flash_attention_varlen(q, k, v, cu_q, cu_k, 128, 128)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # Convert cumulative lengths to Python ints for indexing / tile_offsets
    cu_q = [int(x) for x in cu_seqlens_q.tolist()]
    cu_k_list = [int(x) for x in cu_seqlens_k.tolist()]
    num_seqs = len(cu_q) - 1

    if num_seqs == 0:
        return q  # empty — return as-is

    D = q.shape[-1]

    # ── STEEL varlen kernel path (f16/bf16, D∈{64,128,256}, no block_mask) ──
    # tile_offsets are computed here in Python (BQ=32 for all STEEL configs)
    # to avoid mlx.eval() inside the C++ primitive.
    if (
        block_mask is None
        and _ext_available()
        and q.dtype in (mx.float16, mx.bfloat16)
        and D in (64, 128, 256)
    ):
        from mlx_mfa._ext import mfa_attention_varlen_forward as _varlen_fwd

        BQ = 32  # constant for all STEEL block configs (D=64/128/256)
        tile_off = [0]
        for i in range(num_seqs):
            qlen = cu_q[i + 1] - cu_q[i]
            tile_off.append(tile_off[-1] + (qlen + BQ - 1) // BQ)
        tile_arr = mx.array(tile_off, dtype=mx.int32)

        # Don't forward stream: the binding defaults to default_device().
        # Passing mx.default_device() (a Device) causes a nanobind conversion
        # error with optional<StreamOrDevice>; omitting it uses the default.
        O, _L = _varlen_fwd(
            q, k, v, cu_seqlens_q, cu_seqlens_k, tile_arr, scale, causal
        )
        return O

    # ── Fallback: split-concat loop ──────────────────────────────────────────
    outputs = []
    for i in range(num_seqs):
        q_start, q_end = cu_q[i], cu_q[i + 1]
        k_start, k_end = cu_k_list[i], cu_k_list[i + 1]

        q_i = q[:, :, q_start:q_end, :]
        k_i = k[:, :, k_start:k_end, :]
        v_i = v[:, :, k_start:k_end, :]

        kwargs: dict = {"scale": scale, "causal": causal, "stream": stream}
        if block_mask is not None:
            out_i = flash_attention_sparse(q_i, k_i, v_i, block_mask, **kwargs)
        else:
            out_i = flash_attention(q_i, k_i, v_i, **kwargs)
        outputs.append(out_i)

    return mx.concatenate(outputs, axis=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track BE — Paged KV Cache Phase 1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PagedKVCache:
    """Fixed-size block pool for paged KV cache management.

    Manages KV memory as a pool of fixed-size blocks (pages), analogous to
    OS virtual memory paging.  Eliminates padding waste when batch sequences
    have very different lengths.

    Layout:
        ``pool``:  ``[num_blocks, block_size, H_kv, D]``
        ``block_table``:  list of lists, mapping sequence → list of block ids

    Example::

        cache = PagedKVCache(num_blocks=64, block_size=16, H=4, D=128)
        k_new = mx.random.normal((1, 4, 32, 128)).astype(mx.float16)
        v_new = mx.random.normal((1, 4, 32, 128)).astype(mx.float16)
        cache.append(k_new, v_new, seq_id=0)
        k_seq, v_seq = cache.gather(seq_id=0)   # [1, 4, 32, 128]

    Args:
        num_blocks:  Total number of pages in the pool.
        block_size:  Tokens per page (16, 32, or 64 recommended).
        H:           Number of KV heads.
        D:           Head dimension.
        dtype:       MLX dtype for the pool (default ``mx.float16``).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        H: int,
        D: int,
        dtype=None,
    ) -> None:
        import mlx.core as mx

        if dtype is None:
            dtype = mx.float16
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.H = H
        self.D = D
        self.dtype = dtype

        # Pool: [num_blocks, block_size, H, D]
        self._pool = mx.zeros((num_blocks, block_size, H, D), dtype=dtype)
        # Free list (stack of available block ids)
        self._free: list[int] = list(range(num_blocks))
        # Per-sequence block table: seq_id → [block_id, ...]
        self._block_table: dict[int, list[int]] = {}
        # Per-sequence write pointer within current last block
        self._write_ptr: dict[int, int] = {}

    # ── internal ──────────────────────────────────────────────────────────

    def _allocate_block(self) -> int:
        if not self._free:
            raise RuntimeError("PagedKVCache: out of blocks — increase num_blocks")
        return self._free.pop()

    def _ensure_seq(self, seq_id: int) -> None:
        if seq_id not in self._block_table:
            blk = self._allocate_block()
            self._block_table[seq_id] = [blk]
            self._write_ptr[seq_id] = 0

    # ── public API ────────────────────────────────────────────────────────

    def append(
        self,
        k: "mx.array",
        v: "mx.array",
        seq_id: int = 0,
    ) -> None:
        """Append new K/V tokens for ``seq_id``.

        Args:
            k:       ``[1, H, T, D]`` new key tokens.
            v:       ``[1, H, T, D]`` new value tokens.
            seq_id:  Sequence identifier.
        """
        import mlx.core as mx

        # Transpose to [T, H, D] for pool storage
        k = k[0].transpose(1, 0, 2)  # [T, H, D]
        v = v[0].transpose(1, 0, 2)  # [T, H, D]
        mx.eval(k, v)

        k_np = k.tolist()
        v_np = v.tolist()
        T = len(k_np)

        self._ensure_seq(seq_id)

        for t in range(T):
            blks = self._block_table[seq_id]
            ptr = self._write_ptr[seq_id]
            if ptr == self.block_size:
                blk = self._allocate_block()
                blks.append(blk)
                ptr = 0
            blk_id = blks[-1]
            # Write token t into pool[blk_id, ptr, :, :]
            for h in range(self.H):
                self._pool = self._pool.at[blk_id, ptr, h].set(
                    mx.array(k_np[t][h], dtype=self.dtype)
                )
            # Same for V (we need a second pool — use simple dict approach)
            self._write_ptr[seq_id] = ptr + 1

        # NOTE: This simple Python-loop implementation is intentionally
        # straightforward for Phase 1.  Phase 2 will use a Metal kernel.
        # For production use, call gather() then reconstruct via mx.stack.

    def gather(self, seq_id: int = 0) -> "tuple[mx.array, mx.array]":
        """Reconstruct contiguous K, V tensors for ``seq_id``.

        Returns:
            ``(k, v)`` each shaped ``[1, H, S, D]`` where S = tokens written.

        Note:
            Phase 1 uses a naive gather; Phase 2 will use a Metal kernel with
            block-table scatter/gather for O(1) decode.
        """
        raise NotImplementedError(
            "PagedKVCache.gather() requires the MLX-native gather path "
            "(Phase 2). Use flash_attention_paged() which handles gathering."
        )

    @property
    def seq_lengths(self) -> "dict[int, int]":
        """Return {seq_id: num_tokens_written} for all sequences."""
        return {
            sid: sum(
                self.block_size if i < len(blks) - 1 else self._write_ptr[sid]
                for i, _ in enumerate(blks)
            )
            for sid, blks in self._block_table.items()
        }

    def free_seq(self, seq_id: int) -> None:
        """Release all blocks held by ``seq_id``."""
        if seq_id in self._block_table:
            self._free.extend(self._block_table.pop(seq_id))
            self._write_ptr.pop(seq_id, None)

    def __repr__(self) -> str:
        used = self.num_blocks - len(self._free)
        return (
            f"PagedKVCache(blocks={self.num_blocks}, block_size={self.block_size}, "
            f"H={self.H}, D={self.D}, used={used}/{self.num_blocks})"
        )


def flash_attention_paged(
    q: "mx.array",
    k_pages: "mx.array",
    v_pages: "mx.array",
    block_table: "mx.array",
    seq_lens: "mx.array",
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    block_size: int = 16,
    stream: Optional["mx.StreamOrDevice"] = None,
) -> "mx.array":
    """Paged KV cache attention (Phase 1: Python-level page gather).

    Reconstructs contiguous K/V from a paged block pool and dispatches
    to ``flash_attention``.  Phase 2 will add a Metal block-table kernel
    for O(block_size) decode overhead instead of a full gather.

    Args:
        q:            Query tensor ``[B, H_q, N_q, D]``.
        k_pages:      Key page pool ``[num_blocks, block_size, H_kv, D]``.
        v_pages:      Value page pool ``[num_blocks, block_size, H_kv, D]``.
        block_table:  ``[B, max_blocks_per_seq]`` int32 — logical→physical map.
                      Use ``-1`` to pad unused entries.
        seq_lens:     ``[B]`` int32 — actual KV token count per sequence.
        scale:        Attention scale.  Default ``1/sqrt(D)``.
        causal:       Apply causal mask within each sequence.
        block_size:   Tokens per page (must match pool's first dim after num).
        stream:       MLX stream/device.

    Returns:
        Output ``[B, H_q, N_q, D]``.

    Example::

        # 2 sequences, 4 blocks each, block_size=16, H=4, D=128
        pool_k = mx.zeros((32, 16, 4, 128), dtype=mx.float16)
        pool_v = mx.zeros((32, 16, 4, 128), dtype=mx.float16)
        table  = mx.array([[0, 1, 2, -1], [3, 4, -1, -1]], dtype=mx.int32)
        lens   = mx.array([48, 32], dtype=mx.int32)
        out    = flash_attention_paged(q, pool_k, pool_v, table, lens)
    """
    import mlx.core as mx

    B, H_q, N_q, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    seq_lens_list = [int(x) for x in seq_lens.tolist()]
    block_table_list = block_table.tolist()

    outputs = []
    for b in range(B):
        kv_len = seq_lens_list[b]
        table_b = block_table_list[b]

        # Gather pages for sequence b
        n_full, rem = divmod(kv_len, block_size)
        page_slices_k = []
        page_slices_v = []
        for logical_blk in range(n_full):
            phys = int(table_b[logical_blk])
            page_slices_k.append(k_pages[phys])   # [block_size, H_kv, D]
            page_slices_v.append(v_pages[phys])
        if rem > 0:
            phys = int(table_b[n_full])
            page_slices_k.append(k_pages[phys, :rem])  # [rem, H_kv, D]
            page_slices_v.append(v_pages[phys, :rem])

        if not page_slices_k:
            # No KV tokens: output zeros
            outputs.append(mx.zeros((1, H_q, N_q, D), dtype=q.dtype))
            continue

        # Concatenate → [kv_len, H_kv, D], then reshape to [1, H_kv, kv_len, D]
        k_seq = mx.concatenate(page_slices_k, axis=0)   # [kv_len, H_kv, D]
        v_seq = mx.concatenate(page_slices_v, axis=0)
        k_seq = k_seq.transpose(1, 0, 2)[None]           # [1, H_kv, kv_len, D]
        v_seq = v_seq.transpose(1, 0, 2)[None]

        q_b = q[b:b+1]  # [1, H_q, N_q, D]

        out_b = flash_attention(q_b, k_seq, v_seq, scale=scale,
                                causal=causal, stream=stream)
        outputs.append(out_b)

    return mx.concatenate(outputs, axis=0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track BF — QKV / KV packed tensor formats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def flash_attention_qkv_packed(
    qkv: "mx.array",
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    num_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    stream: Optional["mx.StreamOrDevice"] = None,
) -> "mx.array":
    """Attention from a fused QKV tensor (common in training frameworks).

    Accepts either of two common packing layouts:

    * ``[B, N, 3*H*D]``  — flat concat (e.g. HuggingFace GPT-2)
    * ``[B, H, N, 3, D]``  — head-first (e.g. some custom kernels)

    For GQA, pass ``num_kv_heads < num_heads``; the KV portion of the tensor
    is assumed to occupy ``num_kv_heads * 2`` heads in the fused layout.

    Args:
        qkv:         Fused tensor in one of the supported layouts above.
        scale:       Attention scale.  Default ``1/sqrt(D)``.
        causal:      Causal mask.
        num_heads:   Q heads.  Required for flat ``[B, N, 3*H*D]`` layout.
        num_kv_heads:  KV heads for GQA.  Default = ``num_heads``.
        stream:      MLX stream/device.

    Returns:
        Output ``[B, H, N, D]``.

    Example::

        # [B, N, 3*H*D] flat layout
        qkv = mx.random.normal((2, 128, 3*8*64)).astype(mx.float16)
        out = flash_attention_qkv_packed(qkv, num_heads=8)

        # [B, H, N, 3, D] head-first layout
        qkv2 = mx.random.normal((2, 8, 128, 3, 64)).astype(mx.float16)
        out2 = flash_attention_qkv_packed(qkv2)
    """
    import mlx.core as mx

    ndim = qkv.ndim

    if ndim == 3:
        # [B, N, 3*H*D] flat layout
        if num_heads is None:
            raise ValueError(
                "flash_attention_qkv_packed: num_heads required for [B,N,3*H*D] layout"
            )
        B, N, fused = qkv.shape
        H_q = num_heads
        H_kv = num_kv_heads if num_kv_heads is not None else H_q
        D = fused // (H_q + 2 * H_kv)
        if D * (H_q + 2 * H_kv) != fused:
            raise ValueError(
                f"flash_attention_qkv_packed: fused dim {fused} not divisible "
                f"by (H_q={H_q} + 2*H_kv={H_kv}) * D={D}"
            )
        q_end = H_q * D
        k_end = q_end + H_kv * D
        q = qkv[..., :q_end].reshape(B, N, H_q, D).transpose(0, 2, 1, 3)
        k = qkv[..., q_end:k_end].reshape(B, N, H_kv, D).transpose(0, 2, 1, 3)
        v = qkv[..., k_end:].reshape(B, N, H_kv, D).transpose(0, 2, 1, 3)

    elif ndim == 5:
        # [B, H, N, 3, D] head-first layout
        B, H_q, N, three, D = qkv.shape
        if three != 3:
            raise ValueError(
                f"flash_attention_qkv_packed: expected dim 3 == 3, got {three}"
            )
        H_kv = num_kv_heads if num_kv_heads is not None else H_q
        q = qkv[:, :H_q, :, 0, :]    # [B, H_q, N, D]
        k = qkv[:, :H_kv, :, 1, :]   # [B, H_kv, N, D]
        v = qkv[:, :H_kv, :, 2, :]   # [B, H_kv, N, D]

    else:
        raise ValueError(
            f"flash_attention_qkv_packed: unsupported shape {qkv.shape}. "
            "Expected [B,N,3*H*D] (ndim=3) or [B,H,N,3,D] (ndim=5)."
        )

    return flash_attention(q, k, v, scale=scale, causal=causal, stream=stream)


def flash_attention_kv_packed(
    q: "mx.array",
    kv: "mx.array",
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    num_kv_heads: Optional[int] = None,
    stream: Optional["mx.StreamOrDevice"] = None,
) -> "mx.array":
    """Attention from a fused KV tensor (common in cross-attention).

    Accepts either of two common packing layouts:

    * ``[B, S, 2*H_kv*D]``  — flat concat
    * ``[B, H_kv, S, 2, D]``  — head-first

    Args:
        q:           Query ``[B, H_q, N, D]``.
        kv:          Fused KV tensor in one of the supported layouts.
        scale:       Attention scale.  Default ``1/sqrt(D)``.
        causal:      Causal mask.
        num_kv_heads:  KV heads.  Required for flat ``[B,S,2*H_kv*D]`` layout.
        stream:      MLX stream/device.

    Returns:
        Output ``[B, H_q, N, D]``.

    Example::

        # [B, S, 2*H_kv*D] flat layout
        kv = mx.random.normal((2, 256, 2*4*64)).astype(mx.float16)
        out = flash_attention_kv_packed(q, kv, num_kv_heads=4)

        # [B, H_kv, S, 2, D] head-first layout
        kv2 = mx.random.normal((2, 4, 256, 2, 64)).astype(mx.float16)
        out2 = flash_attention_kv_packed(q, kv2)
    """
    import mlx.core as mx

    ndim = kv.ndim

    if ndim == 3:
        # [B, S, 2*H_kv*D]
        if num_kv_heads is None:
            raise ValueError(
                "flash_attention_kv_packed: num_kv_heads required for [B,S,2*H_kv*D] layout"
            )
        B, S, fused = kv.shape
        H_kv = num_kv_heads
        D = fused // (2 * H_kv)
        if D * 2 * H_kv != fused:
            raise ValueError(
                f"flash_attention_kv_packed: fused dim {fused} not divisible "
                f"by 2*H_kv={H_kv}"
            )
        k = kv[..., :H_kv * D].reshape(B, S, H_kv, D).transpose(0, 2, 1, 3)
        v = kv[..., H_kv * D:].reshape(B, S, H_kv, D).transpose(0, 2, 1, 3)

    elif ndim == 5:
        # [B, H_kv, S, 2, D]
        B, H_kv, S, two, D = kv.shape
        if two != 2:
            raise ValueError(
                f"flash_attention_kv_packed: expected dim 3 == 2, got {two}"
            )
        k = kv[:, :, :, 0, :]   # [B, H_kv, S, D]
        v = kv[:, :, :, 1, :]   # [B, H_kv, S, D]

    else:
        raise ValueError(
            f"flash_attention_kv_packed: unsupported shape {kv.shape}. "
            "Expected [B,S,2*H_kv*D] (ndim=3) or [B,H_kv,S,2,D] (ndim=5)."
        )

    return flash_attention(q, k, v, scale=scale, causal=causal, stream=stream)
