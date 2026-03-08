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

_MFA_SUPPORTED_HDIMS = {64, 128, 256, 512}
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
    window_size: Optional[tuple] = None,
    return_lse: bool = False,
    stream: Optional[mx.Stream] = None,
    attn_bias: Optional[mx.array] = None,
    backend: str = "auto",
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
        window_size: Optional ``(left, right)`` tuple for sliding window
            attention.  ``left`` is the number of tokens to the left of each
            query that are visible; ``right`` is currently ignored (use
            ``causal=True`` to mask future tokens).  When ``left >= 0``,
            the STEEL kernel uses native tile-skip to skip K-tiles entirely
            outside the window.  Pass ``None`` (default) to disable.
        return_lse: When True, also return the log-sum-exp tensor
            ``L [B, H, N]`` in **log2 domain** alongside the output.
            Useful for Flash Decoding, speculative decoding, and any
            application that needs the attention normaliser.  When the MFA
            extension is available and the inputs are simple (no softcap,
            ALiBi, or dropout), ``L`` comes directly from the Metal kernel
            (free — no extra compute).  Otherwise a pure-MLX O(N·S) LSE
            materialisation is performed.  Mutually exclusive with
            ``return_attn_weights``.
        stream: MLX stream for async execution. Defaults to the default GPU
            stream. Currently only honoured on the fallback path; the MFA
            kernel always uses the default GPU stream.
        attn_bias: Optional additive bias added to attention scores before
            softmax, broadcastable to ``[B, H, N, S]``.  Can be used for
            padding masks (``-inf`` for padding positions), relative position
            encodings, or any per-element score adjustment.  When provided,
            the call falls back to ``mx.fast.scaled_dot_product_attention``
            (which passes it as the ``mask`` argument) because the MFA Metal
            kernel does not support a generic additive bias buffer.
            Mutually exclusive with ``alibi_slopes`` and ``softcap``.
        backend: Backend selection.  One of:

            * ``"auto"`` *(default)*: use the MFA Metal kernel when supported
              (head_dim ∈ {64,128,256,512}, dtype ∈ {f16,bf16,f32}, extension
              compiled); fall back to ``mx.fast.scaled_dot_product_attention``
              otherwise.
            * ``"mfa"``: force the MFA Metal kernel.  Raises ``RuntimeError``
              if the C++ extension is not compiled or the configuration is
              unsupported.
            * ``"sdpa"``: always use ``mx.fast.scaled_dot_product_attention``.
              Useful for baseline benchmarks or debugging.

    Returns:
        When ``return_attn_weights=False`` and ``return_lse=False``
        (default): attention output of shape
        ``[batch, heads, seq_len, head_dim]`` in the same dtype as q.

        When ``return_attn_weights=True``: a 2-tuple
        ``(output, attn_weights)`` where ``output`` is ``[B, H, N, D]`` and
        ``attn_weights`` is ``float32 [B, H, N, S]``.

        When ``return_lse=True``: a 2-tuple ``(output, L)`` where ``L`` is
        ``float32 [B, H, N]`` in log2 domain
        (i.e. ``L = log2(sum_j 2^{score_j})``).  Mutually exclusive with
        ``return_attn_weights``.

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
    _VALID_BACKENDS = {"auto", "mfa", "sdpa"}
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"flash_attention: backend must be one of {sorted(_VALID_BACKENDS)},"
            f" got {backend!r}."
        )

    # --- backend='sdpa': unconditional SDPA fallback -------------------------
    if backend == "sdpa":
        mask = attn_bias  # may be None
        if causal:
            N, S = q.shape[2], k.shape[2]
            causal_mask = mx.triu(
                mx.full((N, S), float("-inf"), dtype=q.dtype), k=S - N + 1
            )
            mask = causal_mask if mask is None else causal_mask + mask
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])),
            mask=mask,
        )

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
        if return_lse:
            raise ValueError(
                "return_attn_weights and return_lse are mutually exclusive."
            )
        return _sdpa_with_weights(q, k, v, scale, causal, softcap, dropout_p)

    # Track AG: dropout falls back to Python SDPA (MFA kernel has no dropout).
    if dropout_p > 0.0:
        return _dropout_sdpa(q, k, v, scale, causal, dropout_p)

    # Track ID: attn_bias — MFA kernel has no generic additive bias path;
    # fall back to SDPA which passes it as the mask argument.
    if attn_bias is not None:
        mask = attn_bias
        if causal:
            N, S = q.shape[2], k.shape[2]
            causal_mask = mx.triu(
                mx.full((N, S), float("-inf"), dtype=q.dtype), k=S - N + 1
            )
            mask = causal_mask + mask
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask,
        )

    # Track ID: backend='mfa' — force MFA; raise if unavailable.
    use_mfa = _can_use_mfa(q, head_dim) and not v_dim_mismatch
    if backend == "mfa" and not use_mfa:
        try:
            from mlx_mfa import _ext  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "flash_attention(backend='mfa'): the MFA C++ extension is not "
                "compiled. Run: pip install -e . (with cmake args)."
            )
        raise RuntimeError(
            f"flash_attention(backend='mfa'): unsupported configuration — "
            f"head_dim={head_dim}, dtype={q.dtype}, v_dim_mismatch={v_dim_mismatch}. "
            f"Supported: head_dim∈{{64,128,256,512}}, dtype∈{{f16,bf16,f32}}."
        )

    if not use_mfa:
        if softcap != 0.0:
            return _softcap_sdpa_ref(q, k, v, scale, causal, softcap)
        if alibi_slopes is not None:
            return _alibi_sdpa_ref(q, k, v, alibi_slopes, scale, causal)
        if return_lse:
            return _fallback_sdpa_with_lse(q, k, v, scale, causal)
        return _fallback_sdpa(q, k, v, scale, causal, stream)

    # ALiBi requires f16/bf16 for the Metal kernel (f32 has no STEEL ALiBi).
    if alibi_slopes is not None:
        if q.dtype == mx.float32:
            return _alibi_sdpa_ref(q, k, v, alibi_slopes, scale, causal)
        return _mfa_alibi_forward(q, k, v, alibi_slopes, scale, causal)

    # Convert window_size=(left, right) → window_left for the STEEL kernel.
    # Only f16/bf16 support native window; f32 falls back to masked SDPA.
    window_left = -1
    if window_size is not None:
        wl = window_size[0]
        if wl >= 0 and q.dtype != mx.float32:
            window_left = wl
        else:
            # f32 or negative left: windowed SDPA fallback
            N, S = q.shape[2], k.shape[2]
            wl = max(wl, 0)
            q_idx = mx.arange(S - N, S, dtype=mx.int32)[:, None]
            k_idx = mx.arange(S, dtype=mx.int32)[None, :]
            in_win = k_idx >= q_idx - wl
            if causal:
                in_win = in_win & (k_idx <= q_idx)
            mask = mx.where(in_win,
                            mx.zeros((N, S), dtype=q.dtype),
                            mx.full((N, S), float("-inf"), dtype=q.dtype))
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask)

    # Track FX-1: return_lse — use mfa_forward_with_lse to get L for free.
    if return_lse:
        from mlx_mfa._ext import mfa_forward_with_lse
        q = mx.contiguous(q)
        k = mx.contiguous(k)
        v = mx.contiguous(v)
        O, L = mfa_forward_with_lse(q, k, v, scale, causal)
        return O, L

    return _mfa_forward(q, k, v, scale, causal, softcap, window_left, stream)


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
    rotary_dim: Optional[int] = None,
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
                rope_3d=None, interleaved=interleaved,
                rotary_dim=rotary_dim, stream=stream,
            )
            for b, cs in enumerate(cs_list)
        ]
        return mx.concatenate(chunks, axis=0)

    # RoPE requires f16/bf16 on the STEEL path.
    # Partial RoPE (rotary_dim < head_dim) also forces MLX fallback since the
    # STEEL kernel rotates the full head dimension.
    _partial_rope = rotary_dim is not None and rotary_dim < head_dim
    if not _can_use_mfa(q, head_dim) or q.dtype == mx.float32 or _partial_rope:
        q_rot = _apply_rope_mlx(q, rotary_cos, rotary_sin,
                                offset=cache_seqlens, interleaved=interleaved,
                                rotary_dim=rotary_dim)
        k_rot = _apply_rope_mlx(k, rotary_cos, rotary_sin,
                                offset=0, interleaved=interleaved,
                                rotary_dim=rotary_dim)
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

    .. deprecated:: 1.0.1
        Use :func:`flash_attention_kvcache` instead.
        ``flash_attention_with_kv_cache`` returns a 3-tuple and requires
        manually managing the growing cache tensor.
        :func:`flash_attention_kvcache` provides a cleaner API with
        ``cache_seqlens`` support, paged attention, and RoPE appending::

            out = flash_attention_kvcache(q, k_cache, v_cache,
                                          scale=scale, causal=True,
                                          cache_seqlens=mx.array([past_len]))

        This function will be removed in v2.0.

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


def flash_attention_kvcache_rope_append(
    q: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    k_cache: Optional[mx.array],
    v_cache: Optional[mx.array],
    rotary_cos: mx.array,
    rotary_sin: mx.array,
    cache_seqlens: int = 0,
    *,
    scale: Optional[float] = None,
    causal: bool = True,
    interleaved: bool = True,
    stream: Optional[mx.Stream] = None,
) -> tuple:
    """KV-cache append with fused RoPE rotation — stores keys pre-rotated.

    This is the recommended pattern for efficient autoregressive generation
    when using RoPE positional embeddings.  Keys are rotated *before* being
    appended to the cache, so the cache always contains pre-rotated keys.
    Only the new ``k_new`` tokens need rotation at each step, giving
    O(N_new) rotation cost instead of O(cache_len) per decode step.

    Concretely, this function:

    1. Rotates ``q`` at positions ``[cache_seqlens, cache_seqlens + N_q)``.
    2. Rotates ``k_new`` at positions ``[cache_seqlens, cache_seqlens + N_new)``.
    3. Concatenates ``k_new_rotated`` onto ``k_cache`` (and ``v_new`` onto ``v_cache``).
    4. Runs :func:`flash_attention` on the rotated Q and the full K/V.
    5. Returns ``(output, k_cache_updated, v_cache_updated)`` where the cache
       contains pre-rotated keys ready for the next step.

    Usage pattern for incremental decode::

        # Step 0: no cache
        out, k_cache, v_cache = flash_attention_kvcache_rope_append(
            q0, k0, v0, None, None, cos, sin, cache_seqlens=0,
        )
        # Step 1: append to cache
        out, k_cache, v_cache = flash_attention_kvcache_rope_append(
            q1, k1, v1, k_cache, v_cache, cos, sin,
            cache_seqlens=k_cache.shape[2],
        )

    Args:
        q:              Query ``[B, H_q, N_q, D]``.
        k_new:          New key tokens ``[B, H_kv, N_new, D]`` (unrotated).
        v_new:          New value tokens ``[B, H_kv, N_new, D]``.
        k_cache:        Existing key cache ``[B, H_kv, past_len, D]`` (pre-rotated).
                        Pass ``None`` for the first step.
        v_cache:        Existing value cache ``[B, H_kv, past_len, D]``.
        rotary_cos:     ``float32 [max_seq_len, D/2]`` cosine table.
        rotary_sin:     ``float32 [max_seq_len, D/2]`` sine table.
        cache_seqlens:  Current cache length = position of the first new token.
        scale:          Attention scale; defaults to ``1/sqrt(D)``.
        causal:         Apply causal masking (default ``True``).
        interleaved:    RoPE mode: ``True`` = LLaMA; ``False`` = GPT-NeoX.
        stream:         MLX stream.

    Returns:
        3-tuple ``(output, k_cache_updated, v_cache_updated)``:
        - ``output`` — ``[B, H_q, N_q, D]``
        - ``k_cache_updated`` — ``[B, H_kv, past_len + N_new, D]`` pre-rotated
        - ``v_cache_updated`` — ``[B, H_kv, past_len + N_new, D]``
    """
    D = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Rotate Q at current decode positions.
    q_rot = _apply_rope_mlx(q, rotary_cos, rotary_sin,
                             offset=cache_seqlens, interleaved=interleaved)
    # Rotate k_new at current decode positions (same offset as Q).
    k_new_rot = _apply_rope_mlx(k_new, rotary_cos, rotary_sin,
                                 offset=cache_seqlens, interleaved=interleaved)

    # Append to cache.
    if k_cache is not None:
        k_full = mx.concatenate([k_cache, k_new_rot], axis=2)
        v_full = mx.concatenate([v_cache, v_new], axis=2)
    else:
        k_full = k_new_rot
        v_full = v_new

    # Run attention on rotated Q, full pre-rotated K, V.
    out = flash_attention(q_rot, k_full, v_full, scale=scale, causal=causal,
                          stream=stream)
    return out, k_full, v_full


# ---------------------------------------------------------------------------
# Unified KV-cache API  (Track FA)
# ---------------------------------------------------------------------------

def flash_attention_kvcache(
    q: mx.array,
    k_cache: Optional[mx.array],
    v_cache: Optional[mx.array],
    *,
    # Paged mode: pass these instead of dense k_cache / v_cache
    block_table: Optional[mx.array] = None,
    seq_lens: Optional[mx.array] = None,
    block_size: int = 16,
    # Attention hyper-parameters
    scale: Optional[float] = None,
    causal: bool = True,
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    window_size: Optional[tuple] = None,
    # RoPE: applied to Q only (K already stored post-rotation in the cache)
    rotary_cos: Optional[mx.array] = None,
    rotary_sin: Optional[mx.array] = None,
    cache_seqlens: Union[int, "mx.array", Sequence[int]] = 0,
    interleaved: bool = True,
    # Track FX-3: partial RoPE — rotate only first rotary_dim head-dim elements
    rotary_dim: Optional[int] = None,
    # Track FX-2: continuous batching — map logical batch → cache pool slot
    cache_batch_idx: Optional[mx.array] = None,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Unified KV-cache attention — dense and paged modes in one call.

    This function is the recommended entry point for inference with KV caches.
    It consolidates the previously separate :func:`flash_attention_with_kv_cache`,
    :func:`flash_attention_paged`, and :func:`flash_attention_rope` paths and adds
    full support for RoPE, ALiBi, softcap, and sliding-window on both cache modes.

    **Dense mode** (default)::

        out = flash_attention_kvcache(q, k_full, v_full, causal=True)

    ``k_cache`` / ``v_cache`` contain the *complete* KV sequence (past tokens
    already concatenated by the caller).  This is the simplest usage — just
    pass the full accumulated cache each step.

    **Paged mode**::

        out = flash_attention_kvcache(
            q, k_pages, v_pages,
            block_table=table, seq_lens=lens, block_size=16,
        )

    ``k_cache`` / ``v_cache`` are the page *pool* tensors
    ``[num_blocks, block_size, H_kv, D]``; ``block_table`` ``[B, max_blocks]``
    (int32) maps logical pages to physical blocks; ``seq_lens`` ``[B]`` (int32)
    gives the actual KV length per sequence.

    **RoPE** (query-side)::

        out = flash_attention_kvcache(
            q, k_full, v_full,
            rotary_cos=cos, rotary_sin=sin,
            cache_seqlens=past_len, causal=True,
        )

    Only the query is re-rotated at decode time; keys are stored pre-rotated in
    the cache.  When the C++ STEEL kernel is available and the dtype is f16/bf16
    the rotation is fused inside the kernel.  Otherwise it falls back to a
    pure-MLX rotation followed by :func:`flash_attention`.

    **ALiBi**::

        out = flash_attention_kvcache(q, k, v, alibi_slopes=slopes, causal=True)

    ALiBi and RoPE are mutually exclusive.

    Args:
        q:              Query ``[B, H_q, N_q, D]``.
        k_cache:        Key tensor — dense ``[B, H_kv, S, D]`` *or* page pool
                        ``[num_blocks, block_size, H_kv, D]`` (paged mode).
        v_cache:        Value tensor — same layout as ``k_cache``.
        block_table:    ``[B, max_blocks_per_seq]`` int32 page→block map.
                        Providing this switches to **paged mode**.
        seq_lens:       ``[B]`` int32 actual KV length per sequence (paged mode).
        block_size:     Tokens per page pool block (paged mode only, default 16).
        scale:          Attention scale; defaults to ``1/sqrt(D)``.
        causal:         Apply causal masking (default ``True``).
        softcap:        Tanh soft-capping factor (0 = disabled).
        alibi_slopes:   ``float32 [H_q]`` ALiBi per-head slopes.  Mutually
                        exclusive with ``rotary_cos``/``rotary_sin``.
        window_size:    ``(left, right)`` sliding-window radii.  ``-1`` disables
                        that side.  Dense mode only.
        rotary_cos:     ``float32 [max_seq_len, D/2]`` cosine table.
        rotary_sin:     ``float32 [max_seq_len, D/2]`` sine table.
        cache_seqlens:  Absolute position of Q token 0 (scalar or ``[B]``).
                        Used as the RoPE offset for Q.  Typically ``past_len``.
        interleaved:    RoPE pairing mode: ``True`` = LLaMA (default), ``False``
                        = GPT-NeoX split-halves.
        cache_batch_idx: ``int32 [B]`` — optional batch→cache-pool index for
                        continuous batching.  When provided, ``k_cache`` and
                        ``v_cache`` are treated as a *pool* of shape
                        ``[pool_size, H_kv, S, D]`` (or compatible), and row
                        ``i`` of the logical batch selects
                        ``k_cache[cache_batch_idx[i]]``.  Allows multiple
                        logical requests to share a single large cache tensor
                        without copying.  Dense mode only.
        stream:         MLX stream.

    Returns:
        Attention output ``[B, H_q, N_q, D]``.

    Raises:
        ValueError: On shape mismatches, paged mode missing args, or ALiBi + RoPE.
    """
    # --- basic validation ---
    if q.ndim != 4:
        raise ValueError(
            f"flash_attention_kvcache: q must be 4-D [B, H, N, D], got {q.ndim}D."
        )
    D = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # RoPE and ALiBi are mutually exclusive.
    _has_rope = (rotary_cos is not None) or (rotary_sin is not None)
    if _has_rope and alibi_slopes is not None:
        raise ValueError(
            "flash_attention_kvcache: rotary_cos/sin and alibi_slopes are "
            "mutually exclusive."
        )

    # ----------------------------------------------------------------
    # PAGED MODE: block_table provided
    # ----------------------------------------------------------------
    if block_table is not None:
        if seq_lens is None:
            raise ValueError(
                "flash_attention_kvcache: paged mode requires seq_lens."
            )
        if k_cache is None or v_cache is None:
            raise ValueError(
                "flash_attention_kvcache: k_cache and v_cache (page pool) must "
                "be provided in paged mode."
            )
        if window_size is not None:
            raise ValueError(
                "flash_attention_kvcache: window_size is not supported in paged mode."
            )

        # Apply RoPE to Q only (keys are pre-rotated in the cache).
        q_att = q
        if _has_rope:
            if rotary_cos is None or rotary_sin is None:
                raise ValueError(
                    "flash_attention_kvcache: both rotary_cos and rotary_sin "
                    "must be provided together."
                )
            # Use the STEEL rope path if available; else pure-MLX rotation.
            from mlx_mfa.attention import _apply_rope_mlx, _can_use_mfa
            if _can_use_mfa(q, D) and q.dtype != mx.float32:
                # Rotate Q in-kernel: build a dummy single-element K that will
                # be discarded, but the Q rotation is correct.
                # Simplest: use the MLX path for paged + rope.
                pass  # fall through to MLX rotation below
            _cs = cache_seqlens
            if isinstance(_cs, mx.array):
                _cs = int(_cs.tolist()) if _cs.ndim == 0 else _cs
            if not isinstance(_cs, int):
                # per-batch: use the first offset (single decode step assumed)
                _cs = int(list(_cs)[0]) if hasattr(_cs, '__iter__') else int(_cs)
            q_att = _apply_rope_mlx(q, rotary_cos, rotary_sin,
                                    offset=_cs, interleaved=interleaved,
                                    rotary_dim=rotary_dim)

        if alibi_slopes is not None:
            raise ValueError(
                "flash_attention_kvcache: alibi_slopes is not supported in paged mode."
            )

        return flash_attention_paged(
            q_att, k_cache, v_cache, block_table, seq_lens,
            scale=scale, causal=causal, block_size=block_size, stream=stream,
        )

    # ----------------------------------------------------------------
    # DENSE MODE
    # ----------------------------------------------------------------
    if k_cache is None or v_cache is None:
        raise ValueError(
            "flash_attention_kvcache: k_cache and v_cache must be provided in "
            "dense mode (use block_table to enable paged mode)."
        )

    # Track FX-2: cache_batch_idx — select per-request slots from a pool.
    # k_cache / v_cache have shape [pool_size, H_kv, S, D].
    # After indexing: [B, H_kv, S, D] — same as the standard dense layout.
    if cache_batch_idx is not None:
        k_cache = k_cache[cache_batch_idx]
        v_cache = v_cache[cache_batch_idx]

    # RoPE in dense mode: use fused kernel when possible.
    if _has_rope:
        if rotary_cos is None or rotary_sin is None:
            raise ValueError(
                "flash_attention_kvcache: both rotary_cos and rotary_sin "
                "must be provided together."
            )
        return flash_attention_rope(
            q, k_cache, v_cache,
            rotary_cos=rotary_cos, rotary_sin=rotary_sin,
            scale=scale, causal=causal,
            cache_seqlens=cache_seqlens,
            interleaved=interleaved,
            rotary_dim=rotary_dim, stream=stream,
        )

    # All other dense features route through flash_attention.
    return flash_attention(
        q, k_cache, v_cache,
        scale=scale, causal=causal, softcap=softcap,
        alibi_slopes=alibi_slopes,
        window_size=window_size,
        stream=stream,
    )


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
        ``backward="sdpa"`` (default): dense SDPA vjp — correct but O(N×S×D).
        ``backward="sdpa_sparse"``: tiled Python sparse backward — O(nnz·BQ·BK·D).
        ``backward="steel_sparse"``: native Metal sparse backward (fastest for
        low-density masks); requires f16/bf16, D≤128.

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
        "sdpa"          (default) — dense mx.fast.sdpa vjp; correct but O(N×S×D)
        "sdpa_sparse"   — tiled Python sparse backward using saved L;
                          O(nnz × BQ × BK × D), benefits large sparse configs
        "steel_sparse"  — Metal STEEL sparse backward; skips inactive tiles in
                          native Metal kernel (fastest for low-density masks)
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

        if backward == "steel_sparse":
            # Native STEEL sparse backward — Metal kernel skips inactive tiles.
            # IMPORTANT: MLX's autograd recycles GPU buffers for q/k/v during
            # backward.  Custom Metal primitives read those buffers directly and
            # see garbage data.  Forcing a CPU round-trip through numpy gives
            # every tensor a freshly-allocated GPU buffer, avoiding the aliasing.
            from mlx_mfa._ext import mfa_steel_backward_sparse as _sbwd
            mx.eval(q, k, v, mask_uint8, dO, O, L)
            def _to_fresh(a, dtype=None):
                a32 = a.astype(mx.float32) if a.dtype != mx.float32 else a
                return mx.array(_np.array(a32), dtype=dtype or a.dtype)
            q2  = _to_fresh(q);  k2  = _to_fresh(k);  v2  = _to_fresh(v)
            O2  = _to_fresh(O);  L2  = _to_fresh(L, dtype=mx.float32)
            dO2 = _to_fresh(dO)
            mu2 = mx.array(_np.array(mask_uint8), dtype=mx.uint8)
            mx.eval(q2, k2, v2, O2, L2, dO2, mu2)
            dQ, dK, dV = _sbwd(q2, k2, v2, O2, L2, dO2, mu2, scale, causal)
            return dQ, dK, dV, mx.zeros_like(mask_uint8)

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
_rope_compile_cache: dict = {}


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
def _make_mfa_custom(scale: float, causal: bool, softcap: float = 0.0,
                     window_left: int = -1):
    """Return a custom-vjp MFA forward function for the given (scale, causal, softcap, window_left).

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
        if window_left >= 0 or softcap != 0.0:
            # Window or softcap variant: pass all params via mfa_attention_forward.
            O = mfa_attention_forward(q, k, v, scale, causal, softcap, window_left)
        else:
            # Fast path: uses the debug binding that also returns L.
            O, _ = mfa_forward_with_lse(q, k, v, scale, causal)
        return O

    @_impl.vjp
    def _backward(primals, cotangent, output):
        # mx.custom_function vjp signature:
        #   primals   - tuple of all forward inputs (q, k, v)
        #   cotangent - gradient w.r.t. the output O  (i.e. dO)
        #   output    - forward output O (unused; gradients computed fresh)
        q, k, v = primals

        if window_left >= 0:
            # Windowed attention backward: re-run reference SDPA with window mask.
            # Sliding window is mainly used for inference; backward is provided
            # for correctness when gradients are needed during training.
            def _windowed_sdpa(q, k, v):
                N, S = q.shape[2], k.shape[2]
                q_idx = mx.arange(S - N, S, dtype=mx.int32)[:, None]  # [N,1]
                k_idx = mx.arange(S, dtype=mx.int32)[None, :]          # [1,S]
                in_win = k_idx >= q_idx - window_left
                if causal:
                    in_win = in_win & (k_idx <= q_idx)
                mask = mx.where(in_win,
                                mx.zeros((N, S), dtype=q.dtype),
                                mx.full((N, S), float("-inf"), dtype=q.dtype))
                return mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask)
            _, (dQ, dK, dV) = mx.vjp(_windowed_sdpa, [q, k, v], [cotangent])
        elif softcap == 0.0:
            # Route f16/bf16 D≤256 to STEEL backward kernels (GQA supported).
            # D=256 uses D-split kernels (BD_HALF=128) to stay within TGP budget.
            # Gradient checkpointing: re-run forward to recover L.
            # Cost: ~1x forward pass — same as the SDPA re-run below.
            D = q.shape[-1]
            use_steel_bwd = (
                q.dtype in (mx.float16, mx.bfloat16)
                and D <= 512
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
                # Fallback: SDPA backward (f32, D>256, etc.).
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
    window_left: int = -1,
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
    impl = _make_mfa_custom(scale, causal, softcap, window_left)
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


def _fallback_sdpa_with_lse(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool,
) -> tuple:
    """Compute SDPA + logsumexp (log2 domain) via pure-MLX ops.

    Used when ``return_lse=True`` and the MFA extension is unavailable.
    Materialises the full ``[B, H, N, S]`` logit matrix — O(N·S) memory.

    Returns:
        (O [B,H,N,D], L [B,H,N]) — L is in log2 domain:
        ``L[b,h,i] = log2(sum_j 2^{score[b,h,i,j]})`` where
        ``score = scale * q @ k^T`` (with causal masking applied).
    """
    # Compute raw attention scores [B, H, N, S]
    scores = mx.matmul(q.astype(mx.float32),
                       mx.swapaxes(k.astype(mx.float32), -2, -1)) * scale
    if causal:
        N, S = q.shape[2], k.shape[2]
        cmask = mx.triu(
            mx.full((N, S), float("-inf"), dtype=mx.float32),
            k=S - N + 1,
        )
        scores = scores + cmask

    # LSE in log2 domain: L = max + log2(sum(2^(scores - max)))
    max_s = scores.max(axis=-1, keepdims=True)          # [B,H,N,1]
    exp2_s = mx.exp2(scores - max_s)                    # [B,H,N,S]
    lse = max_s.squeeze(-1) + mx.log2(exp2_s.sum(axis=-1))  # [B,H,N]

    # Standard softmax attention output (use built-in for efficiency)
    mask = None
    if causal:
        N, S = q.shape[2], k.shape[2]
        mask = mx.triu(
            mx.full((N, S), float("-inf"), dtype=q.dtype),
            k=S - N + 1,
        )
    O = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return O, lse


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _apply_rope_mlx(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    offset: int = 0,
    interleaved: bool = True,
    rotary_dim: Optional[int] = None,
) -> mx.array:
    """Apply rotary position embeddings to *x* using MLX ops (mx.compile cached).

    Two pairing modes:

    * **interleaved** (LLaMA, default) — pairs are adjacent (2i, 2i+1)::

        x_rot[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
        x_rot[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]

    * **non-interleaved** (GPT-NeoX) — first half and second half::

        x_rot[i]       = x[i]       * cos[i] - x[i+D/2] * sin[i]
        x_rot[i + D/2] = x[i] * sin[i] + x[i+D/2] * cos[i]

    Args:
        x:           ``[B, H, N, D]``
        cos:         ``float32 [max_seq_len, rotary_dim/2]``
        sin:         ``float32 [max_seq_len, rotary_dim/2]``
        offset:      First token position (= cache_seqlens for Q, 0 for K).
        interleaved: True = LLaMA; False = GPT-NeoX.
        rotary_dim:  Number of head-dim elements to rotate (must be even).
                     ``None`` (default) rotates all ``D`` elements.
                     When ``rotary_dim < D`` the first ``rotary_dim`` elements
                     are rotated and the remaining ``D - rotary_dim`` pass
                     through unchanged.

    Returns:
        Rotated tensor, same shape and dtype as *x*.
    """
    D = x.shape[-1]
    rot_dim = rotary_dim if rotary_dim is not None else D

    # Partial RoPE: recursively rotate the first rot_dim elements, concat tail.
    if rot_dim < D:
        x_rot_part = _apply_rope_mlx(
            x[..., :rot_dim], cos, sin, offset, interleaved, rotary_dim=None
        )
        return mx.concatenate([x_rot_part, x[..., rot_dim:]], axis=-1)

    # Full rotation (rot_dim == D):
    # Cache key includes shape, dtype, offset and interleaved flag so
    # mx.compile resolves the branch and scalar slicing at compile time.
    key = (tuple(x.shape), x.dtype, int(offset), bool(interleaved))
    if key not in _rope_compile_cache:
        B, H, N, D_inner = x.shape
        half_D = D_inner // 2
        _off, _inter = int(offset), bool(interleaved)

        if _inter:
            def _impl(x_: mx.array, cos_: mx.array, sin_: mx.array) -> mx.array:
                cos_n = cos_[_off : _off + N, :]
                sin_n = sin_[_off : _off + N, :]
                cos_bc = cos_n[None, None, :, :].astype(x_.dtype)
                sin_bc = sin_n[None, None, :, :].astype(x_.dtype)
                x_pairs = x_.reshape(B, H, N, half_D, 2)
                x0 = x_pairs[..., 0]
                x1 = x_pairs[..., 1]
                x0_rot = x0 * cos_bc - x1 * sin_bc
                x1_rot = x0 * sin_bc + x1 * cos_bc
                return mx.stack([x0_rot, x1_rot], axis=-1).reshape(B, H, N, D_inner)
        else:
            def _impl(x_: mx.array, cos_: mx.array, sin_: mx.array) -> mx.array:
                cos_n = cos_[_off : _off + N, :]
                sin_n = sin_[_off : _off + N, :]
                cos_bc = cos_n[None, None, :, :].astype(x_.dtype)
                sin_bc = sin_n[None, None, :, :].astype(x_.dtype)
                x0 = x_[..., :half_D]
                x1 = x_[..., half_D:]
                return mx.concatenate(
                    [x0 * cos_bc - x1 * sin_bc,
                     x0 * sin_bc + x1 * cos_bc], axis=-1)

        _rope_compile_cache[key] = mx.compile(_impl)

    return _rope_compile_cache[key](x, cos, sin)


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
#           Track EA — Differentiable varlen (mx.custom_function, v0.9.3)
# ---------------------------------------------------------------------------


def _varlen_split_concat(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_q: list,
    cu_k: list,
    scale: float,
    causal: bool,
    block_mask,
    stream,
) -> mx.array:
    """Per-sequence split → flash_attention → concat.  Internal helper."""
    num_seqs = len(cu_q) - 1
    outputs = []
    for i in range(num_seqs):
        q_i = q[:, :, cu_q[i] : cu_q[i + 1], :]
        k_i = k[:, :, cu_k[i] : cu_k[i + 1], :]
        v_i = v[:, :, cu_k[i] : cu_k[i + 1], :]
        if block_mask is not None:
            out_i = flash_attention_sparse(
                q_i, k_i, v_i, block_mask, scale=scale, causal=causal, stream=stream
            )
        else:
            out_i = flash_attention(q_i, k_i, v_i, scale=scale, causal=causal, stream=stream)
        outputs.append(out_i)
    return mx.concatenate(outputs, axis=2)


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

    # Materialise cu_seqlens to Python lists ONCE here — safe to close over.
    # mx.arrays must NOT be used for slicing inside a custom_function backward.
    cu_q = [int(x) for x in cu_seqlens_q.tolist()]
    cu_k_list = [int(x) for x in cu_seqlens_k.tolist()]
    num_seqs = len(cu_q) - 1

    if num_seqs == 0:
        return q  # empty — return as-is

    D = q.shape[-1]

    # ── block_mask: direct split-concat (no STEEL varlen for sparse) ─────────
    if block_mask is not None:
        return _varlen_split_concat(
            q, k, v, cu_q, cu_k_list, scale, causal, block_mask, stream
        )

    # ── Differentiable STEEL varlen path ─────────────────────────────────────
    # Forward: STEEL single-dispatch varlen kernel when conditions are met.
    # Backward: split-concat per-sequence through flash_attention, which
    # has STEEL backward (D≤256 f16/bf16) or SDPA VJP fallback.
    #
    # cu_q / cu_k_list are Python list[int] closed over from the outer scope.
    # They are transparent to MLX autograd — no trace nodes are created.

    @mx.custom_function
    def _varlen_impl(q_, k_, v_):
        if (
            _ext_available()
            and q_.dtype in (mx.float16, mx.bfloat16)
            and D in (64, 128, 256)
        ):
            from mlx_mfa._ext import mfa_attention_varlen_forward as _varlen_fwd

            BQ = 32  # constant for all STEEL block configs (D=64/128/256)
            tile_off = [0]
            for i in range(num_seqs):
                qlen = cu_q[i + 1] - cu_q[i]
                tile_off.append(tile_off[-1] + (qlen + BQ - 1) // BQ)
            tile_arr = mx.array(tile_off, dtype=mx.int32)
            O, _L = _varlen_fwd(
                q_, k_, v_, cu_seqlens_q, cu_seqlens_k, tile_arr, scale, causal
            )
            return O
        # f32 or unsupported D: per-sequence split-concat
        return _varlen_split_concat(
            q_, k_, v_, cu_q, cu_k_list, scale, causal, None, stream
        )

    @_varlen_impl.vjp
    def _varlen_bwd(primals, cotangent, _output):
        q_, k_, v_ = primals
        dO = cotangent
        # Split-concat backward: each sequence goes through flash_attention
        # (which has STEEL backward for f16/bf16 D≤256).
        dQ_parts: list = []
        dK_parts: list = []
        dV_parts: list = []
        for i in range(num_seqs):
            qs, qe = cu_q[i], cu_q[i + 1]
            ks, ke = cu_k_list[i], cu_k_list[i + 1]
            q_i  = q_[:, :, qs:qe, :]
            k_i  = k_[:, :, ks:ke, :]
            v_i  = v_[:, :, ks:ke, :]
            dO_i = dO[:, :, qs:qe, :]
            _, (dq_i, dk_i, dv_i) = mx.vjp(
                lambda qi, ki, vi: flash_attention(
                    qi, ki, vi, scale=scale, causal=causal
                ),
                [q_i, k_i, v_i],
                [dO_i],
            )
            dQ_parts.append(dq_i)
            dK_parts.append(dk_i)
            dV_parts.append(dv_i)
        dQ = mx.concatenate(dQ_parts, axis=2)
        dK = mx.concatenate(dK_parts, axis=2)
        dV = mx.concatenate(dV_parts, axis=2)
        return dQ, dK, dV

    return _varlen_impl(q, k, v)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track BE — Paged KV Cache Phase 1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PagedKVCache:
    """Paged KV cache manager — dual-pool block allocator.

    Manages separate K and V page pools as fixed-size blocks.  Eliminates
    padding waste when batch sequences have different lengths.  Designed to
    integrate directly with :func:`flash_attention_paged` and
    :func:`flash_attention_kvcache` (paged mode).

    Pool layout: ``[num_blocks, block_size, H_kv, D]``

    Example::

        cache = PagedKVCache(num_blocks=256, block_size=16, H=8, D=128)

        # Prefill: append 512 tokens for sequence 0
        cache.append(k_prefill, v_prefill, seq_id=0)   # k: [1, H, 512, D]

        # Decode: append 1 new token per step
        cache.append(k_new, v_new, seq_id=0)            # k: [1, H, 1, D]

        # Attend via unified API
        out = flash_attention_kvcache(
            q, cache.k_pool, cache.v_pool,
            block_table=cache.get_block_table(),
            seq_lens=cache.get_seq_lens(),
            block_size=cache.block_size, causal=True,
        )

        # Or via paged API
        out = flash_attention_paged(
            q, cache.k_pool, cache.v_pool,
            cache.get_block_table(), cache.get_seq_lens(),
            block_size=cache.block_size,
        )

        # Free when done
        cache.free_seq(0)

    Args:
        num_blocks:  Total number of pages in the pool.
        block_size:  Tokens per page (16, 32, or 64 recommended).
        H:           Number of KV heads.
        D:           Head dimension.
        dtype:       MLX dtype for the pool (default ``mx.float16``).

    Note — Performance:
        ``append()`` uses MLX-native concatenation to splice new tokens into
        the pool.  ``mx.eval()`` is called at the end of each ``append()`` to
        materialise the lazy graph and prevent O(N) graph growth during long
        decode loops.  For pools with thousands of blocks a Metal scatter
        kernel would be faster; this implementation targets typical inference
        workloads (B=1, ≤512 blocks).
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

        # MLX-native pool arrays — updated via concatenation in append().
        self._k_pool = mx.zeros((num_blocks, block_size, H, D), dtype=dtype)
        self._v_pool = mx.zeros((num_blocks, block_size, H, D), dtype=dtype)
        mx.eval(self._k_pool, self._v_pool)

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

    # ── pool properties ───────────────────────────────────────────────────

    @property
    def k_pool(self) -> "mx.array":
        """Key page pool ``[num_blocks, block_size, H, D]``."""
        return self._k_pool

    @property
    def v_pool(self) -> "mx.array":
        """Value page pool ``[num_blocks, block_size, H, D]``."""
        return self._v_pool

    # ── public API ────────────────────────────────────────────────────────

    def append(
        self,
        k: "mx.array",
        v: "mx.array",
        seq_id: int = 0,
    ) -> None:
        """Append new K/V tokens for ``seq_id``.

        Uses MLX-native concatenation — no numpy roundtrip.  ``mx.eval()``
        is called at the end to materialise the lazy graph and prevent
        unbounded graph growth during long decode loops.

        Args:
            k:       ``[1, H, T, D]`` new key tokens.
            v:       ``[1, H, T, D]`` new value tokens.
            seq_id:  Sequence identifier (default 0).
        """
        import mlx.core as mx

        # [1, H, T, D] → [T, H, D], cast to pool dtype.
        k_tokens = k[0].transpose([1, 0, 2]).astype(self.dtype)
        v_tokens = v[0].transpose([1, 0, 2]).astype(self.dtype)
        T = k_tokens.shape[0]

        self._ensure_seq(seq_id)

        written = 0
        while written < T:
            blks = self._block_table[seq_id]
            ptr = self._write_ptr[seq_id]

            if ptr == self.block_size:
                blk = self._allocate_block()
                blks.append(blk)
                ptr = 0
                self._write_ptr[seq_id] = 0

            blk_id = blks[-1]
            room = self.block_size - ptr
            chunk = min(room, T - written)

            # Build replacement block: [prefix | new_chunk | suffix] on axis 0.
            parts_k: list = []
            parts_v: list = []
            if ptr > 0:
                parts_k.append(self._k_pool[blk_id, :ptr])
                parts_v.append(self._v_pool[blk_id, :ptr])
            parts_k.append(k_tokens[written : written + chunk])
            parts_v.append(v_tokens[written : written + chunk])
            tail = self.block_size - ptr - chunk
            if tail > 0:
                parts_k.append(self._k_pool[blk_id, ptr + chunk :])
                parts_v.append(self._v_pool[blk_id, ptr + chunk :])

            new_k = mx.concatenate(parts_k, axis=0)[None]  # [1, block_size, H, D]
            new_v = mx.concatenate(parts_v, axis=0)[None]

            # Splice block blk_id into pool.
            self._k_pool = mx.concatenate(
                [self._k_pool[:blk_id], new_k, self._k_pool[blk_id + 1 :]], axis=0
            )
            self._v_pool = mx.concatenate(
                [self._v_pool[:blk_id], new_v, self._v_pool[blk_id + 1 :]], axis=0
            )

            self._write_ptr[seq_id] = ptr + chunk
            written += chunk

        # Materialise the lazy graph — prevents O(N) graph depth in decode loops.
        mx.eval(self._k_pool, self._v_pool)

    def gather(self, seq_id: int = 0) -> "tuple[mx.array, mx.array]":
        """Reconstruct contiguous K, V tensors for ``seq_id``.

        Useful for inspection, debugging, or dense-attention fallback.
        For inference, prefer :func:`flash_attention_paged` or
        :func:`flash_attention_kvcache` which read tiles directly from
        the pool without materialising a full contiguous copy.

        Returns:
            ``(k, v)`` each shaped ``[1, H, S, D]`` where S = tokens written.
        """
        import mlx.core as mx

        blks = self._block_table.get(seq_id, [])
        seqlen = self.seq_lengths.get(seq_id, 0)

        if not blks or seqlen == 0:
            return (
                mx.zeros((1, self.H, 0, self.D), dtype=self.dtype),
                mx.zeros((1, self.H, 0, self.D), dtype=self.dtype),
            )

        # Gather blocks: [num_blks, block_size, H, D] → [S_full, H, D] → trim.
        blk_idx = mx.array(blks, dtype=mx.int32)
        k_flat = self._k_pool[blk_idx].reshape(-1, self.H, self.D)[:seqlen]
        v_flat = self._v_pool[blk_idx].reshape(-1, self.H, self.D)[:seqlen]

        # [S, H, D] → [1, H, S, D]
        k_out = k_flat.transpose([1, 0, 2])[None]
        v_out = v_flat.transpose([1, 0, 2])[None]
        return k_out, v_out

    def get_block_table(
        self,
        seq_ids: "Optional[list[int]]" = None,
    ) -> "mx.array":
        """Block table for given sequences.

        Args:
            seq_ids: Sequences to include (default: all active, sorted by id).

        Returns:
            ``int32 [B, max_blocks_per_seq]`` — unused slots padded with ``-1``.
        """
        import mlx.core as mx

        if seq_ids is None:
            seq_ids = sorted(self._block_table.keys())
        if not seq_ids:
            return mx.zeros((0, 0), dtype=mx.int32)
        max_blks = max(len(self._block_table[s]) for s in seq_ids)
        table = []
        for s in seq_ids:
            blks = self._block_table[s]
            row = blks + [-1] * (max_blks - len(blks))
            table.append(row)
        return mx.array(table, dtype=mx.int32)

    def get_seq_lens(
        self,
        seq_ids: "Optional[list[int]]" = None,
    ) -> "mx.array":
        """Sequence lengths for given sequences.

        Args:
            seq_ids: Sequences to include (default: all active, sorted by id).

        Returns:
            ``int32 [B]`` — token count per sequence.
        """
        import mlx.core as mx

        if seq_ids is None:
            seq_ids = sorted(self._block_table.keys())
        lens = [self.seq_lengths.get(s, 0) for s in seq_ids]
        return mx.array(lens, dtype=mx.int32)

    def block_table_and_seq_lens(
        self,
        seq_ids: "list[int]",
    ) -> "tuple[mx.array, mx.array]":
        """Convenience wrapper: ``(get_block_table(seq_ids), get_seq_lens(seq_ids))``."""
        return self.get_block_table(seq_ids), self.get_seq_lens(seq_ids)

    @property
    def seq_lengths(self) -> "dict[int, int]":
        """Return ``{seq_id: num_tokens_written}`` for all active sequences."""
        return {
            sid: (len(blks) - 1) * self.block_size + self._write_ptr[sid]
            for sid, blks in self._block_table.items()
        }

    def free_seq(self, seq_id: int) -> None:
        """Release all blocks held by ``seq_id`` back to the free list."""
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
    """Paged KV cache attention with Metal gather kernel.

    Gathers K/V from a paged block pool into contiguous tensors via a single
    Metal dispatch (``mfa_paged_kv_gather``), then runs ``flash_attention``.
    Supports autograd: ``dQ`` is computed correctly; ``dK_pages``/``dV_pages``
    are returned as zeros (KV pools are cache buffers, not trainable parameters
    in standard use — use non-paged ``flash_attention`` for end-to-end training).

    Args:
        q:            Query tensor ``[B, H_q, N_q, D]``.
        k_pages:      Key page pool ``[num_blocks, block_size, H_kv, D]``.
        v_pages:      Value page pool ``[num_blocks, block_size, H_kv, D]``.
        block_table:  ``[B, max_blocks_per_seq]`` int32 — logical→physical map.
                      Use ``-1`` to pad unused entries.
        seq_lens:     ``[B]`` int32 — actual KV token count per sequence.
        scale:        Attention scale.  Default ``1/sqrt(D)``.
        causal:       Apply causal mask within each sequence.
        block_size:   Tokens per page (must match pool layout).
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
    H_kv = k_pages.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Materialise index data as Python scalars — transparent to autograd.
    seq_lens_list = [int(x) for x in seq_lens.tolist()]
    block_table_list = block_table.tolist()
    max_kv_len = max(seq_lens_list) if seq_lens_list else 0

    if max_kv_len == 0:
        return mx.zeros((B, H_q, N_q, D), dtype=q.dtype)

    def _gather_contig(k_p: "mx.array", v_p: "mx.array"):
        """Gather pool pages → contiguous [B, H_kv, max_kv_len, D]."""
        if _ext_available() and k_p.dtype in (mx.float16, mx.bfloat16):
            from mlx_mfa._ext import mfa_paged_kv_gather
            K = mfa_paged_kv_gather(k_p, block_table, seq_lens, max_kv_len)
            V = mfa_paged_kv_gather(v_p, block_table, seq_lens, max_kv_len)
            return K, V
        # Python fallback gather (all dtypes).
        K_list, V_list = [], []
        for b in range(B):
            kv_len = seq_lens_list[b]
            table_b = block_table_list[b]
            n_full, rem = divmod(kv_len, block_size)
            slices_k, slices_v = [], []
            for lb in range(n_full):
                phys = int(table_b[lb])
                slices_k.append(k_p[phys])
                slices_v.append(v_p[phys])
            if rem > 0:
                phys = int(table_b[n_full])
                slices_k.append(k_p[phys, :rem])
                slices_v.append(v_p[phys, :rem])
            if slices_k:
                k_seq = mx.concatenate(slices_k, axis=0)  # [kv_len, H_kv, D]
                v_seq = mx.concatenate(slices_v, axis=0)
            else:
                k_seq = mx.zeros([0, H_kv, D], dtype=k_p.dtype)
                v_seq = mx.zeros([0, H_kv, D], dtype=v_p.dtype)
            pad = max_kv_len - k_seq.shape[0]
            if pad > 0:
                k_seq = mx.pad(k_seq, [(0, pad), (0, 0), (0, 0)])
                v_seq = mx.pad(v_seq, [(0, pad), (0, 0), (0, 0)])
            K_list.append(k_seq.transpose(1, 0, 2)[None])  # [1, H_kv, max_kv_len, D]
            V_list.append(v_seq.transpose(1, 0, 2)[None])
        return mx.concatenate(K_list, axis=0), mx.concatenate(V_list, axis=0)

    def _attn_per_seq(q_, K_contig, V_contig):
        """Per-sequence attention using exact kv_len slices (avoids padding leak)."""
        outputs = []
        for b in range(B):
            kv_len = seq_lens_list[b]
            out_b = flash_attention(
                q_[b:b+1],
                K_contig[b:b+1, :, :kv_len, :],
                V_contig[b:b+1, :, :kv_len, :],
                scale=scale, causal=causal, stream=stream)
            outputs.append(out_b)
        return mx.concatenate(outputs, axis=0)

    # ── Paged STEEL fast path (Track FD) ─────────────────────────────────
    # Kernel-level paged KV: K/V tiles read directly from pool via block_table,
    # eliminating the gather→attend round-trip.  Only f16/bf16 + D∈{64,128,256}.
    _USE_PAGED_STEEL = (
        _ext_available()
        and q.dtype in (mx.float16, mx.bfloat16)
        and D in (64, 128, 256)
    )

    # ── Paged Flash Decode path (Track FD-decode) ─────────────────────────
    # For decode steps (N_q ≤ 4, long KV ≥ 256), gather K/V into contiguous
    # tensors first, then route to flash_attention() which activates Flash
    # Decoding (split-KV two-phase) for better GPU parallelism.
    # The gather itself is a single fast Metal dispatch (mfa_paged_kv_gather).
    _USE_PAGED_FLASH_DECODE = (
        _USE_PAGED_STEEL
        and N_q <= 4
        and max_kv_len >= 256
    )
    if _USE_PAGED_FLASH_DECODE:
        K_contig, V_contig = _gather_contig(k_pages, v_pages)
        # Flash Decode is activated inside flash_attention when N≤4 and S≥256.
        # Per-sequence slicing ensures each batch item sees only its kv_len.
        return _attn_per_seq(q, K_contig, V_contig)

    if _USE_PAGED_STEEL:
        from mlx_mfa._ext import mfa_paged_steel_forward as _raw_paged_steel

        @mx.custom_function
        def _paged_steel_impl(q_, k_pages_, v_pages_):
            O, _L = _raw_paged_steel(
                q_, k_pages_, v_pages_, block_table, seq_lens,
                scale=scale, causal=causal,
                window_left=-1, block_size=block_size)
            return O

        @_paged_steel_impl.vjp
        def _paged_steel_bwd(primals, cotangent, _output):
            q_, k_pages_, v_pages_ = primals
            dO = cotangent
            # Backward uses gather+per-seq vjp (same as non-paged path).
            K_contig, V_contig = _gather_contig(k_pages_, v_pages_)
            dQ_parts = []
            for b in range(B):
                kv_len = seq_lens_list[b]
                q_b  = q_[b:b+1]
                K_b  = K_contig[b:b+1, :, :kv_len, :]
                V_b  = V_contig[b:b+1, :, :kv_len, :]
                dO_b = dO[b:b+1]
                _, (dq_b, _dk_b, _dv_b) = mx.vjp(
                    lambda qi, ki, vi: flash_attention(
                        qi, ki, vi, scale=scale, causal=causal),
                    [q_b, K_b, V_b], [dO_b])
                dQ_parts.append(dq_b)
            dQ = mx.concatenate(dQ_parts, axis=0)
            return dQ, mx.zeros_like(k_pages_), mx.zeros_like(v_pages_)

        return _paged_steel_impl(q, k_pages, v_pages)

    @mx.custom_function
    def _paged_impl(q_, k_pages_, v_pages_):
        K_contig, V_contig = _gather_contig(k_pages_, v_pages_)
        return _attn_per_seq(q_, K_contig, V_contig)

    @_paged_impl.vjp
    def _paged_bwd(primals, cotangent, _output):
        q_, k_pages_, v_pages_ = primals
        dO = cotangent
        K_contig, V_contig = _gather_contig(k_pages_, v_pages_)
        dQ_parts = []
        for b in range(B):
            kv_len = seq_lens_list[b]
            q_b = q_[b:b+1]
            K_b = K_contig[b:b+1, :, :kv_len, :]
            V_b = V_contig[b:b+1, :, :kv_len, :]
            dO_b = dO[b:b+1]
            _, (dq_b, _dk_b, _dv_b) = mx.vjp(
                lambda qi, ki, vi: flash_attention(
                    qi, ki, vi, scale=scale, causal=causal),
                [q_b, K_b, V_b], [dO_b])
            dQ_parts.append(dq_b)
        dQ = mx.concatenate(dQ_parts, axis=0)
        # KV pool gradients are zeros: pools are cache buffers, not parameters.
        return dQ, mx.zeros_like(k_pages_), mx.zeros_like(v_pages_)

    return _paged_impl(q, k_pages, v_pages)


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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track EC — Varlen packed tensor formats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def flash_attention_varlen_qkv_packed(
    qkv: "mx.array",
    cu_seqlens_q: "mx.array",
    cu_seqlens_k: "mx.array",
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    num_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    stream: Optional["mx.StreamOrDevice"] = None,
) -> "mx.array":
    """Varlen attention from a fused QKV packed tensor.

    Splits a fused QKV tensor into Q, K, V then dispatches to
    :func:`flash_attention_varlen`.  Supports the same two layouts as
    :func:`flash_attention_qkv_packed`:

    * ``[1, H, total_tokens, 3, D]``  — head-first (preferred)
    * ``[1, total_tokens, 3*H*D]``    — flat concat

    Args:
        qkv:           Packed QKV tensor.
        cu_seqlens_q:  int32 ``[num_seqs+1]`` cumulative query lengths.
        cu_seqlens_k:  int32 ``[num_seqs+1]`` cumulative key lengths.
        max_seqlen_q:  Maximum query sequence length.
        max_seqlen_k:  Maximum key sequence length.
        scale:         Attention scale.  Default ``1/sqrt(D)``.
        causal:        Causal mask.
        num_heads:     Q heads.  Required for flat layout.
        num_kv_heads:  KV heads for GQA.  Default = ``num_heads``.
        stream:        MLX stream/device.

    Returns:
        Output ``[1, H_q, total_tokens, D]``.
    """
    import mlx.core as mx

    ndim = qkv.ndim

    if ndim == 5:
        # [1, H, total_tokens, 3, D]
        _, H_q, total, three, D = qkv.shape
        if three != 3:
            raise ValueError(
                f"flash_attention_varlen_qkv_packed: expected dim 3 == 3, got {three}"
            )
        H_kv = num_kv_heads if num_kv_heads is not None else H_q
        q = qkv[:, :H_q, :, 0, :]    # [1, H_q, total, D]
        k = qkv[:, :H_kv, :, 1, :]   # [1, H_kv, total, D]
        v = qkv[:, :H_kv, :, 2, :]   # [1, H_kv, total, D]

    elif ndim == 3:
        # [1, total_tokens, 3*H*D]
        if num_heads is None:
            raise ValueError(
                "flash_attention_varlen_qkv_packed: num_heads required for "
                "[1, total_tokens, 3*H*D] layout"
            )
        _, total, fused = qkv.shape
        H_q = num_heads
        H_kv = num_kv_heads if num_kv_heads is not None else H_q
        D = fused // (H_q + 2 * H_kv)
        if D * (H_q + 2 * H_kv) != fused:
            raise ValueError(
                f"flash_attention_varlen_qkv_packed: fused dim {fused} not "
                f"divisible by (H_q={H_q} + 2*H_kv={H_kv})"
            )
        q_end = H_q * D
        k_end = q_end + H_kv * D
        q = qkv[..., :q_end].reshape(1, total, H_q, D).transpose(0, 2, 1, 3)
        k = qkv[..., q_end:k_end].reshape(1, total, H_kv, D).transpose(0, 2, 1, 3)
        v = qkv[..., k_end:].reshape(1, total, H_kv, D).transpose(0, 2, 1, 3)

    else:
        raise ValueError(
            f"flash_attention_varlen_qkv_packed: unsupported shape {qkv.shape}. "
            "Expected [1,H,total,3,D] (ndim=5) or [1,total,3*H*D] (ndim=3)."
        )

    return flash_attention_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        scale=scale, causal=causal, stream=stream)


def flash_attention_varlen_kv_packed(
    q: "mx.array",
    kv: "mx.array",
    cu_seqlens_q: "mx.array",
    cu_seqlens_k: "mx.array",
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    num_kv_heads: Optional[int] = None,
    stream: Optional["mx.StreamOrDevice"] = None,
) -> "mx.array":
    """Varlen attention from a fused KV packed tensor.

    Splits a fused KV tensor into K, V then dispatches to
    :func:`flash_attention_varlen`.  Supports the same two layouts as
    :func:`flash_attention_kv_packed`:

    * ``[1, H_kv, total_kv, 2, D]``  — head-first (preferred)
    * ``[1, total_kv, 2*H_kv*D]``    — flat concat

    Args:
        q:             Query ``[1, H_q, total_q, D]``.
        kv:            Fused KV tensor.
        cu_seqlens_q:  int32 ``[num_seqs+1]`` cumulative query lengths.
        cu_seqlens_k:  int32 ``[num_seqs+1]`` cumulative key lengths.
        max_seqlen_q:  Maximum query sequence length.
        max_seqlen_k:  Maximum key sequence length.
        scale:         Attention scale.  Default ``1/sqrt(D)``.
        causal:        Causal mask.
        num_kv_heads:  KV heads.  Required for flat layout.
        stream:        MLX stream/device.

    Returns:
        Output ``[1, H_q, total_q, D]``.
    """
    import mlx.core as mx

    ndim = kv.ndim

    if ndim == 5:
        # [1, H_kv, total_kv, 2, D]
        _, H_kv, total_kv, two, D = kv.shape
        if two != 2:
            raise ValueError(
                f"flash_attention_varlen_kv_packed: expected dim 3 == 2, got {two}"
            )
        k = kv[:, :, :, 0, :]   # [1, H_kv, total_kv, D]
        v = kv[:, :, :, 1, :]   # [1, H_kv, total_kv, D]

    elif ndim == 3:
        # [1, total_kv, 2*H_kv*D]
        if num_kv_heads is None:
            raise ValueError(
                "flash_attention_varlen_kv_packed: num_kv_heads required for "
                "[1, total_kv, 2*H_kv*D] layout"
            )
        _, total_kv, fused = kv.shape
        H_kv = num_kv_heads
        D = fused // (2 * H_kv)
        if D * 2 * H_kv != fused:
            raise ValueError(
                f"flash_attention_varlen_kv_packed: fused dim {fused} not "
                f"divisible by 2*H_kv={H_kv}"
            )
        k = kv[..., :H_kv * D].reshape(1, total_kv, H_kv, D).transpose(0, 2, 1, 3)
        v = kv[..., H_kv * D:].reshape(1, total_kv, H_kv, D).transpose(0, 2, 1, 3)

    else:
        raise ValueError(
            f"flash_attention_varlen_kv_packed: unsupported shape {kv.shape}. "
            "Expected [1,H_kv,total_kv,2,D] (ndim=5) or [1,total_kv,2*H_kv*D] (ndim=3)."
        )

    return flash_attention_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        scale=scale, causal=causal, stream=stream)
