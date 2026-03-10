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

# Module-level caches (avoid repeated import probes / set allocations per call)
_ext_avail_cached: Optional[bool] = None
_sage_avail_cached: Optional[bool] = None
_VALID_BACKENDS: frozenset = frozenset({"auto", "mfa", "sdpa"})


class DispatchPolicy:
    """Backend selection constants for :func:`flash_attention`.

    Pass one of these string constants as the ``backend=`` argument to
    :func:`flash_attention` to explicitly control GPU kernel routing.

    Attributes:
        AUTO: ``"auto"`` — use the MFA Metal kernel when conditions are met
              (supported head_dim / dtype / no dropout), fall back to
              ``mx.fast.scaled_dot_product_attention`` otherwise.
        MFA:  ``"mfa"``  — force the MFA Metal kernel; raise ``RuntimeError``
              if the C++ extension is unavailable or the config is unsupported.
        SDPA: ``"sdpa"`` — always use ``mx.fast.scaled_dot_product_attention``;
              useful for correctness comparisons and CI without a Metal GPU.

    Example::

        from mlx_mfa import flash_attention, DispatchPolicy
        out = flash_attention(q, k, v, backend=DispatchPolicy.MFA)
        ref = flash_attention(q, k, v, backend=DispatchPolicy.SDPA)
    """

    AUTO: str = "auto"
    MFA:  str = "mfa"
    SDPA: str = "sdpa"

# Optional C++ scatter primitive for O(1) paged KV pool writes (Phase 4-C.1+E.2).
try:
    from mlx_mfa._ext import mfa_scatter_kv as _mfa_scatter_kv_cpp
    _USE_SCATTER_KV = True
except ImportError:
    _USE_SCATTER_KV = False

# ---------------------------------------------------------------------------
# Phase 1.1 — dispatch overhead guard: cache is_m3_plus once at module load.
# get_device_info() makes a C++ Metal API call; calling it per-attention-op
# adds ~5% overhead at sub-millisecond workloads.
# ---------------------------------------------------------------------------
_cached_is_m3_plus: "bool | None" = None


def _get_is_m3_plus_cached() -> bool:
    """Return cached is_m3_plus to avoid repeated MTLDevice queries."""
    global _cached_is_m3_plus
    if _cached_is_m3_plus is None:
        info = get_device_info()
        _cached_is_m3_plus = bool(info.get("is_m3_plus", False))
    return _cached_is_m3_plus


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
    - ``head_dim`` is in ``{64, 128, 256, 512}``
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
            query that are visible; ``right`` is the number of tokens to the
            right.  Use ``-1`` to disable either side.  Both sides are
            natively supported by the STEEL kernel: tiles entirely outside
            the window are skipped and per-element masking handles boundary
            tiles.  Pass ``None`` (default) to disable the feature entirely.
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
            the call always falls back to
            ``mx.fast.scaled_dot_product_attention`` (which accepts it as the
            ``mask`` argument).  **This is an intentional architectural
            decision**: the MFA Metal kernel uses fused online softmax with no
            generic additive-bias buffer; adding one would require a separate
            pre-pass and negate the bandwidth savings.  Use ``alibi_slopes``
            for relative-position biases (handled natively in Metal).
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
    _mfa_capable = _can_use_mfa(q, head_dim) and not v_dim_mismatch
    if backend == "mfa" and not _mfa_capable:
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

    # Phase 1.1 — smart dispatch: only activate MFA when it is expected to be
    # faster than SDPA based on empirical crossover thresholds.
    # Window-size and sparse are handled inside should_use_mfa (always MFA).
    if _mfa_capable and backend == "auto":
        from mlx_mfa.dispatch_policy import should_use_mfa as _should_mfa
        # Mixed-dtype inputs (q f32 + k/v f16) bypass smart dispatch: MFA handles
        # the cast internally, but mx.fast.sdpa produces NaN on mixed dtypes.
        _mixed_dtype = (k.dtype != q.dtype or v.dtype != q.dtype)
        # Use cached is_m3_plus to avoid per-call MTLDevice query overhead.
        use_mfa = _mixed_dtype or _should_mfa(
            head_dim, q.shape[2], causal,
            _get_is_m3_plus_cached(),
            window_size=window_size,
            sparse=False,
            backend=backend,
        )
    else:
        use_mfa = _mfa_capable  # backend='mfa' forces True; not capable → False

    if not use_mfa:
        if softcap != 0.0:
            return _softcap_sdpa_ref(q, k, v, scale, causal, softcap)
        if alibi_slopes is not None:
            return _alibi_sdpa_ref(q, k, v, alibi_slopes, scale, causal)
        # return_lse: when MFA-capable, always route to MFA regardless of shape
        # dispatch (the kernel returns LSE for free; Python fallback has broken
        # mx.exp2 in some MLX versions).  When not capable: use the Python path.
        if return_lse and _mfa_capable:
            use_mfa = True  # fall through to MFA path below
        elif return_lse:
            return _fallback_sdpa_with_lse(q, k, v, scale, causal)
        else:
            return _fallback_sdpa(q, k, v, scale, causal, stream)

    # ALiBi requires f16/bf16 for the Metal kernel (f32 has no STEEL ALiBi).
    if alibi_slopes is not None:
        if q.dtype == mx.float32:
            return _alibi_sdpa_ref(q, k, v, alibi_slopes, scale, causal)
        return _mfa_alibi_forward(q, k, v, alibi_slopes, scale, causal)

    # Convert window_size=(left, right) → window_left / window_right for the STEEL kernel.
    # Both f16 and bf16 support native left+right window masking.
    # f32 falls back to masked SDPA (no native kernel support for f32).
    window_left = -1
    window_right = -1
    if window_size is not None:
        wl = window_size[0]
        wr = window_size[1] if len(window_size) > 1 else -1
        if q.dtype != mx.float32 and (wl >= 0 or wr >= 0):
            # Native STEEL kernel path: both sides supported.
            if wl >= 0:
                window_left = wl
            if wr >= 0:
                window_right = wr
        else:
            # f32 or both disabled: windowed SDPA fallback.
            N, S = q.shape[2], k.shape[2]
            wl_eff = max(wl, 0) if wl >= 0 else S
            wr_eff = max(wr, 0) if wr >= 0 else S
            q_idx = mx.arange(S - N, S, dtype=mx.int32)[:, None]
            k_idx = mx.arange(S, dtype=mx.int32)[None, :]
            in_win = (k_idx >= q_idx - wl_eff) & (k_idx <= q_idx + wr_eff)
            if causal:
                in_win = in_win & (k_idx <= q_idx)
            mask = mx.where(in_win,
                            mx.zeros((N, S), dtype=q.dtype),
                            mx.full((N, S), float("-inf"), dtype=q.dtype))
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=mask)

    # Track FX-1: return_lse — use mfa_forward_with_lse to get L for free.
    # D.5: contiguity is now enforced inside mfa_forward_with_lse C++ binding.
    if return_lse:
        from mlx_mfa._ext import mfa_forward_with_lse
        O, L = mfa_forward_with_lse(q, k, v, scale, causal)
        return O, L

    return _mfa_forward(q, k, v, scale, causal, softcap, window_left, window_right, stream)


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
    import numpy as _np  # cold path: rope table generation, not per-step

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track JB — Unified RoPE entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def flash_attention_rope_unified(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rotary_cos: Optional[mx.array] = None,
    rotary_sin: Optional[mx.array] = None,
    *,
    # Cache: when provided, k/v are treated as NEW tokens to append
    k_cache: Optional[mx.array] = None,
    v_cache: Optional[mx.array] = None,
    # Paged KV (mutually exclusive with dense k_cache)
    block_table: Optional[mx.array] = None,
    seq_lens: Optional[mx.array] = None,
    block_size: int = 16,
    # Attention hyper-parameters
    scale: Optional[float] = None,
    causal: bool = True,
    # RoPE parameters
    cache_seqlens: Union[int, "mx.array", Sequence[int]] = 0,
    k_offset: Optional[int] = None,
    interleaved: bool = True,
    rotary_dim: Optional[int] = None,
    rope_3d: Optional[dict] = None,
    # Output
    return_updated_cache: bool = False,
    stream: Optional[mx.Stream] = None,
) -> Union[mx.array, tuple]:
    """Unified RoPE-fused attention — single entry point for all RoPE modes.

    Dispatches automatically to the right sub-path based on which cache
    parameters are provided.  All four modes share the same Q/K rotation
    logic via :func:`_apply_rope_to_qk`.

    **Mode selection**:

    * **Standalone** (``k_cache=None``, ``block_table=None``):
      ``k``/``v`` are the full key/value sequences.  Q is rotated at
      ``[cache_seqlens, cache_seqlens + N_q)``;  K is rotated starting at
      ``k_offset`` (defaults to 0).  Equivalent to :func:`flash_attention_rope`.

    * **Cache-append** (``k_cache`` or ``v_cache`` provided):
      ``k``/``v`` are the *new* tokens to append.  Both Q and ``k`` are rotated
      at ``[cache_seqlens, …)``, i.e. ``k_offset`` defaults to ``cache_seqlens``.
      The new rotated ``k`` and ``v`` are concatenated onto the cache before
      attention.  Returns ``(output, k_updated, v_updated)`` when
      ``return_updated_cache=True`` (default when ``k_cache`` is provided).

    * **Cache-consume** (``k_cache`` provided, ``k`` is ``None`` / empty):
      Q is rotated; the full cache is attended without appending.  Set
      ``return_updated_cache=False`` explicitly.

    * **Paged** (``block_table`` provided):
      Q is rotated at ``[cache_seqlens, …)``;  K/V are read from the paged
      pool via :func:`flash_attention_paged`.  ``k``/``v`` args are ignored.

    **3D RoPE**: pass ``rope_3d`` dict instead of ``rotary_cos``/``rotary_sin``
    (mutually exclusive).  Tables are built automatically.

    Args:
        q:             Query ``[B, H_q, N_q, D]``.
        k:             Key tensor.  Meaning depends on mode:
                       standalone → full ``[B, H_kv, S, D]``;
                       cache-append → new tokens ``[B, H_kv, N_new, D]``;
                       paged → ignored (pass ``k_pages`` as ``v`` for paged).
        v:             Value tensor (same conventions as ``k``).
        rotary_cos:    ``float32 [max_seq_len, D/2]`` cosine table (1D RoPE).
        rotary_sin:    ``float32 [max_seq_len, D/2]`` sine table (1D RoPE).
        k_cache:       Past key cache ``[B, H_kv, past_len, D]`` (pre-rotated).
                       ``None`` for standalone or first step.
        v_cache:       Past value cache.
        block_table:   ``[B, max_blocks]`` int32 — triggers paged mode.
        seq_lens:      ``[B]`` int32 — per-sequence KV lengths (paged mode).
        block_size:    Page size (paged mode only).
        scale:         Attention scale; defaults to ``1/sqrt(D)``.
        causal:        Causal masking (default ``True``).
        cache_seqlens: Q rotation offset = current cache length.  Can be a
                       per-batch int array.
        k_offset:      K rotation start position.  Defaults to ``0`` in
                       standalone mode and ``cache_seqlens`` in cache mode.
        interleaved:   ``True`` = LLaMA; ``False`` = GPT-NeoX split-halves.
        rotary_dim:    Rotate only the first ``rotary_dim`` head-dim elements.
        rope_3d:       3D RoPE config dict (mutually exclusive with cos/sin).
        return_updated_cache: Return ``(output, k_updated, v_updated)`` tuple.
                       Defaults to ``True`` when ``k_cache`` is provided,
                       ``False`` otherwise.
        stream:        MLX stream.

    Returns:
        When ``return_updated_cache`` is ``False``:
            Output ``[B, H_q, N_q, D]``.
        When ``return_updated_cache`` is ``True``:
            3-tuple ``(output, k_cache_updated, v_cache_updated)``.
    """
    # ── 0. Validate ───────────────────────────────────────────────────────
    if q.ndim != 4:
        raise ValueError(
            "flash_attention_rope_unified expects 4-D tensors "
            f"[B, H, N, D]. Got q={q.ndim}D."
        )

    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # _ret_cache: whether to return (out, k_updated, v_updated).
    # _cache_mode: whether k/v are *new* tokens (rather than full sequences).
    # Cache-mode is triggered either by k_cache being provided OR by the caller
    # explicitly requesting the updated cache (e.g. first-step append).
    _ret_cache = bool(return_updated_cache)
    _cache_mode = (k_cache is not None) or _ret_cache

    # ── 1. 3D RoPE table construction ─────────────────────────────────────
    if rope_3d is not None:
        if rotary_cos is not None or rotary_sin is not None:
            raise ValueError(
                "rope_3d and rotary_cos/rotary_sin are mutually exclusive."
            )
        grid_h = rope_3d["grid_h"]
        grid_w = rope_3d["grid_w"]
        num_frames = rope_3d.get("num_frames", 1)
        rotary_cos, rotary_sin = make_rope_3d_tables(
            grid_h, grid_w, num_frames,
            d_h=rope_3d.get("d_h"),
            d_w=rope_3d.get("d_w"),
            d_t=rope_3d.get("d_t"),
            head_dim=head_dim,
            theta=rope_3d.get("theta", 10000.0),
        )
        cache_seqlens = 0

    if rotary_cos is None or rotary_sin is None:
        raise ValueError(
            "flash_attention_rope_unified requires rotary_cos/rotary_sin "
            "or rope_3d."
        )

    # ── 2. Per-batch cache_seqlens dispatch ───────────────────────────────
    if not isinstance(cache_seqlens, int):
        if isinstance(cache_seqlens, mx.array):
            cs_list = [int(x) for x in cache_seqlens.tolist()]  # GPU sync: per-batch RoPE routing
        else:
            cs_list = [int(x) for x in cache_seqlens]
        B = q.shape[0]
        if len(cs_list) != B:
            raise ValueError(
                f"cache_seqlens length {len(cs_list)} must equal B={B}"
            )
        # H.4: All offsets identical → skip per-batch loop; one batched call.
        # C.2 partial fix: full Metal per-batch rope_q_base deferred to v1.3.0.
        if len(set(cs_list)) == 1:
            return flash_attention_rope_unified(
                q, k, v, rotary_cos, rotary_sin,
                k_cache=k_cache, v_cache=v_cache,
                block_table=block_table, seq_lens=seq_lens,
                block_size=block_size,
                scale=scale, causal=causal,
                cache_seqlens=cs_list[0], k_offset=k_offset,
                interleaved=interleaved, rotary_dim=rotary_dim,
                return_updated_cache=_ret_cache, stream=stream,
            )
        chunks_out, chunks_k, chunks_v = [], [], []
        for b, cs in enumerate(cs_list):
            kc_b = k_cache[b:b+1] if k_cache is not None else None
            vc_b = v_cache[b:b+1] if v_cache is not None else None
            result = flash_attention_rope_unified(
                q[b:b+1], k[b:b+1], v[b:b+1],
                rotary_cos, rotary_sin,
                k_cache=kc_b, v_cache=vc_b,
                block_table=block_table[b:b+1] if block_table is not None else None,
                seq_lens=seq_lens[b:b+1] if seq_lens is not None else None,
                block_size=block_size,
                scale=scale, causal=causal,
                cache_seqlens=cs, k_offset=k_offset,
                interleaved=interleaved, rotary_dim=rotary_dim,
                return_updated_cache=_ret_cache, stream=stream,
            )
            if _ret_cache:
                chunks_out.append(result[0])
                chunks_k.append(result[1])
                chunks_v.append(result[2])
            else:
                chunks_out.append(result)
        out_cat = mx.concatenate(chunks_out, axis=0)
        if _ret_cache:
            return (out_cat,
                    mx.concatenate(chunks_k, axis=0),
                    mx.concatenate(chunks_v, axis=0))
        return out_cat

    # Single-batch path from here.
    cs = int(cache_seqlens) if isinstance(cache_seqlens, (int, float)) else cache_seqlens

    # ── 3. Paged mode ─────────────────────────────────────────────────────
    if block_table is not None:
        if seq_lens is None:
            raise ValueError("seq_lens is required in paged mode.")
        # J.1: both branches were identical; unconditional call.
        q_rot, _ = _apply_rope_to_qk(
            q, k, rotary_cos, rotary_sin,
            q_offset=cs, k_offset=cs,
            interleaved=interleaved, rotary_dim=rotary_dim,
        )
        out = flash_attention_paged(
            q_rot, k, v, block_table, seq_lens,
            scale=scale, causal=causal, block_size=block_size, stream=stream,
        )
        if _ret_cache:
            return out, k, v
        return out

    # ── 4. Determine K rotation offset ────────────────────────────────────
    # In cache mode K starts at the same position as Q (new tokens).
    # In standalone mode K starts at 0 (full sequence from the beginning).
    if k_offset is None:
        _k_off = cs if _cache_mode else 0
    else:
        _k_off = k_offset

    # ── 5. Rotate Q and K, then attend ────────────────────────────────────
    _partial_rope = rotary_dim is not None and rotary_dim < head_dim

    def _make_kv_full(k_rot):
        """Concat rotated k/v onto cache (or return as-is for first step)."""
        if k_cache is not None:
            return (mx.concatenate([k_cache, k_rot], axis=2),
                    mx.concatenate([v_cache, v], axis=2))
        return k_rot, v

    if not _can_use_mfa(q, head_dim) or q.dtype == mx.float32 or _partial_rope:
        q_rot, k_rot = _apply_rope_to_qk(
            q, k, rotary_cos, rotary_sin,
            q_offset=cs, k_offset=_k_off,
            interleaved=interleaved, rotary_dim=rotary_dim,
        )
        k_full, v_full = _make_kv_full(k_rot) if _cache_mode else (k_rot, v)
        out = flash_attention(q_rot, k_full, v_full, scale=scale, causal=causal,
                              stream=stream)
        if _ret_cache:
            return out, k_full, v_full
        return out

    # STEEL / MFA fast path.
    if _cache_mode:
        q_rot, k_new_rot = _apply_rope_to_qk(
            q, k, rotary_cos, rotary_sin,
            q_offset=cs, k_offset=_k_off,
            interleaved=interleaved,
        )
        k_full, v_full = _make_kv_full(k_new_rot)
        out = flash_attention(q_rot, k_full, v_full, scale=scale, causal=causal,
                              stream=stream)
        if _ret_cache:
            return out, k_full, v_full
        return out
    else:
        # Standalone: use the in-kernel MFA RoPE path (K rotated from pos 0).
        return _mfa_rope_forward(q, k, v, rotary_cos, rotary_sin,
                                 scale, causal, cs, interleaved)


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
    # Thin wrapper — full logic lives in flash_attention_rope_unified.
    return flash_attention_rope_unified(
        q, k, v, rotary_cos, rotary_sin,
        k_cache=None, v_cache=None,
        scale=scale, causal=causal,
        cache_seqlens=cache_seqlens, k_offset=0,
        interleaved=interleaved, rotary_dim=rotary_dim, rope_3d=rope_3d,
        return_updated_cache=False, stream=stream,
    )


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

        - ``"device_name"`` (str | None): MTLDevice name, e.g. ``"Apple M1 Max"``.
        - ``"gpu_family_gen"`` (int | None): Apple GPU family generation number.
          13 = M1, 14 = M2, 15 = M3, 16 = M4.
        - ``"is_m3_plus"`` (bool | None): True for M3/M4 (uses different block
          params and ``preferAsyncCache`` vs ``preferAsyncLoad``).
        - ``"chip_name"`` (str | None): Inferred chip family, e.g. ``"M1"``.
        - ``"gpu_cores"`` (int | None): Estimated physical GPU core count, parsed
          from the device name.  Correct per variant: M1 Max=32, M1=8, M2 Max=38.
          Falls back to conservative gen-based estimate for unknown devices.
        - ``"extension_available"`` (bool): Whether the C++ extension loaded.

    Example::

        from mlx_mfa import get_device_info
        info = get_device_info()
        print(info["device_name"])  # "Apple M1 Max"
        print(info["chip_name"])    # "M1"
        print(info["gpu_cores"])    # 32
    """
    if not _ext_available():
        return {
            "device_name": None,
            "gpu_family_gen": None,
            "is_m3_plus": None,
            "is_m5_plus": None,
            "chip_name": None,
            "gpu_cores": None,
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
        "gpu_cores":           raw.get("gpu_cores"),
        "extension_available": True,
    }


def get_supported_configs() -> dict:
    """Return the full feature matrix for this build of mlx-mfa.

    Returns:
        Dictionary with keys:

        - ``"head_dims"``: frozenset of supported integer head dimensions.
        - ``"dtypes"``: frozenset of supported MLX dtype values.
        - ``"extension_available"``: bool — whether the C++ extension loaded.
        - ``"features"``: dict mapping feature name → bool.
        - ``"kernel_types"``: int — number of distinct Metal kernel variants
          compiled into the extension (0 when extension not available).

    Example::

        from mlx_mfa import get_supported_configs
        cfg = get_supported_configs()
        print(cfg["head_dims"])          # frozenset({64, 128, 256, 512})
        print(cfg["features"]["rope"])   # True
    """
    ext = _ext_available()
    features = {
        # --- attention variants ---
        "causal":               True,
        "gqa":                  True,   # native GQA without KV expansion
        "block_sparse":         True,
        "sliding_window":       True,   # both left and right sides supported
        "rope":                 True,
        "paged_kv":             True,
        "varlen":               True,
        "flash_decode":         True,   # split-KV decode for short queries
        # --- score modifiers ---
        "alibi":                True,
        "softcap":              True,
        "attn_bias":            True,
        # --- API knobs ---
        "backend_select":       True,   # "auto" | "mfa" | "sdpa"
        "dropout":              True,   # SDPA fallback only
        "return_lse":           True,
        # --- backward ---
        "native_backward":      "ext",  # STEEL backward kernels active for f16/bf16 D≤512
        "sparse_backward":      True,   # tiled FA-2 sparse backward
        # --- hardware routing ---
        "m3_routing":           True,   # M3+ block config (gen ≥ 15)
        "m5_stub":              True,   # M5 detection stub (gen ≥ 17)
        # --- extended API ---
        "kvcache_rope_append":  True,
        "packed_api":           True,   # qkv_packed / kv_packed variants
        "sage_attention":           ext,    # int8 Q/K quantized attention (Track KB/KC)
        "sage_attention_kvcache":   ext,    # decode variant (N_q != N_k, Track LA)
        "sage_inference_context":   ext,    # stateful sage decode wrapper (Track LA)
        "warmup_kernels":           True,   # pre-compile Metal shaders before first use
        # --- dtype / dim ---
        "bfloat16":             True,
        "float16":              True,
        "d512":                 True,
    }
    # 16 distinct Metal kernel types (0–15):
    #   AttentionForward/BwdDQ/BwdDKV (ccv legacy),
    #   SteelForward, FlashDecodePartial/Reduce,
    #   SteelBackwardDQ/DKV, SteelVarlenForward,
    #   PagedKVGather, PagedSteelForward, SageForward,
    #   QuantizePerBlock, ScatterKV, SmoothQuantizeMean/K
    kernel_types = 16 if ext else 0
    return {
        "head_dims":           frozenset(_MFA_SUPPORTED_HDIMS),
        "dtypes":              frozenset(_MFA_SUPPORTED_DTYPES),
        "extension_available": ext,
        "features":            features,
        "kernel_types":        kernel_types,
    }


def warmup_kernels(
    head_dims: Optional[list] = None,
    dtypes: Optional[list] = None,
    causal: bool = True,
) -> None:
    """Pre-compile Metal shaders for the specified configurations.

    Dispatches small (N=BQ) forward passes to trigger JIT shader compilation
    before the first real attention call.  Without warmup, the first call for
    each new (D, dtype, causal) combination incurs ~100–300 ms of Metal shader
    compilation.  After warmup, all subsequent calls reuse cached pipelines.

    No-op when the C++ extension is unavailable.

    Args:
        head_dims: Head dimensions to warm up (default: ``[64, 128]``).
        dtypes:    MLX dtypes to warm up (default: ``[mx.float16]``).
        causal:    Whether to compile causal variants (default: ``True``).

    Example::

        from mlx_mfa import warmup_kernels
        warmup_kernels(head_dims=[64, 128, 256], dtypes=[mx.float16])
        # Now flash_attention() has no first-call latency.
    """
    if not _ext_available():
        return

    if head_dims is None:
        head_dims = [64, 128]
    if dtypes is None:
        dtypes = [mx.float16]

    for D in head_dims:
        for dtype in dtypes:
            # Use BQ (32) — minimum viable tile for STEEL kernel.
            N = 32
            q = mx.zeros([1, 1, N, D], dtype=dtype)
            k = mx.zeros([1, 1, N, D], dtype=dtype)
            v = mx.zeros([1, 1, N, D], dtype=dtype)
            out = flash_attention(q, k, v, scale=1.0 / math.sqrt(D), causal=causal)
            mx.eval(out)


# ---------------------------------------------------------------------------
# SageAttention — quantized Q/K forward pass (Track KC)
# ---------------------------------------------------------------------------

def sage_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    causal: bool = False,
    apply_smooth_k: bool = True,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Compute attention with int8-quantized Q and K (SageAttention style).

    Reduces Q/K device→threadgroup memory traffic by 2× vs fp16 by loading
    Q and K as int8 and dequantizing to fp16 inside the Metal kernel.
    V is always fp16/bf16 (P@V is memory-access-bound only at very large D).

    Speedup is meaningful for long sequences (S ≥ 2048) where memory
    bandwidth dominates arithmetic throughput.

    When the MFA extension is not available, falls back to standard
    ``flash_attention`` (fp16 SDPA) with a runtime warning.

    Args:
        q: Query tensor ``[B, H, N, D]``.  fp16 or bf16.
        k: Key tensor ``[B, H_kv, S, D]``.  fp16 or bf16.  GQA: ``H_kv``
           must divide ``H``.
        v: Value tensor ``[B, H_kv, S, D]``.  fp16 or bf16.
        scale: Attention scale.  Defaults to ``1 / sqrt(D)``.
        causal: Whether to apply causal masking.
        apply_smooth_k: When ``True`` (default), subtracts the per-channel
            mean of K before quantizing (SageAttention K-smoothing) and
            applies an approximate output correction term to account for it.
            This dramatically reduces quantization error at negligible cost
            (one extra ``mx.mean`` + ``mx.sum`` over V).
            Set to ``False`` to skip smoothing (faster but less accurate).
        stream: MLX stream for async execution.

    Returns:
        ``[B, H, N, D]`` attention output in the same dtype as ``q``.

    Note:
        This function does **not** support autograd.  It is inference-only.
        For training, use ``flash_attention`` which uses fp16 STEEL kernels.

    Example::

        import mlx.core as mx
        from mlx_mfa import sage_attention

        q = mx.random.normal([1, 8, 2048, 128]).astype(mx.float16)
        k = mx.random.normal([1, 8, 2048, 128]).astype(mx.float16)
        v = mx.random.normal([1, 8, 2048, 128]).astype(mx.float16)
        out = sage_attention(q, k, v, causal=True)  # [1, 8, 2048, 128]
    """
    from mlx_mfa.quantize import (
        quantize_per_block,
        sage_block_sizes,
        smooth_k as _smooth_k,
    )

    D = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    BQ, BK = sage_block_sizes(D)

    # Check extension availability (cached)
    global _sage_avail_cached
    if _sage_avail_cached is None:
        try:
            from mlx_mfa._ext import mfa_sage_forward  # noqa: F401
            _sage_avail_cached = True
        except ImportError:
            _sage_avail_cached = False
    _ext_ok = _sage_avail_cached
    if _ext_ok:
        from mlx_mfa._ext import mfa_sage_forward as _sage_fwd  # noqa: F811

    if not _ext_ok:
        import warnings
        warnings.warn(
            "sage_attention: MFA extension not available; "
            "falling back to fp16 flash_attention.",
            RuntimeWarning,
            stacklevel=2,
        )
        return flash_attention(q, k, v, scale=scale, causal=causal, stream=stream)

    # K smoothing: subtract per-channel mean to reduce int8 quantization error.
    # Mathematical note: subtracting a constant k_mean from all key positions
    # adds a query-specific scalar bias to every attention score for position i
    # (bias_i = q_i · k_mean * scale, independent of j). Since bias_i cancels
    # in the softmax ratio, the output is identical to unsmoothed attention.
    # No output correction is needed; smooth_k purely improves int8 precision.
    #
    # Phase 1.1: use fused smooth+quantize kernel when available — eliminates
    # intermediate K_smooth fp16 tensor and reduces dispatch count 3 → 2.
    _fused_sq = None
    if apply_smooth_k:
        try:
            from mlx_mfa._ext import mfa_smooth_quantize_k as _fused_sq
        except ImportError:
            _fused_sq = None

    if _fused_sq is not None:
        # Fused: mean → subtract → absmax → int8 in one C++ primitive.
        k_int8, k_scale, _ = _fused_sq(k, BK)          # _ = k_mean (unused)
    else:
        if apply_smooth_k:
            k_work, _ = _smooth_k(k)
        else:
            k_work = k
        k_int8, k_scale = quantize_per_block(k_work, BK)

    # Quantize Q to int8 per STEEL tile (Q is not smoothed).
    q_int8, q_scale = quantize_per_block(q, BQ)       # q_scale: [B, H, NQ, 1]

    # Kernel expects 3-D scales (squeeze trailing broadcast dim)
    q_scale = q_scale.squeeze(-1)   # [B, H, NQ]
    k_scale = k_scale.squeeze(-1)   # [B, H_kv, NK]

    # Dispatch SageAttention Metal kernel
    O, _ = _sage_fwd(q_int8, k_int8, v, q_scale, k_scale, scale, causal, stream)
    return O


def sage_attention_kvcache(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: Optional[float] = None,
    causal: bool = True,
    apply_smooth_k: bool = True,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Sage attention for KV-cache decode (N_q may differ from N_kv).

    Convenience wrapper around :func:`sage_attention` for the autoregressive
    decode pattern where the query has fewer tokens than the accumulated KV
    cache.  The Metal sage kernel supports ``N_q != N_kv`` natively; this
    function documents and enforces the intended calling convention.

    Args:
        q:              Query  ``[B, H_q, N_new, D]``.  fp16 or bf16.
        k:              Key cache  ``[B, H_kv, seqlen, D]``.  fp16 or bf16.
        v:              Value cache  ``[B, H_kv, seqlen, D]``.  fp16 or bf16.
        scale:          Attention scale.  Defaults to ``1/sqrt(D)``.
        causal:         Whether to apply causal masking (default ``True``).
        apply_smooth_k: K-smoothing before int8 quantization (default ``True``).
        stream:         Optional MLX stream.

    Returns:
        ``[B, H_q, N_new, D]`` attention output.

    Note:
        K is re-quantized on every call (O(seqlen × D) overhead).  For
        production decode loops with seqlen ≥ 4096, pre-quantizing K into a
        :class:`QuantizedKVCache` eliminates this per-step cost.
    """
    return sage_attention(
        q, k, v,
        scale=scale,
        causal=causal,
        apply_smooth_k=apply_smooth_k,
        stream=stream,
    )


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


def _resolve_scalar_seqlens(cache_seqlens) -> int:
    """I.4: Resolve cache_seqlens (int, 0-D array, or iterable) to a plain int.

    Used on the paged-append hot path to obtain the current sequence length for
    RoPE offset calculation, avoiding duplicated type-checking branches.
    """
    if isinstance(cache_seqlens, int):
        return cache_seqlens
    if isinstance(cache_seqlens, mx.array):
        if cache_seqlens.ndim == 0:
            return int(cache_seqlens.item())
        return int(cache_seqlens.reshape(-1)[0].item())
    # Sequence / list / generator
    return int(next(iter(cache_seqlens)))


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
    # Thin wrapper — full logic lives in flash_attention_rope_unified.
    return flash_attention_rope_unified(
        q, k_new, v_new, rotary_cos, rotary_sin,
        k_cache=k_cache, v_cache=v_cache,
        scale=scale, causal=causal,
        cache_seqlens=cache_seqlens,
        interleaved=interleaved,
        return_updated_cache=True, stream=stream,
    )


# ---------------------------------------------------------------------------
# Unified KV-cache API  (Track FA)
# ---------------------------------------------------------------------------

def flash_attention_kvcache(
    q: mx.array,
    k_cache: Optional[mx.array],
    v_cache: Optional[mx.array],
    *,
    # Append mode: new tokens to concat onto the cache before attention
    k_new: Optional[mx.array] = None,
    v_new: Optional[mx.array] = None,
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
    # RoPE: applied to Q (and k_new when k_new is provided)
    rotary_cos: Optional[mx.array] = None,
    rotary_sin: Optional[mx.array] = None,
    cache_seqlens: Union[int, "mx.array", Sequence[int]] = 0,
    interleaved: bool = True,
    # Track FX-3: partial RoPE — rotate only first rotary_dim head-dim elements
    rotary_dim: Optional[int] = None,
    # Track FX-2: continuous batching — map logical batch → cache pool slot
    cache_batch_idx: Optional[mx.array] = None,
    stream: Optional[mx.Stream] = None,
) -> Union[mx.array, tuple]:
    """Unified KV-cache attention — dense and paged modes in one call.

    This function is the recommended entry point for inference with KV caches.
    It consolidates :func:`flash_attention_paged`, :func:`flash_attention_rope`,
    and the append-cache path into one API with full support for RoPE, ALiBi,
    softcap, and sliding-window on both cache modes.

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

    **Cross-attention** (encoder–decoder, Q from decoder, K/V from encoder)::

        # Encoder output: [B, H, S_enc, D] — fixed, not growing
        out = flash_attention_kvcache(
            q_dec, k_enc, v_enc,
            causal=False,   # every decoder token attends all encoder positions
        )

    ``flash_attention_kvcache`` is the recommended entry point for cross-
    attention: the encoder KV tensors (``k_enc``, ``v_enc``) are passed as
    ``k_cache`` / ``v_cache`` without a ``block_table``, so the function uses
    the dense attention path.  Set ``causal=False`` — the decoder can attend
    to all encoder positions.  GQA is supported; H_kv may be less than H_q.

    RoPE is typically **not** used for cross-attention (positional info comes
    from the encoder); if your encoder uses absolute embeddings, omit
    ``rotary_cos``/``rotary_sin``.

    Cross-attention has full autograd support: dQ, dK, dV are all computed
    correctly via the SDPA-based backward.

    **Append mode** (cache concat + attend)::

        out, k_updated, v_updated = flash_attention_kvcache(
            q, k_cache, v_cache,
            k_new=k_new, v_new=v_new, causal=True,
        )

    When ``k_new`` and ``v_new`` are provided, they are appended to the cache
    along the sequence dimension before attention is computed.  If
    ``rotary_cos``/``rotary_sin`` are also provided, ``k_new`` is rotated at
    the ``cache_seqlens`` offset before the append (Q is still rotated
    separately by the attention dispatch).  The function then returns a 3-tuple
    ``(output, k_updated, v_updated)`` so the caller can propagate the enlarged
    cache to the next step.

    Args:
        k_new:          New key tokens ``[B, H_kv, N, D]`` to append.  Must
                        be paired with ``v_new``.  Dense mode only.
        v_new:          New value tokens ``[B, H_kv, N, D]`` to append.

    Returns:
        * ``mx.array`` ``[B, H_q, N_q, D]`` when ``k_new`` is ``None``
          (existing behaviour, backward-compatible).
        * ``tuple (output, k_updated, v_updated)`` when ``k_new`` is provided
          (append mode).

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
    # APPEND MODE: k_new / v_new provided — concat onto cache, then attend
    # ----------------------------------------------------------------
    if k_new is not None or v_new is not None:
        if k_new is None or v_new is None:
            raise ValueError(
                "flash_attention_kvcache: k_new and v_new must both be provided "
                "or both be None."
            )
        if block_table is not None:
            # ── Paged-append path ──────────────────────────────────────────
            # Scatter k_new/v_new tokens into the paged pool, then attend.
            # Cost: O(num_blocks) pool rebuild (MLX is functional — no in-place).
            if seq_lens is None:
                raise ValueError(
                    "flash_attention_kvcache: seq_lens is required for "
                    "paged-append mode (k_new + block_table)."
                )
            if k_cache is None or v_cache is None:
                raise ValueError(
                    "flash_attention_kvcache: k_cache/v_cache (page pool) must "
                    "be provided for paged-append mode."
                )
            if cache_batch_idx is not None:
                raise NotImplementedError(
                    "flash_attention_kvcache: paged-append with cache_batch_idx "
                    "is not currently supported."
                )
            num_blks = k_cache.shape[0]
            blk_sz   = k_cache.shape[1]
            H_kv_p   = k_cache.shape[2]
            D_p      = k_cache.shape[3]
            B_p      = k_new.shape[0]
            N_new_p  = k_new.shape[2]
            # E.3: seq_lens.tolist() GPU sync needed for RoPE offset + fallback loop.
            seq_lens_list_p = [int(x) for x in seq_lens.tolist()]  # GPU sync: RoPE offset
            # E.3: block_table.tolist() deferred to the fallback else-branch;
            # the _USE_SCATTER_KV fast-path uses block_table as an MLX array.

            # Rotate k_new if RoPE requested.
            q_to_att = q
            if rotary_cos is not None:
                if rotary_sin is None:
                    raise ValueError(
                        "flash_attention_kvcache: rotary_sin required with "
                        "rotary_cos in paged-append mode."
                    )
                _cs_p = seq_lens_list_p[0] if len(seq_lens_list_p) == 1 else int(
                    min(seq_lens_list_p))
                q_to_att, k_new = _apply_rope_to_qk(
                    q, k_new, rotary_cos, rotary_sin,
                    q_offset=_cs_p, k_offset=_cs_p,
                    interleaved=interleaved, rotary_dim=rotary_dim,
                )

            if _USE_SCATTER_KV:
                # F.2: Vectorised scatter targets — O(1) MLX ops, no per-token loop.
                # positions[b, t] = seq_lens[b] + t  →  [B_p, N_new_p]
                _kv_l = seq_lens.astype(mx.int32)                     # [B_p]
                _t    = mx.arange(N_new_p, dtype=mx.int32)            # [N_new_p]
                _pos  = _kv_l[:, None] + _t[None, :]                  # [B_p, N_new_p]
                _bi   = (_pos // blk_sz).astype(mx.int32)             # block indices
                _bo   = (_pos %  blk_sz).astype(mx.int32)             # block offsets
                # Gather physical block IDs: block_table [B_p, max_blks]
                _ri   = mx.arange(B_p, dtype=mx.int32)[:, None]      # [B_p, 1]
                _ph   = block_table[_ri, _bi]                         # [B_p, N_new_p]
                # GPU sync: evaluate phys+offsets in one command buffer flush.
                # Subsequent tolist() calls hit already-evaluated data (no extra sync).
                # MLX lacks boolean indexing, so a Python-level valid-slot filter
                # is required (phys < 0 means unallocated block).
                _ph_flat = _ph.reshape(-1)
                mx.eval(_ph_flat, _bo)   # batched sync: ph + offsets together
                ph_list = _ph_flat.tolist()   # GPU sync: filter invalid physical blocks
                bo_list = _bo.reshape(-1).tolist()  # no extra sync — already evaluated
                valid   = [i for i, p in enumerate(ph_list) if p >= 0]
                # k_new [B_p, H_kv, N_new_p, D] → [B_p*N_new_p, H_kv, D]
                _kf = k_new.transpose(0, 2, 1, 3).reshape(B_p * N_new_p, H_kv_p, D_p)
                _vf = v_new.transpose(0, 2, 1, 3).reshape(B_p * N_new_p, H_kv_p, D_p)
                if valid:
                    _idx         = mx.array(valid, dtype=mx.int32)
                    blk_ids_arr  = mx.array([ph_list[i] for i in valid], dtype=mx.int32)
                    blk_offs_arr = mx.array([bo_list[i] for i in valid], dtype=mx.int32)
                    k_pages_new = _mfa_scatter_kv_cpp(
                        k_cache, _kf[_idx], blk_ids_arr, blk_offs_arr)
                    v_pages_new = _mfa_scatter_kv_cpp(
                        v_cache, _vf[_idx], blk_ids_arr, blk_offs_arr)
                else:
                    k_pages_new = k_cache
                    v_pages_new = v_cache
            else:
                # Fallback: Python loop builds per-block update dicts.
                # E.3: block_table.tolist() GPU sync here (fallback only;
                # production path is _USE_SCATTER_KV which avoids this).
                block_table_list_p = block_table.tolist()
                sc_blk_ids: list = []
                sc_blk_offs: list = []
                sc_k_rows: list = []
                sc_v_rows: list = []
                for b in range(B_p):
                    kv_len = seq_lens_list_p[b]
                    tb = block_table_list_p[b]
                    for t in range(N_new_p):
                        pos = kv_len + t
                        blk_idx = pos // blk_sz
                        blk_off = pos % blk_sz
                        phys = int(tb[blk_idx])
                        if phys < 0:
                            continue
                        sc_blk_ids.append(phys)
                        sc_blk_offs.append(blk_off)
                        sc_k_rows.append(k_new[b, :, t, :])
                        sc_v_rows.append(v_new[b, :, t, :])
                # Fallback: MLX-native pool rebuild (extension unavailable).
                if sc_blk_ids:
                    k_updates: dict = {phys: {} for phys in sc_blk_ids}
                    v_updates: dict = {phys: {} for phys in sc_blk_ids}
                    for phys, off, kr, vr in zip(
                            sc_blk_ids, sc_blk_offs, sc_k_rows, sc_v_rows):
                        k_updates[phys][off] = kr
                        v_updates[phys][off] = vr
                    k_blocks, v_blocks = [], []
                    for i in range(num_blks):
                        if i in k_updates:
                            rows_k = [
                                (k_updates[i][j][None] if j in k_updates[i]
                                 else k_cache[i, j:j+1])
                                for j in range(blk_sz)
                            ]
                            rows_v = [
                                (v_updates[i][j][None] if j in v_updates[i]
                                 else v_cache[i, j:j+1])
                                for j in range(blk_sz)
                            ]
                            k_blocks.append(mx.concatenate(rows_k, axis=0))
                            v_blocks.append(mx.concatenate(rows_v, axis=0))
                        else:
                            k_blocks.append(k_cache[i])
                            v_blocks.append(v_cache[i])
                    k_pages_new = mx.stack(k_blocks)
                    v_pages_new = mx.stack(v_blocks)
                else:
                    k_pages_new = k_cache
                    v_pages_new = v_cache
            seq_lens_new = mx.array(
                [sl + N_new_p for sl in seq_lens_list_p], dtype=mx.int32)

            out = flash_attention_paged(
                q_to_att, k_pages_new, v_pages_new, block_table, seq_lens_new,
                scale=scale, causal=causal, block_size=blk_sz, stream=stream,
            )
            return out, k_pages_new, v_pages_new
        if cache_batch_idx is not None:
            raise ValueError(
                "flash_attention_kvcache: k_new/v_new append is not supported "
                "together with cache_batch_idx (continuous-batching pool mode)."
            )

        # I.4: Resolve cache_seqlens to a plain int for the k_new RoPE offset.
        _cs_int: int = _resolve_scalar_seqlens(cache_seqlens)

        # Optionally rotate both q and k_new before appending.
        # Keys are stored pre-rotated in the cache; k_new must be rotated at
        # cache_seqlens before concat.  Q is also rotated here so the
        # subsequent flash_attention call does NOT need RoPE (passing
        # rotary_cos to flash_attention_kvcache with a pre-rotated K would
        # incorrectly re-rotate the entire cache).
        q_to_attend = q
        if rotary_cos is not None:
            if rotary_sin is None:
                raise ValueError(
                    "flash_attention_kvcache: both rotary_cos and rotary_sin "
                    "must be provided together."
                )
            q_to_attend, k_new = _apply_rope_to_qk(
                q, k_new, rotary_cos, rotary_sin,
                q_offset=_cs_int, k_offset=_cs_int,
                interleaved=interleaved, rotary_dim=rotary_dim,
            )

        # Append to cache (or start a fresh cache from k_new / v_new alone).
        if k_cache is not None:
            k_updated = mx.concatenate([k_cache, k_new], axis=2)
            v_updated = mx.concatenate([v_cache, v_new], axis=2)
        else:
            k_updated = k_new
            v_updated = v_new

        # Dispatch attention on the rotated Q and the full updated cache.
        # No RoPE here — rotation was applied explicitly above.
        out = flash_attention(
            q_to_attend, k_updated, v_updated,
            scale=scale, causal=causal, softcap=softcap,
            alibi_slopes=alibi_slopes, window_size=window_size,
            stream=stream,
        )
        return out, k_updated, v_updated

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
            # _apply_rope_mlx and _can_use_mfa are module-level in attention.py.
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
        block_mask: Boolean tile mask.  Supported shapes:

                    * ``[NQ, NK]`` — shared across all batches and heads.
                    * ``[H, NQ, NK]`` — per-head mask (same for all batches).
                    * ``[B, H, NQ, NK]`` — per-batch, per-head mask.

                    ``NQ = ceil(N / BQ)``, ``NK = ceil(S / BK)`` where BQ/BK
                    come from ``_steel_block_config(D)``.
                    Use :func:`make_causal_block_mask` or
                    :func:`make_sliding_window_mask` to generate 2-D masks.
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
    if block_mask.ndim not in (2, 3, 4):
        raise ValueError(
            "block_mask must be 2-D [NQ, NK], 3-D [H, NQ, NK], or 4-D [B, H, NQ, NK]; "
            f"got shape {list(block_mask.shape)}"
        )

    BQ, BK = _steel_block_config(D)
    NQ_expected = (N + BQ - 1) // BQ
    NK_expected = (S + BK - 1) // BK
    if block_mask.shape[-2] != NQ_expected or block_mask.shape[-1] != NK_expected:
        raise ValueError(
            f"block_mask last two dims {list(block_mask.shape[-2:])} do not match "
            f"expected [{NQ_expected}, {NK_expected}] "
            f"for seq_len={N}/{S}, head_dim={D} (BQ={BQ}, BK={BK})"
        )
    if block_mask.ndim == 3 and block_mask.shape[0] != H:
        raise ValueError(
            f"3-D block_mask shape[0]={block_mask.shape[0]} must equal H={H}"
        )
    if block_mask.ndim == 4:
        if block_mask.shape[0] != B:
            raise ValueError(
                f"4-D block_mask shape[0]={block_mask.shape[0]} must equal B={B}"
            )
        if block_mask.shape[1] != H:
            raise ValueError(
                f"4-D block_mask shape[1]={block_mask.shape[1]} must equal H={H}"
            )

    # Collapse to 2-D for fallback SDPA (no per-head routing in pure Python).
    if not _ext_available():
        mask_2d = block_mask
        if block_mask.ndim == 4:
            mask_2d = block_mask.any(axis=(0, 1))
        elif block_mask.ndim == 3:
            mask_2d = block_mask.any(axis=0)
        return _sparse_fallback_sdpa(q, k, v, mask_2d, BQ, BK, scale, causal)

    impl = _make_mfa_sparse_custom(scale, causal, head_dim=D, backward=backward)
    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    mask_uint8 = block_mask.astype(mx.uint8)
    mask_uint8 = mx.contiguous(mask_uint8)
    # _impl returns (O, L); public API returns only O.
    O, _L = impl(q, k, v, mask_uint8)
    return O


@functools.lru_cache(maxsize=32)
def _make_mfa_sparse_custom(
    scale: float,
    causal: bool,
    head_dim: int = 128,
    backward: str = "sdpa",
):
    """Build and cache a custom_function wrapping the sparse STEEL kernel.

    Cached by (scale, causal, head_dim, backward) — block_mask is passed at
    call time via mask_uint8 so the factory is hashable and lru_cache works.

    The forward returns (O, L) where L is the logsumexp [B, H, N, float32].
    L is in log2 domain (STEEL convention: L = log2(e) * L_natural).

    backward options:
        "sdpa"          (default) — dense mx.fast.sdpa vjp; correct but O(N×S×D)
        "sdpa_sparse"   — tiled Python sparse backward using saved L;
                          O(nnz × BQ × BK × D), benefits large sparse configs
        "steel_sparse"  — Metal STEEL sparse backward; skips inactive tiles in
                          native Metal kernel (fastest for low-density masks)
    """
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

        # Derive the 2-D collapsed mask from the primal mask_uint8.
        # mask_uint8 may be 2-D, 3-D [H,NQ,NK], or 4-D [B,H,NQ,NK].
        _block_mask_2d = mask_uint8
        if mask_uint8.ndim == 4:
            _block_mask_2d = mask_uint8.any(axis=(0, 1))
        elif mask_uint8.ndim == 3:
            _block_mask_2d = mask_uint8.any(axis=0)

        if backward == "steel_sparse":
            # Native STEEL sparse backward — Metal kernel skips inactive tiles.
            # IMPORTANT: MLX's autograd recycles GPU buffers for q/k/v during
            # backward.  Custom Metal primitives read those buffers directly and
            # see garbage data.  After mx.eval() the buffers are pinned;
            # mx.contiguous() then returns a view with fresh graph ancestry so
            # the Metal allocator cannot alias them with earlier outputs.
            # NOTE: backward kernel uses 2-D mask indexing; pass collapsed mask.
            from mlx_mfa._ext import mfa_steel_backward_sparse as _sbwd
            mask_bwd = _block_mask_2d.astype(mx.uint8)
            mx.eval(q, k, v, mask_bwd, dO, O, L)
            q2  = mx.contiguous(q)
            k2  = mx.contiguous(k)
            v2  = mx.contiguous(v)
            O2  = mx.contiguous(O)
            L2  = mx.contiguous(L.astype(mx.float32) if L.dtype != mx.float32 else L)
            dO2 = mx.contiguous(dO)
            mu2 = mx.contiguous(mask_bwd)
            dQ, dK, dV = _sbwd(q2, k2, v2, O2, L2, dO2, mu2, scale, causal)
            return dQ, dK, dV, mx.zeros((1,), dtype=mask_uint8.dtype)  # G.2: scalar zero

        if backward == "sdpa_sparse":
            # C.3: Deprecate in favour of steel_sparse now that C.4 (numpy
            # round-trip) has been eliminated.  steel_sparse uses the native
            # Metal kernel and is faster for all practical inputs.
            import warnings
            warnings.warn(
                "backward='sdpa_sparse' is deprecated and will be removed in "
                "a future release.  Use backward='steel_sparse' instead, which "
                "skips inactive tiles natively in Metal without a Python loop.",
                DeprecationWarning,
                stacklevel=4,
            )
            # J.3: numpy only needed by the deprecated sdpa_sparse branch.
            import numpy as _np  # cold path: deprecated sdpa_sparse backward only
            # Tiled sparse backward using saved L — skips inactive tiles.
            # Use 2-D collapsed mask (tiled backward is per-block, not per-head).
            D = q.shape[-1]
            bq, bk = _steel_block_config(D)
            block_mask_np = _np.array(_block_mask_2d.astype(mx.uint8))
            dQ, dK, dV = _sparse_backward_tiled(
                q, k, v, O, L, dO, block_mask_np, bq, bk, scale, causal
            )
            return dQ, dK, dV, mx.zeros((1,), dtype=mask_uint8.dtype)  # G.2: scalar zero

        # Dense SDPA backward (correct, no sparsity speedup).
        float_mask = _block_mask_to_float_bias(
            _block_mask_2d, q.shape[2], k.shape[2], scale_q_dtype=q.dtype
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
        return dQ, dK, dV, mx.zeros((1,), dtype=mask_uint8.dtype)  # G.2: scalar zero

    return _impl


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track JD — LLM inference helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def flash_attention_speculative_verify(
    q_target: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    draft_ids: mx.array,
    *,
    scale: Optional[float] = None,
    causal: bool = True,
    temperature: float = 1.0,
    stream: Optional[mx.Stream] = None,
) -> tuple:
    """Verify draft tokens from a speculative decoder against the target model.

    Computes token acceptance probabilities for *N_draft* draft tokens using
    the target model's KV cache.  The returned acceptance mask can be used to
    select the longest accepted prefix before resampling from the target
    distribution.

    Algorithm (simplified):

    1. Run target attention over ``q_target`` (N_draft queries) against the
       full KV cache, obtaining output ``O`` and log-sum-exp ``L``.
    2. Convert per-token logits via ``p_target = softmax(O[:, :, i, :] / T)``.
    3. Accept draft token ``i`` with probability
       ``min(1, p_target(draft_ids[i]) / p_draft(draft_ids[i]))``.
       (This function only computes the target logit for ``draft_ids``; the
       caller supplies the draft probabilities separately.)

    Args:
        q_target:   Target-model query projections ``[B, H, N_draft, D]``.
        k_cache:    Full KV cache keys ``[B, H_kv, S, D]``.
        v_cache:    Full KV cache values ``[B, H_kv, S, D]``.
        draft_ids:  Draft token IDs ``[B, N_draft]`` int32.  Used to index
                    the output logits to retrieve ``p_target(draft_ids[i])``.
        scale:      Attention scale; defaults to ``1/sqrt(D)``.
        causal:     Apply causal mask (default ``True``).
        temperature: Softmax temperature (default 1.0).
        stream:     MLX stream.

    Returns:
        3-tuple ``(output, lse, target_logprobs)``:

        * ``output``       — ``[B, H, N_draft, D]``, raw attention output.
        * ``lse``          — ``[B, H, N_draft]``, log-sum-exp (log-partition).
        * ``target_logprobs`` — ``[B, N_draft]``, log p_target for each draft
          token (logits projected through the V dimension; see note below).

    Note:
        ``target_logprobs`` is the log-softmax of the attention output's norm
        along the D dimension, indexed at ``draft_ids``.  This is an
        *approximation* of the target log-prob — the caller must project
        through the language model head for exact probabilities.  This
        function only provides the attention component.
    """
    B, H, N_draft, D = q_target.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    out, lse = flash_attention(
        q_target, k_cache, v_cache,
        scale=scale, causal=causal, return_lse=True, stream=stream,
    )
    # Note: no mx.eval() needed here — ops below are pure MLX and can fuse lazily.

    # Compute per-token log-softmax over D (output dimension) to get a
    # proxy for target log-probability indexed by draft_ids.
    # Shape: [B, N_draft, D] after mean-pooling heads
    out_mean = out.mean(axis=1)  # [B, N_draft, D]
    logits = out_mean / temperature  # scale by temperature
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # [B, N_draft, D]

    # Vectorised gather: no Python loop, no GPU→CPU scalar round-trips
    ids = mx.clip(draft_ids.astype(mx.int32), 0, D - 1)  # [B, N_draft]
    target_logprobs = mx.take_along_axis(
        log_probs, ids[..., None], axis=-1
    ).squeeze(-1).astype(mx.float32)  # [B, N_draft]

    return out, lse, target_logprobs


def make_shared_prefix_cache(
    prefix_q: mx.array,
    prefix_k: mx.array,
    prefix_v: mx.array,
    *,
    scale: Optional[float] = None,
    stream: Optional[mx.Stream] = None,
) -> tuple:
    """Pre-compute KV cache for a shared prompt prefix.

    When many sequences share an identical prompt prefix (e.g. a system
    prompt), computing K and V for the prefix once and reusing across requests
    avoids redundant projection.  This function attends Q over the prefix and
    returns ``(k_prefix, v_prefix)`` ready to be concatenated with per-request
    K/V before the suffix attention step.

    Usage::

        # Shared system prompt — compute once.
        _, k_pre, v_pre = make_shared_prefix_cache(q_pre, k_pre, v_pre)

        # Per-request suffix — concatenate and attend.
        k_full = mx.concatenate([k_pre, k_suffix], axis=2)
        v_full = mx.concatenate([v_pre, v_suffix], axis=2)
        out = flash_attention(q_suffix, k_full, v_full, causal=True)

    Args:
        prefix_q:  Query projections for the prefix ``[B, H, N_pre, D]``.
        prefix_k:  Key projections for the prefix ``[B, H_kv, N_pre, D]``.
        prefix_v:  Value projections for the prefix ``[B, H_kv, N_pre, D]``.
        scale:     Attention scale; defaults to ``1/sqrt(D)``.
        stream:    MLX stream.

    Returns:
        3-tuple ``(prefix_out, k_prefix, v_prefix)``:

        * ``prefix_out`` — ``[B, H, N_pre, D]``, attention over the prefix.
        * ``k_prefix``   — ``[B, H_kv, N_pre, D]``, pass to suffix attention.
        * ``v_prefix``   — ``[B, H_kv, N_pre, D]``, pass to suffix attention.
    """
    D = prefix_q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    prefix_out = flash_attention(
        prefix_q, prefix_k, prefix_v,
        scale=scale, causal=True, stream=stream,
    )
    return prefix_out, prefix_k, prefix_v


def flash_attention_splitfuse(
    q_prefill: Optional[mx.array],
    k_prefill: Optional[mx.array],
    v_prefill: Optional[mx.array],
    q_decode: Optional[mx.array],
    k_cache_decode: Optional[mx.array],
    v_cache_decode: Optional[mx.array],
    *,
    scale: Optional[float] = None,
    causal: bool = True,
    stream: Optional[mx.Stream] = None,
) -> tuple:
    """Split-fuse attention for continuous-batching inference.

    Processes prefill and decode requests in a single Metal dispatch sequence.
    Prefill tokens use standard causal attention; decode tokens use
    :func:`flash_attention` which activates Flash Decode automatically when
    ``N_q ≤ 4`` and ``S ≥ 256``.

    Both sub-batches are independent — there is no cross-attention between
    prefill and decode sequences.

    Args:
        q_prefill:        Prefill queries ``[B_p, H, N_prefill, D]``.
                          Pass ``None`` if there are no prefill requests.
        k_prefill:        Prefill keys ``[B_p, H_kv, N_prefill, D]``.
        v_prefill:        Prefill values ``[B_p, H_kv, N_prefill, D]``.
        q_decode:         Decode queries ``[B_d, H, N_decode, D]``
                          (typically ``N_decode ≤ 4``).
                          Pass ``None`` if there are no decode requests.
        k_cache_decode:   Full KV cache for decode seqs ``[B_d, H_kv, S, D]``.
        v_cache_decode:   Full KV cache values ``[B_d, H_kv, S, D]``.
        scale:            Attention scale; defaults to ``1/sqrt(D)``.
        causal:           Apply causal mask (default ``True``).
        stream:           MLX stream.

    Returns:
        2-tuple ``(out_prefill, out_decode)``:

        * ``out_prefill`` — ``[B_p, H, N_prefill, D]`` or ``None``.
        * ``out_decode``  — ``[B_d, H, N_decode, D]``  or ``None``.
    """
    D = (
        q_prefill.shape[-1] if q_prefill is not None
        else q_decode.shape[-1] if q_decode is not None
        else None
    )
    if D is None:
        raise ValueError("flash_attention_splitfuse: both q_prefill and q_decode are None.")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    out_prefill = None
    out_decode  = None

    if q_prefill is not None:
        out_prefill = flash_attention(
            q_prefill, k_prefill, v_prefill,
            scale=scale, causal=causal, stream=stream,
        )

    if q_decode is not None:
        # Flash Decode is triggered inside flash_attention when N_q ≤ 4 and S ≥ 256.
        out_decode = flash_attention(
            q_decode, k_cache_decode, v_cache_decode,
            scale=scale, causal=causal, stream=stream,
        )

    return out_prefill, out_decode


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
    import numpy as _np  # cold path: deprecated tiled sparse backward

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
    """Return True iff the C++ extension module is importable (cached)."""
    global _ext_avail_cached
    if _ext_avail_cached is not None:
        return _ext_avail_cached
    try:
        from mlx_mfa._ext import mfa_attention_forward  # noqa: F401
        _ext_avail_cached = True
    except ImportError:
        _ext_avail_cached = False
    return _ext_avail_cached


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

    After ``mx.eval(arr)`` the buffer is materialised and pinned; a subsequent
    ``mx.contiguous()`` call returns a view with a freshly allocated header that
    carries no lazy-graph ancestry, so the Metal allocator cannot alias it with
    any earlier output buffer.  This avoids the ~5µs elementwise-add kernel that
    ``arr + mx.zeros_like(arr)`` previously dispatched.
    """
    mx.eval(arr)
    return mx.contiguous(arr)


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
    """Dispatch through the MFA+ALiBi custom-vjp path.

    Row-major contiguity is enforced inside mfa_attention_alibi_forward (D.5).
    """
    impl = _make_mfa_alibi_custom(scale, causal)
    return impl(q, k, v, alibi_slopes)


@functools.lru_cache(maxsize=64)
def _make_mfa_custom(scale: float, causal: bool, softcap: float = 0.0,
                     window_left: int = -1, window_right: int = -1):
    """Return a custom-vjp MFA forward function for the given (scale, causal, softcap, window_left, window_right).

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
        if window_left >= 0 or window_right >= 0 or softcap != 0.0:
            # Window or softcap variant: pass all params via mfa_attention_forward.
            O = mfa_attention_forward(q, k, v, scale, causal, softcap,
                                      window_left, window_right)
            # Produce a dummy L so the return type is always (O, L).
            # The windowed/softcap backward uses mx.vjp and never reads L.
            B, H, N = q.shape[0], q.shape[1], q.shape[2]
            L = mx.zeros([B, H, N], dtype=mx.float32)
        else:
            # Fast path: mfa_forward_with_lse returns both O and L in one kernel.
            # B.1: We now *keep* L as the second return value so the backward can
            # use it directly — no gradient-checkpointing re-run needed.
            O, L = mfa_forward_with_lse(q, k, v, scale, causal)
        return O, L  # always return (O, L); callers index [0] to get O

    @_impl.vjp
    def _backward(primals, cotangents, output):
        # mx.custom_function vjp signature (multiple outputs):
        #   primals    - forward inputs (q, k, v)
        #   cotangents - (dO, dL); dL is always zeros since L is not used downstream
        #   output     - (O, L) saved from forward — L is free, no recompute needed
        q, k, v = primals
        dO, _dL = cotangents   # ignore dL
        O, L    = output       # B.1: L already materialised from forward

        if window_left >= 0 or window_right >= 0:
            # Windowed attention backward: re-run reference SDPA with window mask.
            def _windowed_sdpa(q, k, v):
                N, S = q.shape[2], k.shape[2]
                q_idx = mx.arange(S - N, S, dtype=mx.int32)[:, None]
                k_idx = mx.arange(S, dtype=mx.int32)[None, :]
                in_win = mx.ones((N, S), dtype=mx.bool_)
                if window_left >= 0:
                    in_win = in_win & (k_idx >= q_idx - window_left)
                if window_right >= 0:
                    in_win = in_win & (k_idx <= q_idx + window_right)
                if causal:
                    in_win = in_win & (k_idx <= q_idx)
                mask = mx.where(in_win,
                                mx.zeros((N, S), dtype=q.dtype),
                                mx.full((N, S), float("-inf"), dtype=q.dtype))
                return mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask)
            _, (dQ, dK, dV) = mx.vjp(_windowed_sdpa, [q, k, v], [dO])
        elif softcap == 0.0:
            # Phase 1.3 — SDPA vjp is the default backward path.
            #
            # Benchmark data (bench_backward_matrix.py, M1 Max 2026-03-10):
            #   mfa_steel_backward is 0.15-0.63x vs mx.vjp(SDPA) in ALL configs.
            #   D=64 N=4096: steel=35ms vs sdpa=22ms (0.63x)
            #   D=128 N=4096: steel=128ms vs sdpa=31ms (0.24x)
            #   D=256 N=4096: steel=317ms vs sdpa=50ms (0.16x)
            #
            # SDPA vjp is used by default for ALL f16/bf16 shapes.
            # The STEEL backward kernel remains compiled and accessible via
            # backend='mfa' + explicit opt-in (future Track M).
            #
            # PERF-TODO: Track M — native STEEL backward — reinvestigate when
            # mfa_steel_backward achieves parity with mx.vjp(SDPA).
            _, (dQ, dK, dV) = mx.vjp(
                lambda q, k, v: _fallback_sdpa(q, k, v, scale, causal),
                [q, k, v],
                [dO],
            )
        else:
            _, (dQ, dK, dV) = mx.vjp(
                lambda q, k, v: _softcap_sdpa_ref(q, k, v, scale, causal, softcap),
                [q, k, v],
                [dO],
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
    window_right: int = -1,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Dispatch through the MFA custom-vjp path.

    The ``stream`` argument is accepted for API compatibility but the
    custom-vjp path always uses the default GPU stream.
    Row-major contiguity is enforced inside the C++ binding entry points
    (D.5 fix) — no need for Python-level mx.contiguous() here.
    """
    impl = _make_mfa_custom(scale, causal, softcap, window_left, window_right)
    # _impl now returns (O, L); callers only need O.
    O, _L = impl(q, k, v)
    return O


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

    sdpa_mask = None
    if causal:
        N, S = q.shape[2], k.shape[2]
        # Build causal mask once in float32; reuse (cast) for SDPA
        cmask_f32 = mx.triu(
            mx.full((N, S), float("-inf"), dtype=mx.float32),
            k=S - N + 1,
        )
        scores = scores + cmask_f32
        sdpa_mask = cmask_f32.astype(q.dtype)

    # LSE in log2 domain: L = max + log2(sum(2^(scores - max)))
    # Use log2 = log / ln(2) to avoid mx.exp2/mx.log2 which may be absent in
    # older MLX builds.  The constant converts natural-log to log-base-2.
    _LN2 = 0.6931471805599453  # ln(2)
    max_s = scores.max(axis=-1, keepdims=True)                    # [B,H,N,1]
    exp_s = mx.exp((scores - max_s) * _LN2)                       # [B,H,N,S]
    lse   = max_s.squeeze(-1) + mx.log(exp_s.sum(axis=-1)) / _LN2  # [B,H,N]

    # Standard softmax attention output (use built-in for efficiency)
    O = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=sdpa_mask)
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


def _apply_rope_to_qk(
    q: mx.array,
    k: mx.array,
    rotary_cos: mx.array,
    rotary_sin: mx.array,
    q_offset: int = 0,
    k_offset: int = 0,
    interleaved: bool = True,
    rotary_dim: Optional[int] = None,
) -> tuple:
    """Apply RoPE to Q and K independently; return ``(q_rot, k_rot)``.

    Pure rotation helper — does not compute attention.  Use this when the
    caller needs to rotate Q and K before dispatching its own kernel (e.g.
    MFA, paged-KV, varlen).

    Args:
        q:            Query tensor ``[B, H, N, D]``.
        k:            Key tensor ``[B, H, S, D]``.
        rotary_cos:   Cosine table ``[max_seq, rotary_dim/2]``.
        rotary_sin:   Sine table ``[max_seq, rotary_dim/2]``.
        q_offset:     Token position of the first Q token (= cache_seqlens).
        k_offset:     Token position of the first K token (0 for full KV).
        interleaved:  True = LLaMA adjacent-pair layout; False = GPT-NeoX.
        rotary_dim:   Elements to rotate; None rotates all ``D`` elements.

    Returns:
        ``(q_rot, k_rot)`` — rotated tensors, same dtype and shape as inputs.
    """
    q_rot = _apply_rope_mlx(q, rotary_cos, rotary_sin,
                             offset=q_offset, interleaved=interleaved,
                             rotary_dim=rotary_dim)
    k_rot = _apply_rope_mlx(k, rotary_cos, rotary_sin,
                             offset=k_offset, interleaved=interleaved,
                             rotary_dim=rotary_dim)
    return q_rot, k_rot


def _apply_rope_and_attend(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rotary_cos: mx.array,
    rotary_sin: mx.array,
    scale: float,
    causal: bool,
    q_offset: int = 0,
    k_offset: int = 0,
    interleaved: bool = True,
    rotary_dim: Optional[int] = None,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Apply RoPE to Q and K then compute SDPA.

    Convenience helper that unifies the repeated pattern::

        q_rot = _apply_rope_mlx(q, cos, sin, offset=q_offset, ...)
        k_rot = _apply_rope_mlx(k, cos, sin, offset=k_offset, ...)
        return _fallback_sdpa(q_rot, k_rot, v, scale, causal, stream)

    Args:
        q, k, v:      Attention tensors ``[B, H, N, D]``.
        rotary_cos:   Cosine table ``[max_seq, rotary_dim/2]``.
        rotary_sin:   Sine table ``[max_seq, rotary_dim/2]``.
        scale:        Attention scale factor.
        causal:       Whether to apply causal masking.
        q_offset:     Token position of the first Q token (= cache_seqlens).
        k_offset:     Token position of the first K token (usually 0 for
                      full KV; same as ``q_offset`` for decode append).
        interleaved:  True = LLaMA (adjacent pairs); False = GPT-NeoX.
        rotary_dim:   Number of head-dim elements to rotate; None = all D.
        stream:       MLX stream (forwarded to ``_fallback_sdpa``).

    Returns:
        Attention output ``[B, H, N, D]`` in the same dtype as q.
    """
    q_rot, k_rot = _apply_rope_to_qk(
        q, k, rotary_cos, rotary_sin,
        q_offset=q_offset, k_offset=k_offset,
        interleaved=interleaved, rotary_dim=rotary_dim,
    )
    return _fallback_sdpa(q_rot, k_rot, v, scale, causal, stream)


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
            return _apply_rope_and_attend(
                q, k, v, rotary_cos, rotary_sin, scale, causal,
                q_offset=cache_seqlens, k_offset=0, interleaved=interleaved,
            )

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
    # J.2: contiguity ensured by the C++ binding (D.5); no Python-side calls needed.
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
    cu_q = [int(x) for x in cu_seqlens_q.tolist()]  # GPU sync: varlen per-seq slicing
    cu_k_list = [int(x) for x in cu_seqlens_k.tolist()]  # GPU sync: varlen per-seq slicing
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
        # _MFA_SUPPORTED_HDIMS includes 512, but the varlen STEEL kernel lacks
        # the d_split path that the main forward kernel uses for D=512 (TGP
        # would be 65 KB, exceeding the 32 KB threadgroup limit).  Cap at 256
        # until varlen d_split is implemented; D=512 falls back to split-concat.
        if (
            _ext_available()
            and q_.dtype in (mx.float16, mx.bfloat16)
            and D in _MFA_SUPPORTED_HDIMS
            and D <= 256
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
# Phase 2 — KVCacheProtocol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class KVCacheProtocol:
    """Structural base for KV cache implementations.

    Both :class:`DenseKVCache` and :class:`PagedKVCache` implement this
    interface.  Callers can accept ``KVCacheProtocol`` as a type hint to
    write code that works with either backend.

    Protocol methods:

    * :meth:`append` — write new K/V tokens for ``seq_id``
    * :meth:`k_for_attention` — return K ready for attention ``[B,H,S,D]``
    * :meth:`v_for_attention` — return V ready for attention ``[B,H,S,D]``
    * :meth:`seq_length` — current token count for ``seq_id``
    * :meth:`reset` — clear state; if ``seq_id is None`` clears all sequences
    """

    def append(
        self,
        k_new: "mx.array",
        v_new: "mx.array",
        seq_id: int = 0,
    ) -> None:
        """Append ``k_new / v_new`` tokens for sequence ``seq_id``."""
        raise NotImplementedError

    def k_for_attention(self, seq_id: int = 0) -> "mx.array":
        """Return K tensor ``[B, H, S, D]`` ready for attention."""
        raise NotImplementedError

    def v_for_attention(self, seq_id: int = 0) -> "mx.array":
        """Return V tensor ``[B, H, S, D]`` ready for attention."""
        raise NotImplementedError

    def seq_length(self, seq_id: int = 0) -> int:
        """Return the number of tokens stored for ``seq_id``."""
        raise NotImplementedError

    def reset(self, seq_id: "Optional[int]" = None) -> "KVCacheProtocol":
        """Reset cache state.  ``seq_id=None`` clears all sequences."""
        raise NotImplementedError


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# I.2 — DenseKVCache: pre-allocated dense KV buffer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DenseKVCache(KVCacheProtocol):
    """Dense KV cache with pre-allocated buffer and write-pointer.

    Solves the O(seqlen²) graph-accumulation problem of repeated
    ``mx.concatenate([cache, k_new], axis=2)`` calls by writing new tokens
    into a fixed pre-allocated ``[B, H, max_seq_len, D]`` buffer via
    ``mx.slice_update`` (MLX ``__setitem__``).  ``mx.eval()`` is called after
    each append to keep the lazy graph at constant depth, so evaluation cost
    stays O(1) regardless of how many tokens have been appended.

    **Key benefit**: avoids the O(seqlen)-deep lazy graph that accumulates
    when concatenation results are not explicitly evaluated between steps.

    Example::

        cache = DenseKVCache(B=1, H=8, D=128, max_seq_len=4096)

        # Prefill
        cache.append(k_prefill, v_prefill)   # [B, H, N_prefill, D]

        # Decode loop
        for _ in range(steps):
            cache.append(k_new, v_new)       # [B, H, 1, D]
            out = flash_attention_kvcache(
                q, cache.k, cache.v,
                cache_seqlens=cache.seqlen,
            )

        cache.reset()   # reuse for next sequence

    Args:
        B:            Batch size.
        H:            Number of KV heads.
        D:            Head dimension.
        max_seq_len:  Maximum total sequence length (prefill + generated).
        dtype:        MLX dtype for the buffer (default: ``mx.float16``).

    Attributes:
        seqlen (int): Current number of tokens written.
        k:            Active slice ``[B, H, seqlen, D]``.
        v:            Active slice ``[B, H, seqlen, D]``.
    """

    def __init__(
        self,
        B: int,
        H: int,
        D: int,
        max_seq_len: int = 8192,
        dtype=None,
    ) -> None:
        if dtype is None:
            dtype = mx.float16
        self.B = B
        self.H = H
        self.D = D
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self._seqlen: int = 0

        # Pre-allocate fixed buffers; eval immediately to commit GPU memory.
        self._k = mx.zeros([B, H, max_seq_len, D], dtype=dtype)
        self._v = mx.zeros([B, H, max_seq_len, D], dtype=dtype)
        mx.eval(self._k, self._v)

    # -- Properties ----------------------------------------------------------

    @property
    def seqlen(self) -> int:
        """Number of tokens currently in the cache."""
        return self._seqlen

    @property
    def k(self) -> "mx.array":
        """Active K slice ``[B, H, seqlen, D]``."""
        return self._k[:, :, :self._seqlen, :]

    @property
    def v(self) -> "mx.array":
        """Active V slice ``[B, H, seqlen, D]``."""
        return self._v[:, :, :self._seqlen, :]

    # -- Mutation ------------------------------------------------------------

    def append(
        self,
        k_new: "mx.array",
        v_new: "mx.array",
        seq_id: int = 0,
    ) -> None:
        """Scatter ``k_new / v_new`` into the pre-allocated buffer.

        ``seq_id`` is accepted for protocol compatibility with
        :class:`PagedKVCache` but ignored — :class:`DenseKVCache` supports
        only a single sequence (``seq_id=0``).

        Args:
            k_new:  New key tokens   ``[B, H, N_new, D]``.
            v_new:  New value tokens ``[B, H, N_new, D]``.
            seq_id: Sequence identifier (must be 0; ignored).

        Raises:
            ValueError: if ``seqlen + N_new > max_seq_len``.
        """
        n_new = k_new.shape[2]
        end   = self._seqlen + n_new
        if end > self.max_seq_len:
            raise ValueError(
                f"DenseKVCache: seqlen {end} > max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len or switch to PagedKVCache."
            )
        # Scatter into pre-allocated buffer via __setitem__ (MLX slice_update).
        # O(max_seq_len) write but constant graph depth after mx.eval().
        self._k[:, :, self._seqlen:end, :] = k_new.astype(self.dtype)
        self._v[:, :, self._seqlen:end, :] = v_new.astype(self.dtype)
        self._seqlen = end
        # Materialise the scatter to prevent O(seqlen) lazy-graph growth.
        mx.eval(self._k, self._v)

    # -- KVCacheProtocol methods ---------------------------------------------

    def k_for_attention(self, seq_id: int = 0) -> "mx.array":
        """Return K slice ``[B, H, seqlen, D]`` ready for attention."""
        return self.k

    def v_for_attention(self, seq_id: int = 0) -> "mx.array":
        """Return V slice ``[B, H, seqlen, D]`` ready for attention."""
        return self.v

    def seq_length(self, seq_id: int = 0) -> int:
        """Return current token count (``seq_id`` ignored for dense cache)."""
        return self._seqlen

    def reset(self, seq_id: "Optional[int]" = None) -> "DenseKVCache":
        """Reset write pointer to 0 (reuse buffer for a new sequence).

        ``seq_id`` is accepted for protocol compatibility but ignored —
        the dense cache tracks a single sequence.

        Does **not** zero the buffer — stale data is unreachable since we
        track ``seqlen`` and slice with ``[:, :, :seqlen, :]``.

        Returns:
            ``self`` — enables chaining: ``cache.reset().append(...)``
        """
        self._seqlen = 0
        return self

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "DenseKVCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.reset()

    def __repr__(self) -> str:
        return (
            f"DenseKVCache(B={self.B}, H={self.H}, D={self.D}, "
            f"seqlen={self._seqlen}/{self.max_seq_len})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Track BE — Paged KV Cache Phase 1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PagedKVCache(KVCacheProtocol):
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

        if _USE_SCATTER_KV:
            # Phase 4-E.2: single Metal scatter dispatch replaces O(pool_size) concat.
            # Collect all (blk_id, slot_off) targets across all T tokens first.
            all_blk_ids: list = []
            all_blk_offs: list = []
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
                # F.3: extend replaces per-element append loop.
                all_blk_ids.extend([blk_id] * chunk)
                all_blk_offs.extend(range(ptr, ptr + chunk))
                self._write_ptr[seq_id] = ptr + chunk
                written += chunk
            # One scatter call per pool (copy + scatter in one Metal pass).
            blk_ids_arr  = mx.array(all_blk_ids,  dtype=mx.int32)
            blk_offs_arr = mx.array(all_blk_offs, dtype=mx.int32)
            self._k_pool = _mfa_scatter_kv_cpp(
                self._k_pool, k_tokens, blk_ids_arr, blk_offs_arr)
            self._v_pool = _mfa_scatter_kv_cpp(
                self._v_pool, v_tokens, blk_ids_arr, blk_offs_arr)
        else:
            # Fallback: MLX-native concatenation path.
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
                new_k = mx.concatenate(parts_k, axis=0)[None]
                new_v = mx.concatenate(parts_v, axis=0)[None]
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

    # -- KVCacheProtocol methods ---------------------------------------------

    def k_for_attention(self, seq_id: int = 0) -> "mx.array":
        """Return contiguous K ``[1, H, S, D]`` for ``seq_id``.

        Gathers paged blocks into a contiguous tensor via :meth:`gather`.
        For large caches prefer :func:`flash_attention_kvcache` (paged mode)
        which reads tiles directly without materialising a full copy.
        """
        k, _ = self.gather(seq_id)
        return k

    def v_for_attention(self, seq_id: int = 0) -> "mx.array":
        """Return contiguous V ``[1, H, S, D]`` for ``seq_id``."""
        _, v = self.gather(seq_id)
        return v

    def seq_length(self, seq_id: int = 0) -> int:
        """Return the number of tokens stored for ``seq_id``."""
        return self.seq_lengths.get(seq_id, 0)

    def reset(self, seq_id: "Optional[int]" = None) -> "PagedKVCache":
        """Free blocks and reset state.

        Args:
            seq_id: If given, free only that sequence.  If ``None``,
                    free all sequences (resets the pool to its initial state).

        Returns:
            ``self`` — enables chaining: ``cache.reset().append(...)``
        """
        if seq_id is None:
            # Free every active sequence.
            for sid in list(self._block_table.keys()):
                self.free_seq(sid)
        else:
            self.free_seq(seq_id)
        return self

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
    Supports autograd: ``dQ``, ``dK_pages``, and ``dV_pages`` are all computed
    correctly.  Per-sequence gradients are gathered via the dense attention
    backward and then scattered back to the paged pool via ``_scatter_to_pool``
    (gather → dense vjp → scatter-accumulate).  Partial pages are zero-padded
    to ``block_size`` before accumulation.

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
    seq_lens_list = [int(x) for x in seq_lens.tolist()]  # GPU sync: paged backward slicing
    block_table_list = block_table.tolist()  # GPU sync: paged backward block scatter
    max_kv_len = max(seq_lens_list) if seq_lens_list else 0

    if max_kv_len == 0:
        return mx.zeros((B, H_q, N_q, D), dtype=q.dtype)

    def _scatter_to_pool(
        dk_seqs: list,
        dv_seqs: list,
        dtype,
    ):
        """Scatter per-sequence dK/dV gradients back to the paged pool.

        Inverse of ``_gather_contig``.

        Args:
            dk_seqs: List of B arrays, each ``[1, H_kv, kv_len_b, D]``.
            dv_seqs: Same.
            dtype:   Output dtype.

        Returns:
            ``(dk_pages, dv_pages)`` both of shape
            ``[num_blocks, block_size, H_kv, D]``.
        """
        num_blocks = k_pages.shape[0]
        H_kv_s = k_pages.shape[2]
        D_s = k_pages.shape[3]

        dk_acc: dict = {}
        dv_acc: dict = {}
        for b, (kv_len, table_b) in enumerate(zip(seq_lens_list, block_table_list)):
            if kv_len == 0:
                continue
            dk_b = dk_seqs[b][0]  # [H_kv, kv_len, D]
            dv_b = dv_seqs[b][0]
            n_full, rem = divmod(kv_len, block_size)
            for lb in range(n_full):
                phys = int(table_b[lb])
                if phys < 0:
                    continue
                s = lb * block_size
                dk_tile = dk_b[:, s:s + block_size, :].transpose(1, 0, 2)  # [bs, H_kv, D]
                dv_tile = dv_b[:, s:s + block_size, :].transpose(1, 0, 2)
                if phys in dk_acc:
                    dk_acc[phys] = dk_acc[phys] + dk_tile
                    dv_acc[phys] = dv_acc[phys] + dv_tile
                else:
                    dk_acc[phys] = dk_tile
                    dv_acc[phys] = dv_tile
            if rem > 0:
                phys = int(table_b[n_full])
                if phys >= 0:
                    s = n_full * block_size
                    dk_p = dk_b[:, s:s + rem, :].transpose(1, 0, 2)  # [rem, H_kv, D]
                    dv_p = dv_b[:, s:s + rem, :].transpose(1, 0, 2)
                    pad = block_size - rem
                    dk_tile = mx.pad(dk_p, [(0, pad), (0, 0), (0, 0)])
                    dv_tile = mx.pad(dv_p, [(0, pad), (0, 0), (0, 0)])
                    if phys in dk_acc:
                        dk_acc[phys] = dk_acc[phys] + dk_tile
                        dv_acc[phys] = dv_acc[phys] + dv_tile
                    else:
                        dk_acc[phys] = dk_tile
                        dv_acc[phys] = dv_tile

        zero = mx.zeros((block_size, H_kv_s, D_s), dtype=dtype)
        dk_blocks = [dk_acc.get(i, zero) for i in range(num_blocks)]
        dv_blocks = [dv_acc.get(i, zero) for i in range(num_blocks)]
        return mx.stack(dk_blocks), mx.stack(dv_blocks)

    def _gather_contig(k_p: "mx.array", v_p: "mx.array"):
        """Gather pool pages → contiguous [B, H_kv, max_kv_len, D]."""
        if _ext_available() and k_p.dtype in (mx.float16, mx.bfloat16):
            from mlx_mfa._ext import mfa_paged_kv_gather
            K = mfa_paged_kv_gather(k_p, block_table, seq_lens, max_kv_len)
            V = mfa_paged_kv_gather(v_p, block_table, seq_lens, max_kv_len)
            return K, V
        # H.3: Advanced-indexing fallback gather (all dtypes).
        # Gather all blocks per-batch in one mx.take op, eliminating the
        # inner per-block loop.  Outer B loop remains; inner is now O(1) MLX.
        max_n_blocks = (max_kv_len + block_size - 1) // block_size
        K_list, V_list = [], []
        for b in range(B):
            kv_len = seq_lens_list[b]
            table_b = block_table_list[b]
            n_full, rem = divmod(kv_len, block_size)
            n_blocks = n_full + (1 if rem > 0 else 0)
            if n_blocks == 0:
                k_seq = mx.zeros([0, H_kv, D], dtype=k_p.dtype)
                v_seq = mx.zeros([0, H_kv, D], dtype=v_p.dtype)
            else:
                # Gather all needed blocks at once: [n_blocks, block_size, H_kv, D]
                phys_arr = mx.array(table_b[:n_blocks], dtype=mx.int32)
                k_blks = k_p[phys_arr]  # [n_blocks, block_size, H_kv, D]
                v_blks = v_p[phys_arr]
                # Reshape to [n_blocks*block_size, H_kv, D] and trim to kv_len
                k_seq = k_blks.reshape(n_blocks * block_size, H_kv, D)[:kv_len]
                v_seq = v_blks.reshape(n_blocks * block_size, H_kv, D)[:kv_len]
            pad = max_kv_len - k_seq.shape[0]
            if pad > 0:
                k_seq = mx.pad(k_seq, [(0, pad), (0, 0), (0, 0)])
                v_seq = mx.pad(v_seq, [(0, pad), (0, 0), (0, 0)])
            K_list.append(k_seq.transpose(1, 0, 2)[None])  # [1, H_kv, max_kv_len, D]
            V_list.append(v_seq.transpose(1, 0, 2)[None])
        return mx.concatenate(K_list, axis=0), mx.concatenate(V_list, axis=0)

    def _attn_per_seq(q_, K_contig, V_contig):
        """Per-sequence attention using exact kv_len slices (avoids padding-leak NaN).

        The Metal paged-gather kernel can leave uninitialized bytes beyond the
        written pool region, which appear as NaN in K_contig.  Slicing to
        [:kv_len] keeps only the written tokens; flash_attention's STEEL Metal
        kernel additionally quenches NaN via GPU-level score masking.
        Using mx.fast.scaled_dot_product_attention directly would propagate NaN
        through IEEE-compliant softmax into the output — H.1 reverted for safety.
        """
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
    # eliminating the gather→attend round-trip.  f16/bf16 + D≤256 only.
    # D=512 exceeds the paged kernel's 32KB TGP limit (no d_split); it falls
    # back to the gather→flash_attention path which has d_split support.
    _USE_PAGED_STEEL = (
        _ext_available()
        and q.dtype in (mx.float16, mx.bfloat16)
        and D in _MFA_SUPPORTED_HDIMS
        and D <= 256
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

    def _paged_batched_bwd(q_, K_contig, V_contig, dO):
        """E.4: Batched backward — one mx.vjp call for all B sequences.

        Builds a key-padding mask [B, 1, 1, max_kv_len] with -inf at padded
        positions so that padded KV slots contribute zero attention weight.
        This replaces B serial mx.vjp calls with one parallel batch call.

        Returns: (dQ [B,H,N,D], dK_seqs list, dV_seqs list)
        """
        max_kv_len = K_contig.shape[2]
        # Key-padding mask: -inf for positions >= seq_len[b].
        kv_lens_arr = mx.array(seq_lens_list, dtype=mx.int32)   # [B]
        idx = mx.arange(max_kv_len, dtype=mx.int32)              # [max_kv_len]
        valid = idx[None, :] < kv_lens_arr[:, None]              # [B, max_kv_len]
        pad_mask = mx.where(
            valid[:, None, None, :],
            mx.zeros((1,), dtype=q_.dtype),
            mx.full((1,), float("-inf"), dtype=q_.dtype),
        )  # [B, 1, 1, max_kv_len]
        if causal:
            N_q, S = q_.shape[2], max_kv_len
            q_idx = mx.arange(S - N_q, S, dtype=mx.int32)[:, None]
            k_idx = mx.arange(S, dtype=mx.int32)[None, :]
            causal_m = mx.where(
                k_idx <= q_idx,
                mx.zeros((N_q, S), dtype=q_.dtype),
                mx.full((N_q, S), float("-inf"), dtype=q_.dtype),
            )
            pad_mask = pad_mask + causal_m[None, None, :, :]
        _, (dQ, dK_pad, dV_pad) = mx.vjp(
            lambda qi, ki, vi: mx.fast.scaled_dot_product_attention(
                qi, ki, vi, scale=scale, mask=pad_mask),
            [q_, K_contig, V_contig],
            [dO],
        )
        # Crop dK/dV to exact kv_lens (padded positions have grad=0 already).
        dK_seqs = [dK_pad[b:b+1, :, :seq_lens_list[b], :] for b in range(B)]
        dV_seqs = [dV_pad[b:b+1, :, :seq_lens_list[b], :] for b in range(B)]
        return dQ, dK_seqs, dV_seqs

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
            K_contig, V_contig = _gather_contig(k_pages_, v_pages_)
            dQ, dK_seqs, dV_seqs = _paged_batched_bwd(
                q_, K_contig, V_contig, dO)
            dk_pages, dv_pages = _scatter_to_pool(dK_seqs, dV_seqs, k_pages_.dtype)
            return dQ, dk_pages, dv_pages

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
        dQ, dK_seqs, dV_seqs = _paged_batched_bwd(
            q_, K_contig, V_contig, dO)
        dk_pages, dv_pages = _scatter_to_pool(dK_seqs, dV_seqs, k_pages_.dtype)
        return dQ, dk_pages, dv_pages

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
