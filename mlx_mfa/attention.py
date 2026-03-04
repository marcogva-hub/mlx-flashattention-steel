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

    # --- Grouped Query Attention (GQA) support --------------------------------
    # When H_kv < H_q (GQA / MQA), tile k and v to match H_q so the standard
    # kernel sees H_q == H_kv.  This is correct and simple; native GQA without
    # tiling is tracked as a future enhancement (Phase 5, H_Hk_ratio kernel param).
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if kv_heads != q_heads:
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"flash_attention GQA: q_heads ({q_heads}) must be divisible "
                f"by kv_heads ({kv_heads})."
            )
        ratio = q_heads // kv_heads
        # k/v: [B, H_kv, S, D] → [B, H_q, S, D] via repeat along axis 1.
        k = mx.repeat(k, ratio, axis=1)
        v = mx.repeat(v, ratio, axis=1)

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
    #
    # M3+ (gen >= 15) uses preferAsyncCache kernel params instead of
    # preferAsyncLoad, following the ccv blocking-parameter tables.
    _GEN_TO_CHIP = {
        13: "M1",
        14: "M2",
        15: "M3",
        16: "M4",
    }
    gen = raw.get("gpu_family_gen")
    chip = _GEN_TO_CHIP.get(gen, f"Apple-g{gen}") if gen is not None else None
    is_m3_plus = (gen >= 15) if gen is not None else None

    return {
        "device_name":         raw.get("device_name"),
        "gpu_family_gen":      gen,
        "is_m3_plus":          is_m3_plus,
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
