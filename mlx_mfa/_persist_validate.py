"""Volet P5 — single shared K/V persistence-validation contract.

Every surface that PERSISTS K/V into cache/store state (the dense / quantized /
TurboQuant / paged / TQ-paged appenders + the host KV-store `put`) must enforce
the SAME complete contract: rank, batch, heads (GQA-legal), token length, and
head_dim. P1/P4 implemented the check per-site, and the paged sites covered only
a subset of axes (missed batch → a batch-mismatched V was silently ignored, or a
paged append given batch>1 silently sliced `[0]`). Defining the contract ONCE
here — and routing every persistence surface through it — makes a partial subset
structurally impossible.

D-contract (aligned with the function surface `_assert_qkv_mutual_compat`): the
cache buffers are single-`D` (allocated with one head_dim), so persistence
requires `D_v == D_k` (`require_v_dim_eq=True`, the default). This is NOT stricter
than the function surface's asymmetric-`D_v` allowance: that allowance is for the
dense *attention call* (q/k/v passed directly to an SDPA-class path), not for
storing into a one-`D` cache buffer. The asymmetric path (e.g.
`make_shared_prefix_cache`) is an attention call, not a persistence surface, and
correctly keeps accepting `D_v != D_k`.
"""
from __future__ import annotations


def assert_kv_persist_compat(
    k, v, fn: str, *,
    expected_batch=None, expected_heads=None, expected_dim=None,
    require_v_dim_eq: bool = True, accepted_dtypes=None,
) -> None:
    """Validate a K/V pair about to be persisted into cache/store state.

    Args:
        k, v: new key/value tensors `[B, H, N, D]`.
        fn:   caller name for error messages.
        expected_batch/heads/dim: the surface's configured/structural value (the
            paged appends are single-sequence → ``expected_batch=1``; the dense/
            quantized caches pass their buffer's B/H/D). ``None`` skips that
            cross-check (chunk-based caches with no fixed geometry).
        require_v_dim_eq: require ``D_v == D_k`` (default — cache buffers are
            single-`D`). Persistence surfaces always set this True.
        accepted_dtypes: the surface's accepted INPUT dtype set. A storage-dtype
            cache passes its single ``(self.dtype,)`` — a mismatched append must
            RAISE, not silently cast (the dtype-axis defect). A quantizing cache
            with no fixed storage dtype passes its legal input set
            ``(float16, bfloat16)``. ``None`` skips the input-set check (only the
            K↔V consistency rule applies). K↔V dtype consistency is ALWAYS
            enforced (matches the function surface's k/v-dtype rule).
    """
    if getattr(k, "ndim", None) != 4 or getattr(v, "ndim", None) != 4:
        raise ValueError(
            f"{fn}: k and v must be 4-D [B, H, N, D]; got "
            f"k.ndim={getattr(k, 'ndim', None)}, v.ndim={getattr(v, 'ndim', None)}.")
    # batch — K↔V AND (if known) the surface's single-sequence / configured batch.
    if k.shape[0] != v.shape[0]:
        raise ValueError(
            f"{fn}: k and v must share the batch dim (k={k.shape[0]}, v={v.shape[0]}).")
    if expected_batch is not None and k.shape[0] != expected_batch:
        raise ValueError(
            f"{fn}: batch dim {k.shape[0]} != expected {expected_batch}; this "
            "persistence surface is single-sequence (it would otherwise silently "
            "use index 0 and drop the rest).")
    # heads — K↔V AND (if known) the configured KV-head count.
    if k.shape[1] != v.shape[1]:
        raise ValueError(
            f"{fn}: k and v must have the same number of heads "
            f"(k={k.shape[1]}, v={v.shape[1]}); a mismatched V head count would "
            "silently broadcast / store inconsistent state.")
    if expected_heads is not None and k.shape[1] != expected_heads:
        raise ValueError(
            f"{fn}: head count {k.shape[1]} != configured kv-heads {expected_heads}.")
    # token length — K↔V.
    if k.shape[2] != v.shape[2]:
        raise ValueError(
            f"{fn}: k and v must share the new-token length "
            f"(k={k.shape[2]}, v={v.shape[2]}).")
    # head_dim — K vs configured, and V vs K (single-D cache buffers).
    if expected_dim is not None and k.shape[3] != expected_dim:
        raise ValueError(
            f"{fn}: head_dim {k.shape[3]} != configured D {expected_dim}.")
    if require_v_dim_eq and v.shape[3] != k.shape[3]:
        raise ValueError(
            f"{fn}: k and v must share head_dim for single-D cache storage "
            f"(k={k.shape[3]}, v={v.shape[3]}).")
    # dtype — K↔V consistency (always; the function-surface rule) + the surface's
    # accepted INPUT dtype set (no silent cast into a fixed-dtype buffer).
    if k.dtype != v.dtype:
        raise ValueError(
            f"{fn}: k and v must share dtype (k={k.dtype}, v={v.dtype}); a "
            "mismatched K/V dtype would be silently cast / reinterpreted.")
    if accepted_dtypes is not None and k.dtype not in tuple(accepted_dtypes):
        raise ValueError(
            f"{fn}: input dtype {k.dtype} is not accepted (expected one of "
            f"{tuple(accepted_dtypes)}); this surface does not silently cast a "
            "mismatched-dtype append into its storage buffer.")
