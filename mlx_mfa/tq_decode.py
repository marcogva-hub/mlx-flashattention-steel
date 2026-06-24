"""TurboQuant paged DECODE attend via gather/dequant kernels + Apple SDPA.

Sprint III-2 (campaign 2026-06).  §AA.5 premise validation inverted the
TurboQuant P2-P4 "fuse dequant into the attend kernel" premise on M5:
materializing the dequantized K (and gathered V) per decode step and
attending with `mx.fast.scaled_dot_product_attention` (Apple's NAX
sdpa_vector 2-pass) is 7.6-8.3x faster than the fused TQ attend kernel
at the II-7 ladder cells — and the two tiny elementwise kernels here
push the dequant/gather to bandwidth-bound, landing the full step near
the dense-decode floor.

Scores are computed in the rotated space: WHT is orthogonal, so
rotated-q . rotated-k == q . k — no de-rotation anywhere.

V always reads the fp16 pool (it is maintained unconditionally by
`TurboQuantPagedInferenceContext`, even under ``tq_v=True``); this is
both faster and MORE accurate than dequantizing packed V.

Public entry: :func:`tq_decode_attend` (used by
``TurboQuantPagedInferenceContext.step``; opt-out
``MFA_DISABLE_TQ_DECODE_SDPA=1`` restores the fused kernel).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import mlx.core as mx

# Kernel-object caches keyed by CONTENT (config tuples), never ids —
# Sprint A cache-key discipline.
_K_DEQUANT_KERNELS: dict = {}
_V_GATHER_KERNELS: dict = {}

# P9: the tq_decode kernels output the cache dtype natively (the V pool and the
# rotated q are at the context dtype).  Map the MLX dtype to its MSL scalar type
# so the K-dequant accumulator + V-gather load/output are bf16-native for a bf16
# cache (no lossy bf16→fp16→bf16 round-trip) and byte-identical `half` for fp16.
_MSL_TYPE = {mx.float16: "half", mx.bfloat16: "bfloat16_t"}


def _msl_type(dtype) -> str:
    t = _MSL_TYPE.get(dtype)
    if t is None:
        raise ValueError(
            f"tq_decode kernels support float16/bfloat16 cache dtype; got {dtype}.")
    return t

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def _unpack_snippet(bits: int) -> str:
    """MSL snippet computing `idx` (uchar centroid index) for element d.

    Layouts mirror `pack_k_for_metal` exactly:
      3-bit: bit-planar, 32 indices -> 3 planes x 4 bytes = 12 bytes.
      2-bit: 4 indices/byte, little-end first.
      4-bit: 2 indices/byte.
    """
    if bits == 3:
        return """
      const int group = d / 32;
      const int lane  = d % 32;
      const int byte_in_lane = lane / 8;
      const int bit_in_byte  = lane % 8;
      const ulong base = pk_base + (ulong)(group * 12);
      const uchar b0 = k_pool[base + 0 * 4 + byte_in_lane];
      const uchar b1 = k_pool[base + 1 * 4 + byte_in_lane];
      const uchar b2 = k_pool[base + 2 * 4 + byte_in_lane];
      const uchar idx = ((b0 >> bit_in_byte) & 1)
                      | (((b1 >> bit_in_byte) & 1) << 1)
                      | (((b2 >> bit_in_byte) & 1) << 2);
"""
    if bits == 2:
        return """
      const uchar byte = k_pool[pk_base + (ulong)(d / 4)];
      const uchar idx = (byte >> ((d % 4) * 2)) & 3;
"""
    if bits == 4:
        return """
      const uchar byte = k_pool[pk_base + (ulong)(d / 2)];
      const uchar idx = (byte >> ((d % 2) * 4)) & 15;
"""
    raise ValueError(f"bits must be 2, 3, or 4, got {bits}")


def _packed_d(D: int, bits: int) -> int:
    if bits == 3:
        return (D // 32) * 12
    if bits == 2:
        return D // 4
    return D // 2  # bits == 4


def _get_k_dequant_kernel(D: int, Hkv: int, block_size: int, bits: int, out_dtype):
    t = _msl_type(out_dtype)
    key = (D, Hkv, block_size, bits, t)
    kern = _K_DEQUANT_KERNELS.get(key)
    if kern is None:
        pd = _packed_d(D, bits)
        src = f"""
  const uint gid = thread_position_in_grid.x;
  const int S = params[0];
  const int num_blocks = params[1];
  const int n_blk = params[2];
  const uint total = (uint)S * {Hkv} * {D};
  if (gid >= total) return;
  const int d = (int)(gid % {D});
  const int h = (int)((gid / {D}) % {Hkv});
  const int s = (int)(gid / ({D} * {Hkv}));
  const int blk = s / {block_size};
  const int tok = s % {block_size};
  {t} kout_v = ({t})0;
  // CX-TQ-DECODE-01: bounds-guard the logical table index (blk < n_blk) AND the
  // physical block id (0 <= phys < num_blocks) before ANY pool/scale/centroid
  // load. blk past the active table, or phys out of range (incl. -1 padding) →
  // skip (zero), never an out-of-bounds load. Matches the guarded C++ paged
  // gather (phys >= 0 && phys < num_blocks).
  if (blk < n_blk) {{
    const int phys = block_table[blk];
    if (phys >= 0 && phys < num_blocks) {{
      const ulong pk_base = (ulong)phys * {block_size * Hkv * pd}
                          + (ulong)tok * {Hkv * pd}
                          + (ulong)h * {pd};
{_unpack_snippet(bits)}
      const float scl = k_scales[(ulong)phys * {block_size * Hkv}
                               + (ulong)tok * {Hkv} + h];
      kout_v = ({t})((float)centroids[idx] * scl);
    }}
  }}
  // out layout: [1, Hkv, S, D]
  Kout[((ulong)h * (ulong)S + (ulong)s) * {D} + d] = kout_v;
"""
        kern = mx.fast.metal_kernel(
            name=f"tq_decode_kdequant_b{bits}_d{D}_h{Hkv}_bs{block_size}_{t}",
            input_names=["k_pool", "k_scales", "centroids",
                         "block_table", "params"],
            output_names=["Kout"],
            source=src, header=_HEADER, ensure_row_contiguous=True)
        _K_DEQUANT_KERNELS[key] = kern
    return kern


def _get_v_gather_kernel(D: int, Hkv: int, block_size: int, out_dtype):
    t = _msl_type(out_dtype)
    key = (D, Hkv, block_size, t)
    kern = _V_GATHER_KERNELS.get(key)
    if kern is None:
        src = f"""
  const uint gid = thread_position_in_grid.x;
  const int S = params[0];
  const int num_blocks = params[1];
  const int n_blk = params[2];
  const uint total = (uint)S * {Hkv} * {D};
  if (gid >= total) return;
  const int d = (int)(gid % {D});
  const int h = (int)((gid / {D}) % {Hkv});
  const int s = (int)(gid / ({D} * {Hkv}));
  const int blk = s / {block_size};
  const int tok = s % {block_size};
  {t} vout_v = ({t})0;
  // CX-TQ-DECODE-01: same bounds guard as the K kernel — phys out of
  // [0,num_blocks) (incl. -1 padding) or blk past the active table → zero.
  if (blk < n_blk) {{
    const int phys = block_table[blk];
    if (phys >= 0 && phys < num_blocks) {{
      vout_v = v_pool[(ulong)phys * {block_size * Hkv * D}
                    + (ulong)tok * {Hkv * D} + (ulong)h * {D} + d];
    }}
  }}
  Vout[((ulong)h * (ulong)S + (ulong)s) * {D} + d] = vout_v;
"""
        kern = mx.fast.metal_kernel(
            name=f"tq_decode_vgather_d{D}_h{Hkv}_bs{block_size}_{t}",
            input_names=["v_pool", "block_table", "params"],
            output_names=["Vout"],
            source=src, header=_HEADER, ensure_row_contiguous=True)
        _V_GATHER_KERNELS[key] = kern
    return kern


def tq_decode_attend(
    q_rot: mx.array,
    k_pool_tq: mx.array,
    v_pool_fp16: mx.array,
    k_scales: mx.array,
    k_centroids: mx.array,
    block_table_row: mx.array,
    seq_len: int,
    *,
    scale: Optional[float] = None,
    block_size: int,
    tq_bits: int,
    stream=None,
) -> mx.array:
    """Decode attend over a TQ-packed paged K pool + fp16 paged V pool.

    Args:
        q_rot: [1, H_q, N_q, D] WHT-rotated queries (fp16).
        k_pool_tq: [num_blocks, block_size, H_kv, packed_D] uint8.
        v_pool_fp16: [num_blocks, block_size, H_kv, D] fp16.
        k_scales: [num_blocks, block_size, H_kv] float32.
        k_centroids: [2^bits] fp16.
        block_table_row: [n_active_blocks] int32 physical block ids for
            THIS sequence (active prefix of the block-table row).
        seq_len: tokens in the sequence (decode attends to all of them).
        scale: softmax scale (default 1/sqrt(D)).
        block_size, tq_bits: pool geometry.

    Returns [1, H_q, N_q, D] fp16 attention output.
    """
    _num_blocks, bs, Hkv, _pd = k_pool_tq.shape
    D = q_rot.shape[3]
    # M1 (CC final-cert): the K-dequant + V-gather kernels bake D from q_rot and
    # index the V pool / packed-K row by it; v_pool_fp16's real D and k_pool_tq's
    # packed_D were read (_pd) but NOT cross-checked → q_rot.D != v_pool.D drove an
    # OOB read = NaN (silent-wrong). The TQ *class* path is guarded (cache_dim),
    # but this RAW helper was not. Cross-check the pool geometry before dispatch.
    if v_pool_fp16.shape[3] != D:
        raise ValueError(
            f"tq_decode_attend: v_pool head_dim ({v_pool_fp16.shape[3]}) must equal "
            f"q head_dim ({D}); the V-gather reads V at q's D and would read OOB.")
    if _pd != _packed_d(D, tq_bits):
        raise ValueError(
            f"tq_decode_attend: k_pool_tq packed_D ({_pd}) != expected "
            f"{_packed_d(D, tq_bits)} for D={D}, tq_bits={tq_bits}.")
    if tuple(v_pool_fp16.shape[:3]) != (_num_blocks, bs, Hkv):
        raise ValueError(
            f"tq_decode_attend: v_pool_fp16 [num_blocks,block_size,H_kv] "
            f"{tuple(v_pool_fp16.shape[:3])} must match k_pool_tq {(_num_blocks, bs, Hkv)}.")
    # sweep iter-3 (subset-derive OOB): k_scales is the last sibling pool array left
    # unchecked. The K-dequant kernel indexes k_scales at k_pool's block stride
    # (phys guarded only against num_blocks), so an undersized k_scales (e.g. 2
    # blocks vs num_blocks=8) reads OOB → finite NON-deterministic (verified). Mirror
    # the v_pool / packed_D guards above (production step allocates them together; the
    # exposure is this raw helper).
    if tuple(k_scales.shape) != (_num_blocks, bs, Hkv):
        raise ValueError(
            f"tq_decode_attend: k_scales {tuple(k_scales.shape)} must equal "
            f"(num_blocks, block_size, H_kv) {(_num_blocks, bs, Hkv)}; the K-dequant "
            "kernel indexes k_scales at k_pool's block stride and would read OOB.")
    # codebook extent (latent-UB defense; sweep iter-6 audit): the K-dequant kernel
    # indexes k_centroids[code] with code in [0, 2**tq_bits) but never bounds-checks
    # the table length. On M5 an undersized codebook clamps/over-allocates (not a
    # demonstrable within-process silent-wrong), but `centroids[idx]` with idx past
    # the buffer is genuine UB — guard it for class-completeness with the sibling pool
    # checks (mirrors flash_attention_paged_varlen_turboquant's centroid-extent guard).
    if k_centroids.shape[0] < (1 << int(tq_bits)):
        raise ValueError(
            f"tq_decode_attend: k_centroids has {k_centroids.shape[0]} entries but "
            f"tq_bits={tq_bits} indexes {1 << int(tq_bits)} codes — the K-dequant "
            "would index past the codebook.")
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    S = int(seq_len)
    # CX-TQ-DECODE-01: pass num_blocks (physical pool size) + n_active_blocks
    # (logical table-row length) so both kernels can bounds-guard phys / blk.
    n_active_blocks = int(block_table_row.shape[0])
    # seq_len capacity contract (sweep iter-5; scope-decided iter-6): the K/V gather
    # kernels zero-FILL positions whose block index >= n_active_blocks, then SDPA
    # softmaxes over all S positions — so seq_len > n_active_blocks*block_size
    # silently dilutes the result with zero-key positions (finite-WRONG, no raise).
    # NOT gated on MFA_PAGED_TRUST_INDICES (deliberate scope decision): that flag is
    # a PERF opt-out for the EXPENSIVE per-element block_table/seq_lens value-range
    # scan (a .tolist() GPU sync). This capacity check is a CHEAP scalar comparison
    # (Python ints, no GPU sync) on a STRUCTURAL contract (seq_len vs table length),
    # a different axis — gating it would save no perf and would MOVE the silent-wrong
    # under the flag (verified: under the opt-out, over-capacity stayed memory-safe
    # but finite-WRONG, dilution err 0.31 vs the honest capacity reference). So it
    # always applies (Rule 8), even under the opt-out; the flag's scope stays exactly
    # the index/seqlen value-range sync, not capacity.
    if S > n_active_blocks * bs:
        raise ValueError(
            f"tq_decode_attend: seq_len ({S}) exceeds block-table capacity "
            f"(n_active_blocks={n_active_blocks} * block_size={bs} = "
            f"{n_active_blocks * bs}); excess positions would silently attend to "
            "zero-filled K/V and dilute the output.")
    params = mx.array([S, int(_num_blocks), n_active_blocks], dtype=mx.int32)

    # P9: emit K/V in the CACHE dtype natively (the V pool + rotated q are at the
    # context dtype) — bf16-native for a bf16 cache (no lossy fp16 round-trip),
    # byte-identical `half` for fp16.  K/V/q then share dtype for SDPA.
    out_dtype = v_pool_fp16.dtype
    kkern = _get_k_dequant_kernel(D, Hkv, bs, tq_bits, out_dtype)
    vkern = _get_v_gather_kernel(D, Hkv, bs, out_dtype)
    total = S * Hkv * D
    grid = ((total + 255) // 256 * 256, 1, 1)
    tg = (256, 1, 1)
    K = kkern(inputs=[k_pool_tq, k_scales, k_centroids,
                      block_table_row, params],
              output_shapes=[(1, Hkv, S, D)], output_dtypes=[out_dtype],
              grid=grid, threadgroup=tg)[0]
    V = vkern(inputs=[v_pool_fp16, block_table_row, params],
              output_shapes=[(1, Hkv, S, D)], output_dtypes=[out_dtype],
              grid=grid, threadgroup=tg)[0]
    return mx.fast.scaled_dot_product_attention(
        q_rot, K, V, scale=scale, stream=stream) if stream is not None \
        else mx.fast.scaled_dot_product_attention(q_rot, K, V, scale=scale)
