"""Conv3D NAX forward — Phase 1.2 multi-chunk implementation.

Scope (cumulative through Phase 1.2):
  - All single-chunk shapes (mid_resnet from Phase 1.1)
  - Multi-chunk shapes (up1_resnet via M-chunking; Phase 1.3 generalizes)
  - Asymmetric / causal pad_T (pad_T_left != pad_T_right)
  - K_T=1 special case (effectively 2D conv per temporal slice)
  - All forward, FP16/BF16, channels-last layout

Per docs/conv-nax/conv-nax-design.md decision D3: this module orchestrates
two JIT Metal kernels via mx.fast.metal_kernel:

  1. Im2col3D — gathers (M_chunk, K_T*K_H*K_W*C_in) buffer from full
     (B, T, H, W, C_in) input via stride/pad/dil addressing. Takes
     M_OFFSET as a compile-time constant so chunks read correct input
     positions.
  2. matmul2d  — lifts the V6-NAX-validated MPP matmul2d kernel from
     bench/conv_nax_matmul2d_microbench.py. With M_chunk × K × 2 bytes
     < 2 GB per chunk, the int32-byte-address overflow bug in MPP
     matmul2d (Phase 1.2 root-cause, fired at row 77696 for K=13824)
     is avoided.

Chunking heuristic (Phase 1.2 — refined in Phase 1.3):
  chunk_M_max = 2^31 / (K * dtype_bytes), aligned down to M_TILE=32.
  Then n_chunks = ceil(M_total / chunk_M_max),
       chunk_M  = round_up(M_total / n_chunks, M_TILE).

Layout (channels-last, matches mx.conv_general convention):
  input  : (B, T, H, W, C_in)
  weight : (C_out, K_T, K_H, K_W, C_in)
  output : (B, T_out, H_out, W_out, C_out)

Implementation note: full MFAConv3DForward C++ Primitive class is
DEFERRED to Phase 1.5+ post-verdict. This Python orchestration delivers
the same correctness contract as the prescribed C++ Primitive (Phase 1.1
prompt B.2) -- the matmul2d kernel IS the perf-critical path, and
~50-100us Python dispatch overhead is bounded vs the ~6ms+ kernel time
on mid_resnet. Conversion to C++ Primitive is mechanical if Phase 1.5
ship-default verdict is reached.
"""
from __future__ import annotations

from typing import Tuple, Optional
import mlx.core as mx


# ---------------------------------------------------------------------
# Cache for compiled kernels (ConvKey -> (im2col_kernel, matmul_kernel)).
# Keys mirror design D3 unified ConvKey schema.
# ---------------------------------------------------------------------
_KERNEL_CACHE: dict = {}


def _conv_key(B, T, H, W, C_in, T_out, H_out, W_out, C_out,
              K_T, K_H, K_W, sT, sH, sW,
              pT_l, pT_r, pH_l, pH_r, pW_l, pW_r,
              dT, dH, dW, dtype, m_offset, m_chunk):
    """ConvKey per design D3: (Kind=Conv3DForward, shape/conv params, chunk).

    Per Phase 1.2: keyed by (m_offset, m_chunk) so each chunk gets its own
    compiled (im2col, matmul) pair. Asymmetric padding fully encoded.
    """
    return ("Conv3DForward", B, T, H, W, C_in, T_out, H_out, W_out, C_out,
            K_T, K_H, K_W, sT, sH, sW,
            pT_l, pT_r, pH_l, pH_r, pW_l, pW_r,
            dT, dH, dW, str(dtype), m_offset, m_chunk)


# Phase 1.2 root-cause: MPP matmul2d uses int32 internally for byte
# addresses. The im2col output (size M_chunk × K × dtype_bytes) must
# stay below 2^31 bytes (2 GiB) to avoid silent NaN-producing overflow
# at byte_addr >= 2^31. Tested: at K=13824 f16, NaN starts deterministically
# at row 77696 in the output buffer.
INT32_BYTE_BUDGET = 2**31  # 2,147,483,648 bytes
SAFETY_HEADROOM = 0.875    # Use 87.5% of int32 max, accounting for matmul2d
                           # internal address tricks (cooperative tensor +
                           # tile offsets + per-thread fragment addresses).
                           # Empirical: 0.95 * 2^31 fires NaN at K=13824.


# ---------------------------------------------------------------------
# Kernel source generators.
# ---------------------------------------------------------------------
_MATMUL_HEADER = """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

_IM2COL_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# Validated tile config from microbench (matches V6 NAX, hits ~45 TF).
M_TILE = 32
N_TILE = 32
K_TILE = 32
EXEC_SIMDGROUPS = 1
TG_THREADS = 32 * EXEC_SIMDGROUPS


def _round_up_k_tile(K: int) -> int:
    return ((K + K_TILE - 1) // K_TILE) * K_TILE


def _pad_k(a: "mx.array", K: int, k_pad: int) -> "mx.array":
    """Sprint III-6: zero-pad the contraction (last) axis up to a K_TILE
    multiple.  _matmul2d_source's K-loop reads the final tile past K_FULL
    when K % K_TILE != 0 (the slice<K_TILE,…> exceeds the tensor extent ->
    OOB garbage; C_in=16 -> 0.10 MAE/RMS, C_in=31 -> NaN).  Zero
    contraction terms contribute nothing, so the result is EXACT and the
    K-loop only reads in-bounds.  Mirrors the C++ fix in mfa_conv_nax.cpp."""
    if k_pad == K:
        return a
    pad_widths = [(0, 0)] * a.ndim
    pad_widths[-1] = (0, k_pad - K)
    return mx.pad(a, pad_widths)


def _matmul2d_source(M: int, K: int, N: int) -> str:
    """Conv3D-specific matmul: C(M,N) = A(M,K) @ B(N,K)^T  via rightT=true.

    Matches V6 NAX pattern (NAAttentionKernel.cpp:775 also uses rightT=true
    for Q @ K^T). The microbench's variant uses rightT=false because it
    intentionally measures the A@B pattern with B in (K,N) layout.

    Both inputs are row-major in Python:
      A : (M, K)  -- im2col buffer
      B : (N, K)  -- flattened weight (C_out, K_T*K_H*K_W*C_in)
    Output:
      C : (M, N)  row-major

    III-6 Rule-8 contract: the K-loop steps K_TILE at a time over [0, K)
    and does NOT mask a partial final tile, so it is correct ONLY when K
    is a multiple of K_TILE.  Callers MUST zero-pad the contraction
    (_pad_k).  Refuse to generate for an unaligned K so any future caller
    that forgets to pad fails LOUDLY here rather than silently reading
    past the tensor extent (the small-channel corruption this sprint fixed).
    """
    if K % K_TILE != 0:
        raise ValueError(
            f"_matmul2d_source: K={K} is not a multiple of K_TILE={K_TILE} "
            "— the contraction must be zero-padded (_pad_k) before dispatch, "
            "else the K-loop reads past the tensor extent (silent corruption)."
        )
    return f"""
    constexpr uint M_FULL = {M};
    constexpr uint K_FULL = {K};
    constexpr uint N_FULL = {N};

    auto tA = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)A, dextents<int32_t, 2>(K_FULL, M_FULL));
    // B is (N, K) row-major in Python; matmul2d rightT=true will read it
    // as the K-major operand and transpose internally.
    auto tB = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        (device half*)B, dextents<int32_t, 2>(K_FULL, N_FULL));

    const uint m_origin = threadgroup_position_in_grid.y * {M_TILE};
    const uint n_origin = threadgroup_position_in_grid.x * {N_TILE};

    constexpr auto desc = matmul2d_descriptor(
        {M_TILE}, {N_TILE}, {K_TILE},
        false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<{EXEC_SIMDGROUPS}>> op;

    auto mA_init = tA.slice<{K_TILE}, {M_TILE}>(0, m_origin);
    auto mB_init = tB.slice<{K_TILE}, {N_TILE}>(0, n_origin);

    auto cC = op.get_destination_cooperative_tensor<
        decltype(mA_init), decltype(mB_init), float>();
    #pragma clang loop unroll(full)
    for (ushort k = 0; k < cC.get_capacity(); ++k) {{
        if (cC.is_valid_element(k)) cC[k] = 0.0f;
    }}

    for (uint k_start = 0; k_start < K_FULL; k_start += {K_TILE}) {{
        auto mA_k = tA.slice<{K_TILE}, {M_TILE}>(k_start, m_origin);
        auto mB_k = tB.slice<{K_TILE}, {N_TILE}>(k_start, n_origin);
        op.run(mA_k, mB_k, cC);
    }}

    #pragma clang loop unroll(full)
    for (ushort k = 0; k < cC.get_capacity(); ++k) {{
        if (cC.is_valid_element(k)) {{
            auto idx = cC.get_multidimensional_index(k);
            uint m_global = m_origin + idx[1];
            uint n_global = n_origin + idx[0];
            if (m_global < M_FULL && n_global < N_FULL) {{
                C[m_global * N_FULL + n_global] = (half)cC[k];
            }}
        }}
    }}
"""


def _im2col3d_source(B, T, H, W, C_in, T_out, H_out, W_out,
                     K_T, K_H, K_W, sT, sH, sW,
                     pT_l, pH_l, pW_l, dT, dH, dW,
                     m_offset, m_chunk) -> str:
    """Im2col for Conv3D channels-last with per-chunk m_offset.

    Output: (M_CHUNK, K) row-major where
      M_CHUNK = m_chunk (size of this chunk's row count, <= M_total)
      K = K_T*K_H*K_W*C_in
    Each thread writes one (m_local, k) element.

    m_global = m_local + M_OFFSET is used to compute (b, t_out, h_out, w_out).
    Input layout: (B, T, H, W, C_in) row-major.
    Output layout: (M_CHUNK, K) row-major.
    Padding: zero-fill for out-of-bounds spatial coords.

    pT_l, pH_l, pW_l are the LEFT (low-index) paddings. The im2col only
    needs the left-pad because the right-pad's effect on output T_out is
    handled by the Python caller in T_out computation; here we only need
    to translate output coords to input coords.
    """
    return f"""
    constexpr uint cB    = {B};
    constexpr uint cT    = {T};
    constexpr uint cH    = {H};
    constexpr uint cW    = {W};
    constexpr uint cCin  = {C_in};
    constexpr uint cTout = {T_out};
    constexpr uint cHout = {H_out};
    constexpr uint cWout = {W_out};
    constexpr uint cKT   = {K_T};
    constexpr uint cKH   = {K_H};
    constexpr uint cKW   = {K_W};
    constexpr int  csT   = {sT};
    constexpr int  csH   = {sH};
    constexpr int  csW   = {sW};
    constexpr int  cpTl  = {pT_l};
    constexpr int  cpHl  = {pH_l};
    constexpr int  cpWl  = {pW_l};
    constexpr int  cdT   = {dT};
    constexpr int  cdH   = {dH};
    constexpr int  cdW   = {dW};
    constexpr uint cMoff = {m_offset};
    constexpr uint cMchk = {m_chunk};
    constexpr uint cKvol = cKT * cKH * cKW;
    constexpr uint cKfull = cKvol * cCin;

    // Thread index covers (m_local, k) flattened.
    uint tid = thread_position_in_grid.x;
    if (tid >= cMchk * cKfull) return;
    uint m_local = tid / cKfull;
    uint k = tid - m_local * cKfull;
    uint m_global = m_local + cMoff;

    // Unravel m_global -> (b, t_out, h_out, w_out)
    uint rem_m = m_global;
    uint w_out = rem_m % cWout; rem_m /= cWout;
    uint h_out = rem_m % cHout; rem_m /= cHout;
    uint t_out = rem_m % cTout; rem_m /= cTout;
    uint b     = rem_m;

    // Unravel k -> (k_t, k_h, k_w, c_in)
    uint rem_k = k;
    uint c_in = rem_k % cCin; rem_k /= cCin;
    uint k_w  = rem_k % cKW;  rem_k /= cKW;
    uint k_h  = rem_k % cKH;  rem_k /= cKH;
    uint k_t  = rem_k;

    int t_in = (int)t_out * csT + (int)k_t * cdT - cpTl;
    int h_in = (int)h_out * csH + (int)k_h * cdH - cpHl;
    int w_in = (int)w_out * csW + (int)k_w * cdW - cpWl;

    half v = (half)0.0h;
    if (t_in >= 0 && t_in < (int)cT &&
        h_in >= 0 && h_in < (int)cH &&
        w_in >= 0 && w_in < (int)cW) {{
        uint in_idx = ((b * cT + (uint)t_in) * cH + (uint)h_in) * cW * cCin
                     + (uint)w_in * cCin + c_in;
        v = X[in_idx];
    }}
    // Write to (m_local, k) within this chunk's buffer.
    Im2col[m_local * cKfull + k] = v;
"""


def _make_kernels(key, m_chunk, K, N, B, T, H, W, C_in, T_out, H_out, W_out,
                  C_out, K_T, K_H, K_W, sT, sH, sW,
                  pT_l, pT_r, pH_l, pH_r, pW_l, pW_r,
                  dT, dH, dW, m_offset):
    """Compile and cache (im2col, matmul) for one chunk.

    m_chunk = number of output rows in THIS chunk.
    m_offset = where this chunk's rows start in the full M.

    pT_r/pH_r/pW_r aren't used inside the im2col kernel (only pT_l etc.)
    but are still part of the ConvKey so asymmetric pad configurations
    cache distinctly. Right-pad affects T_out/H_out/W_out which are
    computed Python-side.
    """
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    im2col = mx.fast.metal_kernel(
        name=f"im2col3d_{B}_{T}_{H}_{W}_{C_in}_{K_T}{K_H}{K_W}_"
             f"s{sT}{sH}{sW}_pl{pT_l}{pH_l}{pW_l}_pr{pT_r}{pH_r}{pW_r}_"
             f"d{dT}{dH}{dW}_off{m_offset}_chk{m_chunk}",
        input_names=["X"],
        output_names=["Im2col"],
        source=_im2col3d_source(B, T, H, W, C_in, T_out, H_out, W_out,
                                K_T, K_H, K_W, sT, sH, sW,
                                pT_l, pH_l, pW_l, dT, dH, dW,
                                m_offset, m_chunk),
        header=_IM2COL_HEADER,
        ensure_row_contiguous=True,
    )
    # III-6: matmul contracts over the K_TILE-padded K (operands are
    # zero-padded by the caller); im2col above stays at the true K.
    k_pad = _round_up_k_tile(K)
    mm = mx.fast.metal_kernel(
        name=f"conv3d_matmul2d_{m_chunk}_{k_pad}_{N}",
        input_names=["A", "B"],
        output_names=["C"],
        source=_matmul2d_source(m_chunk, k_pad, N),
        header=_MATMUL_HEADER,
        ensure_row_contiguous=True,
    )
    _KERNEL_CACHE[key] = (im2col, mm)
    return im2col, mm


def _compute_chunk_layout(M_total, K, dtype_bytes):
    """Plan M-chunks so each chunk's im2col buffer < INT32_BYTE_BUDGET * SAFETY.

    Returns list of (m_offset, m_chunk) pairs summing to M_total.
    Chunks are uniform M_TILE-aligned except possibly the last.
    """
    max_chunk_bytes = int(INT32_BYTE_BUDGET * SAFETY_HEADROOM)
    max_chunk_M = max_chunk_bytes // (K * dtype_bytes)
    # Align down to M_TILE so each chunk is tile-aligned (helps matmul
    # remainder handling).
    max_chunk_M = (max_chunk_M // M_TILE) * M_TILE
    if max_chunk_M < M_TILE:
        # Pathological: K so large that even one tile-row overflows int32.
        # This is a structural blocker; would need a different algorithm.
        raise ValueError(
            f"conv_nax: K={K} too large for chunking — even one M_TILE row "
            f"({M_TILE * K * dtype_bytes} bytes) overflows int32 budget."
        )

    if M_total <= max_chunk_M:
        return [(0, M_total)]

    n_chunks = (M_total + max_chunk_M - 1) // max_chunk_M
    # Distribute M_total approximately evenly across n_chunks, aligned to M_TILE.
    base = (M_total // n_chunks // M_TILE) * M_TILE
    if base == 0:
        base = M_TILE
    chunks = []
    remaining = M_total
    offset = 0
    for i in range(n_chunks):
        if i == n_chunks - 1:
            m_chunk = remaining
        else:
            m_chunk = base
            remaining -= m_chunk
        chunks.append((offset, m_chunk))
        offset += m_chunk
    # Verify total
    total = sum(c[1] for c in chunks)
    assert total == M_total, f"chunk layout {chunks} sums to {total}, want {M_total}"
    return chunks


# ---------------------------------------------------------------------
# Phase 1.3 — working-set instrumentation.
#
# Per design doc §4.2.3 + prompt §D: peak im2col allocation + matmul
# output per chunk; per-shape working set; total transient memory.
# The Python orchestrator path doesn't ping-pong buffers (each chunk
# allocates fresh), so peak is bounded by max(chunk_M × K + chunk_M × N).
# ---------------------------------------------------------------------

PHASE_1_3_WORKING_SET_HARD_GATE = 16 * 1024**3  # 16 GB


def estimate_working_set(M_total, K, N, dtype_bytes=2):
    """Estimate per-chunk + total transient working set in bytes.

    Returns a dict:
      - chunks: list of (m_offset, m_chunk)
      - per_chunk_im2col_bytes: max across chunks
      - per_chunk_matmul_out_bytes: max across chunks
      - per_chunk_peak_bytes: per_chunk im2col + per_chunk matmul_out
      - concat_out_bytes: M_total × N × dtype (only if >1 chunk)
      - total_peak_bytes: rough upper bound on peak GPU allocation
        (~= max chunk peak + concat output, assuming MLX can reclaim
        chunk i's im2col before chunk i+1's matmul output is allocated;
        in practice MLX's lazy graph may hold all chunk outputs until
        the concat — so we report the conservative concat_held estimate.)
      - within_hard_gate: bool, true iff total_peak_bytes < 16 GB
    """
    chunks = _compute_chunk_layout(M_total, K, dtype_bytes)
    per_chunk_im2col = max(c[1] for c in chunks) * K * dtype_bytes
    per_chunk_matmul_out = max(c[1] for c in chunks) * N * dtype_bytes
    per_chunk_peak = per_chunk_im2col + per_chunk_matmul_out

    if len(chunks) > 1:
        # MLX lazy: chunk outputs accumulate until concat, plus current
        # chunk's im2col is live during its matmul.
        concat_out_bytes = M_total * N * dtype_bytes
        # Conservative upper bound: all chunk outputs held + current im2col.
        total_peak_bytes = concat_out_bytes + per_chunk_im2col
    else:
        concat_out_bytes = 0
        total_peak_bytes = per_chunk_peak

    return {
        "chunks": chunks,
        "n_chunks": len(chunks),
        "per_chunk_im2col_bytes": per_chunk_im2col,
        "per_chunk_matmul_out_bytes": per_chunk_matmul_out,
        "per_chunk_peak_bytes": per_chunk_peak,
        "concat_out_bytes": concat_out_bytes,
        "total_peak_bytes": total_peak_bytes,
        "within_hard_gate": total_peak_bytes < PHASE_1_3_WORKING_SET_HARD_GATE,
        "hard_gate_bytes": PHASE_1_3_WORKING_SET_HARD_GATE,
    }


# ---------------------------------------------------------------------
# Padding normalization: accept int, 3-tuple of int (symmetric), or
# 3-tuple of (left,right) pairs (asymmetric, including causal pad_T).
# ---------------------------------------------------------------------
def _normalize_padding(padding):
    """Return ((pT_l,pT_r), (pH_l,pH_r), (pW_l,pW_r))."""
    if isinstance(padding, int):
        v = padding
        return ((v, v), (v, v), (v, v))
    if not (isinstance(padding, (tuple, list)) and len(padding) == 3):
        raise ValueError(f"conv_nax: padding must be int or 3-tuple, got {padding}")
    out = []
    for i, p in enumerate(padding):
        if isinstance(p, int):
            out.append((p, p))
        elif isinstance(p, (tuple, list)) and len(p) == 2 and \
             all(isinstance(x, int) for x in p):
            out.append((int(p[0]), int(p[1])))
        else:
            raise ValueError(
                f"conv_nax: padding axis {i} must be int or (left,right) "
                f"int-pair, got {p}"
            )
    return tuple(out)


# ---------------------------------------------------------------------
# Sanity asserts (design doc §4, 8 categories).
# ---------------------------------------------------------------------
def _sanity_asserts(x: mx.array, w: mx.array, stride, padding, dilation):
    """Throw if Phase 1.2 constraints not met. Returns shape tuple."""
    # Category 1: dtype
    if x.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError(f"conv_nax: dtype {x.dtype} not in (f16, bf16)")
    if x.dtype != w.dtype:
        raise ValueError(f"conv_nax: x.dtype={x.dtype} != w.dtype={w.dtype}")

    # Category 2: rank
    if x.ndim != 5:
        raise ValueError(f"conv_nax: input must be 5D (B,T,H,W,C_in), got "
                         f"shape={x.shape}")
    if w.ndim != 5:
        raise ValueError(f"conv_nax: weight must be 5D (C_out,K_T,K_H,K_W,C_in)"
                         f", got shape={w.shape}")

    # Category 3: channel match
    if x.shape[-1] != w.shape[-1]:
        raise ValueError(f"conv_nax: C_in mismatch x={x.shape[-1]} "
                         f"w={w.shape[-1]}")

    # Category 4: stride/dilation triples + padding normalization
    for name, v in [("stride", stride), ("dilation", dilation)]:
        if not (isinstance(v, (tuple, list)) and len(v) == 3):
            raise ValueError(f"conv_nax: {name} must be 3-tuple (T,H,W); got {v}")
        for vi in v:
            if not isinstance(vi, int) or vi < 1:
                raise ValueError(f"conv_nax: {name}={v} contains invalid int")
    pad_norm = _normalize_padding(padding)
    (pT_l, pT_r), (pH_l, pH_r), (pW_l, pW_r) = pad_norm
    for pname, pv in (("pT_l", pT_l), ("pT_r", pT_r), ("pH_l", pH_l),
                      ("pH_r", pH_r), ("pW_l", pW_l), ("pW_r", pW_r)):
        if pv < 0:
            raise ValueError(f"conv_nax: {pname}={pv} negative")

    # Category 5: kernel size positive
    for i, ax in enumerate(("K_T", "K_H", "K_W")):
        if w.shape[1 + i] < 1:
            raise ValueError(f"conv_nax: {ax}={w.shape[1+i]} < 1")

    # Category 6: input spatial extent must accommodate kernel after pad
    B, T, H, W, C_in = x.shape
    C_out, K_T, K_H, K_W, _ = w.shape
    sT, sH, sW = stride
    dT, dH, dW = dilation
    eff_T = T + pT_l + pT_r - dT * (K_T - 1) - 1
    eff_H = H + pH_l + pH_r - dH * (K_H - 1) - 1
    eff_W = W + pW_l + pW_r - dW * (K_W - 1) - 1
    if eff_T < 0 or eff_H < 0 or eff_W < 0:
        raise ValueError(f"conv_nax: input too small for kernel after padding: "
                         f"eff_T={eff_T} eff_H={eff_H} eff_W={eff_W}")

    T_out = eff_T // sT + 1
    H_out = eff_H // sH + 1
    W_out = eff_W // sW + 1
    M = B * T_out * H_out * W_out
    K = C_in * K_T * K_H * K_W
    dtype_bytes = 2  # f16 / bf16

    # Category 7: working-set feasibility (Phase 1.3 hard gate).
    # Per design §4.2.3: peak transient = MLX-held chunk outputs +
    # in-flight im2col. Estimated via estimate_working_set.
    ws = estimate_working_set(M, K, C_out, dtype_bytes)
    if not ws["within_hard_gate"]:
        raise ValueError(
            f"conv_nax: estimated peak working set "
            f"{ws['total_peak_bytes']/1e9:.2f} GB exceeds Phase 1.3 hard gate "
            f"({ws['hard_gate_bytes']/1e9:.0f} GB). "
            f"chunks={ws['n_chunks']}, per_chunk_im2col="
            f"{ws['per_chunk_im2col_bytes']/1e9:.2f} GB, "
            f"concat_out={ws['concat_out_bytes']/1e9:.2f} GB. "
            f"Shape too large for unstreamed evaluation; use mx.conv_general."
        )

    # Category 8: alignment / plausibility
    if C_out <= 0 or C_out > 65536:
        raise ValueError(f"conv_nax: implausible C_out={C_out}")

    return (B, T, H, W, C_in, T_out, H_out, W_out, C_out,
            K_T, K_H, K_W, M, K, pad_norm)


# ---------------------------------------------------------------------
# Phase 1.4 — 1×1×1 fast path internals.
# ---------------------------------------------------------------------
import os


def _conv_disable_fast_path() -> bool:
    """Env-var escape hatch: MFA_CONV_NAX_NO_FAST_PATH=1 forces general path.

    Used by tests to compare fast-path output against general-path output
    (sanity), and by callers who need to bypass the fast path for any
    diagnostic reason.
    """
    return os.environ.get("MFA_CONV_NAX_NO_FAST_PATH", "") == "1"


def _make_pointwise_matmul_kernel(m_chunk: int, K: int, N: int, dtype):
    """Compile (or fetch from cache) a matmul kernel for a 1×1×1 chunk.

    The matmul kernel here is the SAME as the general path's (no im2col
    transpose tricks needed) -- the only difference is that we skip the
    im2col preamble.
    """
    key = ("Pointwise1x1x1Matmul", m_chunk, K, N, str(dtype))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    kernel = mx.fast.metal_kernel(
        name=f"conv3d_1x1x1_matmul2d_{m_chunk}_{K}_{N}",
        input_names=["A", "B"],
        output_names=["C"],
        source=_matmul2d_source(m_chunk, K, N),
        header=_MATMUL_HEADER,
        ensure_row_contiguous=True,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def _dispatch_1x1x1_fast_path(x_flat, w_flat, chunks, M, K_eff, N,
                              B, T_out, H_out, W_out, C_out, dtype):
    """Run the chunked matmul-only path for 1×1×1 conv.

    Mirrors the general path's chunk loop but skips im2col. K here is
    C_in (not 27 × C_in).
    """
    chunk_outputs = []
    n_chunks = len(chunks)
    force_per_chunk_eval = n_chunks > 1

    # III-6: pad the contraction K to a K_TILE multiple (matmul2d partial-
    # K-tile fix); w_flat is loop-invariant -> pad once.
    k_pad = _round_up_k_tile(K_eff)
    w_flat = _pad_k(w_flat, K_eff, k_pad)

    for (m_offset, m_chunk) in chunks:
        # Slice the flattened input to this chunk's rows. Row-major
        # contiguous, so this is metadata-only -- but with
        # ensure_row_contiguous=True on the kernel, MLX may still copy
        # the slice. For the single-chunk case (the common case), just
        # pass x_flat directly to avoid that copy.
        if n_chunks == 1:
            x_chunk = x_flat
        else:
            x_chunk = x_flat[m_offset:m_offset + m_chunk, :]
        x_chunk = _pad_k(x_chunk, K_eff, k_pad)

        mm_kernel = _make_pointwise_matmul_kernel(m_chunk, k_pad, N, dtype)
        n_tg_x = (N + N_TILE - 1) // N_TILE
        n_tg_y = (m_chunk + M_TILE - 1) // M_TILE
        chunk_flat = mm_kernel(
            inputs=[x_chunk, w_flat],
            output_shapes=[(m_chunk, N)],
            output_dtypes=[dtype],
            grid=(n_tg_x * TG_THREADS, n_tg_y, 1),
            threadgroup=(TG_THREADS, 1, 1),
        )[0]
        if force_per_chunk_eval:
            mx.async_eval(chunk_flat)
            mx.synchronize()
        chunk_outputs.append(chunk_flat)

    if len(chunk_outputs) == 1:
        flat = chunk_outputs[0]
    else:
        flat = mx.concatenate(chunk_outputs, axis=0)
    return flat.reshape(B, T_out, H_out, W_out, C_out)


# ---------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------
def _conv3d_nax_forward_python_legacy(
    x: mx.array,
    w: mx.array,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding=(0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    causal_pad_t: bool = False,
) -> mx.array:
    """Phase 1.x Python orchestrator (Sprint D legacy reference path).

    Preserved for Track D.1 migration validation tests + as a diagnostic
    fallback. Production users should call `conv3d_nax_forward()` which
    routes through the C++ `_ext.conv3d_nax_forward` binding.

    Equivalent to:
        mx.conv_general(x, w, stride=stride, padding=padding,
                        kernel_dilation=dilation)

    But routes through implicit-GEMM via MPP matmul2d. Phase 1.2 scope:
    - multi-chunk along M (auto-chunked when M × K × 2 > 1.75 GB)
    - forward only (no VJP)
    - fp16 ONLY (campaign 2026-06 Sprint A, A-8): the embedded matmul2d
      Metal source hardcodes ``device half`` buffer casts — a bf16 input
      would be bitwise type-punned as fp16 and produce silently wrong
      values.  bf16 is rejected loudly below (the C++ production path is
      also fp16-only per KD-7).
    - channels-last layout
    - symmetric OR asymmetric (causal) padding

    Args:
        x: input array, shape (B, T, H, W, C_in), dtype f16 or bf16.
        w: weight array, shape (C_out, K_T, K_H, K_W, C_in), same dtype.
        stride: (sT, sH, sW), default (1,1,1).
        padding: either int, 3-tuple of int (symmetric), or 3-tuple of
            (left, right) pairs (asymmetric). Default (0,0,0).
        dilation: (dT, dH, dW), default (1,1,1).
        causal_pad_t: if True, override pT to (K_T-1, 0) for causal
            temporal padding (no future-frame leakage). pH/pW from
            `padding` are still honored.

    Returns:
        Output array, shape (B, T_out, H_out, W_out, C_out), same dtype.

    Raises:
        ValueError: if any sanity check fails (8 categories, see source).
    """
    # Campaign 2026-06 Sprint A (A-8): the matmul2d kernel source casts
    # buffers as `device half*`.  bf16 through this path is a silent
    # type-pun (wrong values) — fail loudly per Rule 8.
    if x.dtype != mx.float16 or w.dtype != mx.float16:
        raise ValueError(
            "_conv3d_nax_forward_python_legacy supports fp16 only "
            f"(got x={x.dtype}, w={w.dtype}); the embedded matmul2d Metal "
            "source hardcodes half-precision buffer casts.  Use the C++ "
            "production path (conv3d_nax_forward) or mx.conv_general."
        )
    sT, sH, sW = stride
    dT, dH, dW = dilation

    # Apply causal_pad_t override BEFORE sanity asserts so the padding
    # 8-cat assert sees the final values.
    if causal_pad_t:
        K_T_for_causal = w.shape[1]
        if isinstance(padding, int):
            padding = ((K_T_for_causal - 1, 0), (padding, padding), (padding, padding))
        elif isinstance(padding, (tuple, list)) and len(padding) == 3:
            # Replace pT axis with the causal pair; keep pH, pW as given.
            padding = ((K_T_for_causal - 1, 0), padding[1], padding[2])

    (B, T, H, W, C_in, T_out, H_out, W_out, C_out,
     K_T, K_H, K_W, M, K, pad_norm) = _sanity_asserts(
        x, w, stride, padding, dilation)
    (pT_l, pT_r), (pH_l, pH_r), (pW_l, pW_r) = pad_norm
    N = C_out

    # ==================================================================
    # Phase 1.4 — 1×1×1 fast path.
    #
    # When K_T = K_H = K_W = 1 AND all paddings are zero AND all strides
    # are 1, the convolution degenerates to a pointwise matmul:
    #   y[b,t,h,w,c_out] = sum_{c_in} x[b,t,h,w,c_in] × w[c_out, c_in]
    # No spatial expansion: K = C_in (not 27 × C_in for 3×3×3).
    # Im2col becomes the identity → we can dispatch matmul directly on
    # the input via metadata-only mx.reshape (channels-last layout makes
    # this a no-copy operation).
    #
    # See D26 in conv-nax-phase1_3-decisions.md for the design rationale.
    # ==================================================================
    is_pointwise = (
        K_T == 1 and K_H == 1 and K_W == 1
        and pT_l == 0 and pT_r == 0 and pH_l == 0 and pH_r == 0
        and pW_l == 0 and pW_r == 0
        and sT == 1 and sH == 1 and sW == 1
    )
    if is_pointwise and not _conv_disable_fast_path():
        # Reshape input (B, T, H, W, C_in) -> (M, C_in) via metadata only.
        # Since channels-last is row-major contiguous, the reshape is free.
        x_flat = x.reshape(M, C_in)
        # Weight is already (C_out, 1, 1, 1, C_in); flatten to (C_out, C_in).
        w_flat_pw = w.reshape(C_out, C_in)
        # Plan chunks based on the smaller K = C_in (vs K_T*K_H*K_W*C_in).
        dtype_bytes = 2
        chunks_pw = _compute_chunk_layout(M, C_in, dtype_bytes)
        return _dispatch_1x1x1_fast_path(
            x_flat, w_flat_pw, chunks_pw, M, C_in, N,
            B, T_out, H_out, W_out, C_out, x.dtype)

    # Reshape weight to (C_out, K_T*K_H*K_W*C_in) row-major (no copy if
    # channels-last input is contiguous over the last 4 dims).
    w_flat = w.reshape(C_out, K_T * K_H * K_W * C_in)

    # III-6: pad the contraction K to a K_TILE multiple (matmul2d partial-
    # K-tile fix); w_flat is loop-invariant -> pad once.  _make_kernels
    # builds the matmul at the same k_pad; im2col stays at the true K.
    k_pad = _round_up_k_tile(K)
    w_flat = _pad_k(w_flat, K, k_pad)

    # Plan M-chunks (Phase 1.2 chunking).
    dtype_bytes = 2  # f16 / bf16
    chunks = _compute_chunk_layout(M, K, dtype_bytes)

    chunk_outputs = []
    n_chunks = len(chunks)
    # When multi-chunking, force per-chunk eval so MLX's lazy graph
    # doesn't accumulate all chunks' im2col buffers simultaneously
    # (Phase 1.3 root-cause: observed 32 GB peak with 17 lazy-held
    # chunks of 1.81 GB each at 1.114 M shape). Per-chunk eval bounds
    # peak to one chunk's transient work + accumulated outputs.
    force_per_chunk_eval = n_chunks > 1
    for (m_offset, m_chunk) in chunks:
        key = _conv_key(B, T, H, W, C_in, T_out, H_out, W_out, C_out,
                        K_T, K_H, K_W, sT, sH, sW,
                        pT_l, pT_r, pH_l, pH_r, pW_l, pW_r,
                        dT, dH, dW, x.dtype, m_offset, m_chunk)
        im2col_kernel, mm_kernel = _make_kernels(
            key, m_chunk, K, N, B, T, H, W, C_in, T_out, H_out, W_out, C_out,
            K_T, K_H, K_W, sT, sH, sW,
            pT_l, pT_r, pH_l, pH_r, pW_l, pW_r,
            dT, dH, dW, m_offset)

        # Step 1: im2col for this chunk -- (m_chunk × K) elements.
        chunk_elems = m_chunk * K
        THREADS_PER_TG_IM2COL = 256
        grid_x = (chunk_elems + THREADS_PER_TG_IM2COL - 1) // THREADS_PER_TG_IM2COL
        im2col_buf = im2col_kernel(
            inputs=[x],
            output_shapes=[(m_chunk, K)],
            output_dtypes=[x.dtype],
            grid=(grid_x * THREADS_PER_TG_IM2COL, 1, 1),
            threadgroup=(THREADS_PER_TG_IM2COL, 1, 1),
        )[0]
        # III-6: pad the im2col buffer's K axis to match w_flat / the
        # k_pad-wide matmul kernel.
        im2col_buf = _pad_k(im2col_buf, K, k_pad)

        # Step 2: matmul2d (m_chunk, k_pad) @ (N, k_pad)^T = (m_chunk, N).
        n_tg_x = (N + N_TILE - 1) // N_TILE
        n_tg_y = (m_chunk + M_TILE - 1) // M_TILE
        chunk_flat = mm_kernel(
            inputs=[im2col_buf, w_flat],
            output_shapes=[(m_chunk, N)],
            output_dtypes=[x.dtype],
            grid=(n_tg_x * TG_THREADS, n_tg_y, 1),
            threadgroup=(TG_THREADS, 1, 1),
        )[0]
        if force_per_chunk_eval:
            # Force realization: chunk_flat now contains data; im2col_buf
            # is released by MLX's garbage collector. Next iteration starts
            # fresh, bounding peak transient memory to one chunk's worth.
            mx.async_eval(chunk_flat)
            mx.synchronize()
        chunk_outputs.append(chunk_flat)

    # Concatenate chunk outputs and reshape to (B, T_out, H_out, W_out, C_out).
    if len(chunk_outputs) == 1:
        flat = chunk_outputs[0]
    else:
        flat = mx.concatenate(chunk_outputs, axis=0)
    return flat.reshape(B, T_out, H_out, W_out, C_out)


def get_chunk_plan(M_total: int, K: int, dtype_bytes: int = 2):
    """Public helper: return the chunking plan for a given (M, K, dtype).

    Useful for tests and Phase 1.3 working-set instrumentation. Returns
    a list of (m_offset, m_chunk) tuples.
    """
    return _compute_chunk_layout(M_total, K, dtype_bytes)


# Campaign 2026-06 Sprint C Track 1 (#13): memoize the padding parse.
# VAE inference calls conv3d_nax_forward thousands of times with constant
# padding; the isinstance/unpack pass is per-call waste.  Manual memo (not
# lru_cache) so unhashable inputs (lists) fall through to direct parse.
_PADDING_MEMO: dict = {}


def _normalize_padding_to_6tuple(padding, K_T_for_causal=None, causal_pad_t=False):
    try:
        memo_key = (padding, K_T_for_causal, causal_pad_t)
    except TypeError:
        memo_key = None
    if memo_key is not None:
        cached = _PADDING_MEMO.get(memo_key)
        if cached is not None:
            return cached
    result = _normalize_padding_to_6tuple_impl(padding, K_T_for_causal, causal_pad_t)
    if memo_key is not None and len(_PADDING_MEMO) < 256:
        try:
            _PADDING_MEMO[memo_key] = result
        except TypeError:
            pass
    return result


def _normalize_padding_to_6tuple_impl(padding, K_T_for_causal=None, causal_pad_t=False):
    """Convert padding (int | 3-tuple of int | 3-tuple of pairs) to flat 6-tuple.

    Returns (pT_l, pT_r, pH_l, pH_r, pW_l, pW_r). Applies causal_pad_t
    override (pad_T = (K_T-1, 0)) when causal_pad_t=True.
    """
    if causal_pad_t and K_T_for_causal is None:
        raise ValueError("causal_pad_t=True requires K_T_for_causal")
    # First normalize to nested form ((pT_l,pT_r),(pH_l,pH_r),(pW_l,pW_r))
    if isinstance(padding, int):
        pn = ((padding, padding), (padding, padding), (padding, padding))
    elif isinstance(padding, (tuple, list)) and len(padding) == 3:
        pn = tuple(
            (p, p) if isinstance(p, int) else (int(p[0]), int(p[1]))
            for p in padding
        )
    elif isinstance(padding, (tuple, list)) and len(padding) == 6 and \
         all(isinstance(p, int) for p in padding):
        # Already in flat 6-tuple form -- accept directly
        pn = ((padding[0], padding[1]), (padding[2], padding[3]),
              (padding[4], padding[5]))
    else:
        raise ValueError(f"conv_nax: padding must be int, 3-tuple, "
                         f"3-tuple-of-pairs, or 6-tuple; got {padding}")
    if causal_pad_t:
        pn = ((K_T_for_causal - 1, 0), pn[1], pn[2])
    return (pn[0][0], pn[0][1], pn[1][0], pn[1][1], pn[2][0], pn[2][1])


def conv3d_nax_forward(
    x: mx.array,
    w: mx.array,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding=(0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    chunk_M: int = 0,
    *,
    causal_pad_t: bool = False,
) -> mx.array:
    """NAX-accelerated Conv3D forward, channels-last, multi-chunk.

    Sprint D thin wrapper: delegates dispatch to the C++ Primitive
    (`_ext.conv3d_nax_forward`). Python's job is input normalization
    (padding shape conversion + causal_pad_t flag handling) and friendly
    error messages; the heavy lifting (chunking, im2col, matmul2d) runs
    in C++.

    Equivalent to::

        mx.conv_general(x, w, stride=stride, padding=padding,
                        kernel_dilation=dilation)

    but routes through MPP matmul2d on M5+ Apple Silicon, achieving a
    median 1.64× speedup over `mx.conv_general` on SeedVR2 VAE
    production shapes (Sprint C Phase 1.5 ship-default verdict).

    Args:
        x: input array, shape `(B, T, H, W, C_in)`, dtype `float16` or
            `bfloat16`. Channels-last layout (matches `mx.conv_general`).
        w: weight array, shape `(C_out, K_T, K_H, K_W, C_in)`, same dtype.
        stride: `(sT, sH, sW)`, default `(1, 1, 1)`. Only stride=1 is
            currently supported across all dims.
        padding: int, 3-tuple of int (symmetric), 3-tuple of (left, right)
            pairs (asymmetric), or flat 6-tuple
            `(pT_l, pT_r, pH_l, pH_r, pW_l, pW_r)`.
        dilation: `(dT, dH, dW)`, default `(1, 1, 1)`. Only dilation=1
            is currently supported.
        chunk_M: 0 = auto from int32-byte-offset heuristic (recommended).
            Override only for benchmarking; the auto value respects the
            MPP matmul2d int32 internal addressing invariant.
        causal_pad_t: if True, override `pT` to `(K_T-1, 0)` for causal
            temporal padding (no future-frame leakage in video conv).
            `pH`, `pW` from `padding` are still honored.

    Returns:
        Output array, shape `(B, T_out, H_out, W_out, C_out)`, same dtype.

    Raises:
        ValueError: padding format invalid.
        RuntimeError: shape/dtype constraints violated (C++ side); see
            `_ext.conv3d_nax_forward` for the canonical list.

    Example:
        >>> import mlx.core as mx
        >>> from mlx_mfa.conv_nax import conv3d_nax_forward
        >>> x = mx.random.normal((1, 5, 64, 64, 512)).astype(mx.float16)
        >>> w = mx.random.normal((512, 3, 3, 3, 512)).astype(mx.float16)
        >>> y = conv3d_nax_forward(x, w, padding=(1, 1, 1))
        >>> y.shape
        (1, 5, 64, 64, 512)

    Diagnostic env vars:
        MFA_CONV_NAX_USE_PYTHON_LEGACY=1 — route through the Phase 1.x
            Python orchestrator instead of the C++ Primitive. Useful for
            bisecting any regression. Not for production use.
    """
    # Python-side validation: normalize padding to 6-tuple, apply causal flag.
    K_T = int(w.shape[1])
    flat_pad = _normalize_padding_to_6tuple(
        padding, K_T_for_causal=K_T, causal_pad_t=causal_pad_t)

    # Diagnostic escape hatch: route through Phase 1.x Python orchestrator.
    if os.environ.get("MFA_CONV_NAX_USE_PYTHON_LEGACY", "") == "1":
        # The legacy orchestrator accepts the same padding form, plus
        # causal_pad_t kwarg.
        return _conv3d_nax_forward_python_legacy(
            x, w, stride=tuple(stride), padding=padding,
            dilation=tuple(dilation), causal_pad_t=causal_pad_t)

    # Production path: C++ binding.
    from mlx_mfa import _ext
    return _ext.conv3d_nax_forward(
        x, w,
        stride=tuple(stride),
        padding=flat_pad,
        dilation=tuple(dilation),
        chunk_M=int(chunk_M),
    )


__all__ = [
    "conv3d_nax_forward",
    "get_chunk_plan",
    "estimate_working_set",
    "_conv3d_nax_forward_python_legacy",
]
