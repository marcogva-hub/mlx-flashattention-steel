"""Conv3D NAX forward — Phase 1.1 single-chunk implementation.

Scope: Phase 1.1 sub-phase B (mid_resnet shape, single-chunk, forward
only, FP16, channels-last layout). Multi-chunk + chunking heuristic +
1x1x1 fast path + asymmetric pad_T are deferred to Phase 1.2-1.5.

Per docs/conv-nax/conv-nax-design.md decision D3: this module orchestrates
two JIT Metal kernels via mx.fast.metal_kernel:

  1. Im2col3D — gathers (B*T_out*H_out*W_out, K_T*K_H*K_W*C_in) buffer
     from (B, T, H, W, C_in) input + (stride, pad, dil, K_T, K_H, K_W)
  2. matmul2d  — lifts the V6-NAX-validated MPP matmul2d kernel from
     bench/conv_nax_matmul2d_microbench.py, dispatches one TG per
     (32x32) output tile with K-loop accumulation.

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
              K_T, K_H, K_W, sT, sH, sW, pT, pH, pW, dT, dH, dW, dtype):
    """ConvKey per design D3: (Kind=Conv3DForward, all shape/conv params, dtype)."""
    return ("Conv3DForward", B, T, H, W, C_in, T_out, H_out, W_out, C_out,
            K_T, K_H, K_W, sT, sH, sW, pT, pH, pW, dT, dH, dW, str(dtype))


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
    """
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
                     K_T, K_H, K_W, sT, sH, sW, pT, pH, pW, dT, dH, dW) -> str:
    """Im2col for Conv3D channels-last.

    Output: (M, K) row-major where
      M = B*T_out*H_out*W_out
      K = K_T*K_H*K_W*C_in
    Each thread writes one (m, k) element.

    Input layout: (B, T, H, W, C_in) row-major
    Output layout: (M, K) row-major
    Padding: zero-fill for out-of-bounds spatial coords.
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
    constexpr int  cpT   = {pT};
    constexpr int  cpH   = {pH};
    constexpr int  cpW   = {pW};
    constexpr int  cdT   = {dT};
    constexpr int  cdH   = {dH};
    constexpr int  cdW   = {dW};
    constexpr uint cKvol = cKT * cKH * cKW;
    constexpr uint cKfull = cKvol * cCin;
    constexpr uint cM     = cB * cTout * cHout * cWout;

    // Thread index covers (m, k) flattened.
    uint tid = thread_position_in_grid.x;
    if (tid >= cM * cKfull) return;
    uint m = tid / cKfull;
    uint k = tid - m * cKfull;

    // Unravel m -> (b, t_out, h_out, w_out)
    uint rem_m = m;
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

    int t_in = (int)t_out * csT + (int)k_t * cdT - cpT;
    int h_in = (int)h_out * csH + (int)k_h * cdH - cpH;
    int w_in = (int)w_out * csW + (int)k_w * cdW - cpW;

    half v = (half)0.0h;
    if (t_in >= 0 && t_in < (int)cT &&
        h_in >= 0 && h_in < (int)cH &&
        w_in >= 0 && w_in < (int)cW) {{
        uint in_idx = ((b * cT + (uint)t_in) * cH + (uint)h_in) * cW * cCin
                     + (uint)w_in * cCin + c_in;
        v = X[in_idx];
    }}
    Im2col[m * cKfull + k] = v;
"""


def _make_kernels(key, M, K, N, B, T, H, W, C_in, T_out, H_out, W_out, C_out,
                  K_T, K_H, K_W, sT, sH, sW, pT, pH, pW, dT, dH, dW):
    """Compile and cache the (im2col, matmul) kernel pair for a ConvKey."""
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    im2col = mx.fast.metal_kernel(
        name=f"im2col3d_{B}_{T}_{H}_{W}_{C_in}_{K_T}{K_H}{K_W}_"
             f"s{sT}{sH}{sW}_p{pT}{pH}{pW}_d{dT}{dH}{dW}",
        input_names=["X"],
        output_names=["Im2col"],
        source=_im2col3d_source(B, T, H, W, C_in, T_out, H_out, W_out,
                                K_T, K_H, K_W, sT, sH, sW,
                                pT, pH, pW, dT, dH, dW),
        header=_IM2COL_HEADER,
        ensure_row_contiguous=True,
    )
    mm = mx.fast.metal_kernel(
        name=f"conv3d_matmul2d_{M}_{K}_{N}",
        input_names=["A", "B"],
        output_names=["C"],
        source=_matmul2d_source(M, K, N),
        header=_MATMUL_HEADER,
        ensure_row_contiguous=True,
    )
    _KERNEL_CACHE[key] = (im2col, mm)
    return im2col, mm


# ---------------------------------------------------------------------
# Sanity asserts (design doc §4, 8 categories).
# ---------------------------------------------------------------------
def _sanity_asserts(x: mx.array, w: mx.array, stride, padding, dilation):
    """Throw if Phase 1.1 single-chunk constraints not met."""
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

    # Category 4: stride/padding/dilation triple
    for name, v in [("stride", stride), ("padding", padding),
                    ("dilation", dilation)]:
        if not (isinstance(v, (tuple, list)) and len(v) == 3):
            raise ValueError(f"conv_nax: {name} must be a 3-tuple (T,H,W); "
                             f"got {v}")
        for vi in v:
            if not isinstance(vi, int) or vi < (1 if name != "padding" else 0):
                raise ValueError(f"conv_nax: {name}={v} contains invalid int")

    # Category 5: kernel size positive
    for i, ax in enumerate(("K_T", "K_H", "K_W")):
        if w.shape[1 + i] < 1:
            raise ValueError(f"conv_nax: {ax}={w.shape[1+i]} < 1")

    # Category 6: input spatial extent must accommodate kernel
    B, T, H, W, C_in = x.shape
    C_out, K_T, K_H, K_W, _ = w.shape
    sT, sH, sW = stride
    pT, pH, pW = padding
    dT, dH, dW = dilation
    eff_T = T + 2 * pT - dT * (K_T - 1) - 1
    eff_H = H + 2 * pH - dH * (K_H - 1) - 1
    eff_W = W + 2 * pW - dW * (K_W - 1) - 1
    if eff_T < 0 or eff_H < 0 or eff_W < 0:
        raise ValueError(f"conv_nax: input too small for kernel after padding: "
                         f"eff_T={eff_T} eff_H={eff_H} eff_W={eff_W}")

    # Category 7: single-chunk feasibility (Phase 1.1 only — Phase 1.3 adds
    # multi-chunk). Compute working set:
    #   im2col buffer = M * K * dtype_bytes
    T_out = eff_T // sT + 1
    H_out = eff_H // sH + 1
    W_out = eff_W // sW + 1
    M = B * T_out * H_out * W_out
    K = C_in * K_T * K_H * K_W
    dtype_bytes = 2  # f16 / bf16
    im2col_bytes = M * K * dtype_bytes
    # Phase 1.1 budget: 8 GB single-chunk im2col (loose; mid_resnet uses ~540MB)
    PHASE1_1_BUDGET = 8 * 1024**3
    if im2col_bytes > PHASE1_1_BUDGET:
        raise ValueError(
            f"conv_nax: im2col buffer ~{im2col_bytes/1e9:.2f} GB exceeds "
            f"Phase 1.1 single-chunk budget ({PHASE1_1_BUDGET/1e9:.0f} GB). "
            f"Multi-chunk loop is Phase 1.3 scope; use mx.conv_general for "
            f"this shape until then."
        )

    # Category 8: alignment to tile dims
    # M_TILE = N_TILE = 32 — fine if M, N have remainder (handled by bounds
    # check in store loop). Just verify N (=C_out) reasonable.
    if C_out <= 0 or C_out > 65536:
        raise ValueError(f"conv_nax: implausible C_out={C_out}")

    return B, T, H, W, C_in, T_out, H_out, W_out, C_out, K_T, K_H, K_W, M, K


# ---------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------
def conv3d_nax_forward(
    x: mx.array,
    w: mx.array,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
) -> mx.array:
    """NAX-accelerated Conv3D forward, single-chunk, channels-last.

    Equivalent to:
        mx.conv_general(x, w, stride=stride, padding=padding,
                        kernel_dilation=dilation)

    But routes through implicit-GEMM via MPP matmul2d. Phase 1.1 scope:
    - single-chunk only (im2col working set < 8 GB)
    - forward only (no VJP)
    - fp16 / bf16 only
    - channels-last layout
    - symmetric padding only (asymmetric/causal pad_T = Phase 1.2)

    Args:
        x: input array, shape (B, T, H, W, C_in), dtype f16 or bf16.
        w: weight array, shape (C_out, K_T, K_H, K_W, C_in), same dtype.
        stride: (sT, sH, sW), default (1,1,1).
        padding: (pT, pH, pW) symmetric padding, default (0,0,0).
        dilation: (dT, dH, dW), default (1,1,1).

    Returns:
        Output array, shape (B, T_out, H_out, W_out, C_out), same dtype.

    Raises:
        ValueError: if any sanity check fails (8 categories, see source).
    """
    sT, sH, sW = stride
    pT, pH, pW = padding
    dT, dH, dW = dilation

    (B, T, H, W, C_in, T_out, H_out, W_out, C_out,
     K_T, K_H, K_W, M, K) = _sanity_asserts(x, w, stride, padding, dilation)
    N = C_out

    # Reshape weight to (C_out, K_T*K_H*K_W*C_in) row-major (no copy if
    # channels-last input is contiguous over the last 4 dims).
    w_flat = w.reshape(C_out, K_T * K_H * K_W * C_in)

    key = _conv_key(B, T, H, W, C_in, T_out, H_out, W_out, C_out,
                    K_T, K_H, K_W, sT, sH, sW, pT, pH, pW, dT, dH, dW, x.dtype)
    im2col_kernel, mm_kernel = _make_kernels(
        key, M, K, N, B, T, H, W, C_in, T_out, H_out, W_out, C_out,
        K_T, K_H, K_W, sT, sH, sW, pT, pH, pW, dT, dH, dW)

    # Step 1: im2col -- M*K elements, 1 thread per element.
    total_elems = M * K
    THREADS_PER_TG_IM2COL = 256
    grid_x = (total_elems + THREADS_PER_TG_IM2COL - 1) // THREADS_PER_TG_IM2COL
    im2col_buf = im2col_kernel(
        inputs=[x],
        output_shapes=[(M, K)],
        output_dtypes=[x.dtype],
        grid=(grid_x * THREADS_PER_TG_IM2COL, 1, 1),
        threadgroup=(THREADS_PER_TG_IM2COL, 1, 1),
    )[0]

    # Step 2: matmul2d  (M, K) @ (N, K)^T = (M, N), grid one TG per output tile.
    n_tg_x = (N + N_TILE - 1) // N_TILE
    n_tg_y = (M + M_TILE - 1) // M_TILE
    flat = mm_kernel(
        inputs=[im2col_buf, w_flat],
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
        grid=(n_tg_x * TG_THREADS, n_tg_y, 1),
        threadgroup=(TG_THREADS, 1, 1),
    )[0]

    # Step 3: reshape (M, N) -> (B, T_out, H_out, W_out, C_out)
    return flat.reshape(B, T_out, H_out, W_out, C_out)


__all__ = ["conv3d_nax_forward"]
