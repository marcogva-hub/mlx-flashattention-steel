"""Sprint D Track D — C++ Primitive vs Python orchestrator equivalence.

Per prompt §6.1: validate the C++ binding produces output bit-exact or
FP-noise equivalent to the Phase 1.x Python orchestrator on all 6
production shapes. The Python orchestrator is preserved as
`_conv3d_nax_forward_python_legacy` for this purpose.

The migration is the substantive Sprint D deliverable; this test gates it.
"""
import os
import pytest
import mlx.core as mx
from mlx_mfa.conv_nax import (
    conv3d_nax_forward,
    _conv3d_nax_forward_python_legacy,
)


# 6 production shapes from Sprint C Phase 1.5 (matching the harness).
# III-7 Class A/C: the original 6 are all C_in % 32 == 0 (the only regime
# the matmul2d path was correct in pre-III-6). Added small + non-%32 C_in
# shapes so the migration-equivalence test also exercises the
# previously-broken regime (now fixed by the III-6 K-tail pad).
SHAPES = [
    # (label, B, T, H, W, C_in, C_out, K_T, K_H, K_W)
    ("mid_resnet",             1,  5, 64,  64,  512, 512, 3, 3, 3),
    ("up1_resnet",             1,  9, 128, 128, 512, 512, 3, 3, 3),
    ("up2_resnet0_chunk_cap",  1, 11, 150, 180, 512, 256, 3, 3, 3),
    ("up3_resnet_chunk_cap",   1, 24, 128, 193, 128, 128, 3, 3, 3),
    ("up2_resnet_full",        1, 17, 256, 256, 256, 256, 3, 3, 3),
    ("up2_resnet0_peakflops",  1, 17, 256, 256, 512, 256, 3, 3, 3),
    # III-7: previously-broken non-%32 / small C_in regime (K-tail fix).
    ("small_cin16",            1,  4,  8,   8,   16,  64, 3, 3, 3),
    ("nonmult_cin48",          1,  4,  8,   8,   48,  64, 3, 3, 3),
    ("nonmult_cin31",          1,  4,  8,   8,   31,  64, 3, 3, 3),
]


@pytest.mark.parametrize("spec", SHAPES, ids=[s[0] for s in SHAPES])
def test_cpp_vs_python_equivalence(spec):
    """C++ binding output == Python orchestrator output (bit-exact or FP-noise).

    Both paths execute the same Metal kernels with the same dispatch
    parameters; only the orchestration runs in different languages.
    Expected: bit-exact (rmse=0).
    """
    label, B, T, H, W, C_in, C_out, K_T, K_H, K_W = spec
    pad = (K_T // 2, K_H // 2, K_W // 2)

    mx.random.seed(0)
    x = (mx.random.uniform(shape=(B, T, H, W, C_in)) * 0.1).astype(mx.float16)
    w = (mx.random.uniform(shape=(C_out, K_T, K_H, K_W, C_in)) * 0.1).astype(mx.float16)
    mx.async_eval(x, w); mx.synchronize()

    # C++ path (default after Sprint D)
    os.environ.pop("MFA_CONV_NAX_USE_PYTHON_LEGACY", None)
    y_cpp = conv3d_nax_forward(x, w, stride=(1,1,1),
                                padding=pad, dilation=(1,1,1))
    mx.async_eval(y_cpp); mx.synchronize()

    # Python legacy path (explicit)
    y_py = _conv3d_nax_forward_python_legacy(
        x, w, stride=(1,1,1), padding=pad, dilation=(1,1,1))
    mx.async_eval(y_py); mx.synchronize()

    assert y_cpp.shape == y_py.shape
    err = mx.abs(y_cpp.astype(mx.float32) - y_py.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(err * err)))
    maxe = float(mx.max(err))
    mag = float(mx.max(mx.abs(y_py.astype(mx.float32))))
    rel = rmse / mag if mag > 0 else 0.0
    # Expected: bit-exact. Allow tiny FP-noise tolerance for
    # potentially different dispatch ordering in chunk concat.
    assert rel < 1e-5, (
        f"{label}: cpp vs python_legacy diverged: "
        f"rmse={rmse:.6e} max={maxe:.4e} rel={rel:.4e} (bar 1e-5)"
    )

    # III-7 lesson #11: cpp and python_legacy SHARE the matmul2d kernel, so
    # the equivalence check above is blind to a bug both share (the exact
    # conv3d trap). Anchor BOTH against an INDEPENDENT fp32 reference.
    # fp32 mx.conv_general can never be hooked to NAX (NAX is fp16/bf16
    # only), so it is independent regardless of installed-hook state.
    y_f32 = mx.conv_general(x.astype(mx.float32), w.astype(mx.float32),
                            stride=(1, 1, 1), padding=list(pad),
                            kernel_dilation=(1, 1, 1))
    mx.eval(y_f32)
    ref_rms = float(mx.sqrt(mx.mean(y_f32 * y_f32)))
    for name, y in (("cpp", y_cpp), ("python_legacy", y_py)):
        assert not bool(mx.any(mx.isnan(y)).item()), f"{label}: NaN in {name}"
        mae = float(mx.mean(mx.abs(y.astype(mx.float32) - y_f32))) / max(ref_rms, 1e-6)
        assert mae < 0.01, (
            f"{label}: {name} diverges from fp32 reference: MAE/RMS {mae:.5f} "
            "(both paths could share a regression the equivalence check misses)")
