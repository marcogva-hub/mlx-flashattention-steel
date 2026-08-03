#!/usr/bin/env python
"""mfa-side microbench: full spatial-pad rescue path (pad->conv3d_nax_forward->slice, WITH
pad/slice overhead) vs native mx.conv_general, for the SeedVR2 VAE spatial-tail families.
Fresh process per order (argv[1] = A|B), warmup+reps median, correctness vs native.
This is the number that enters CHANGELOG 2.62.1 for family #2 (54x66)."""
import sys, os, json, statistics, time
os.environ["MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE"] = "1"
import mlx.core as mx
import numpy as np
import mlx_mfa  # installs hooks at import
from mlx_mfa._auto_hooks import _ORIGINAL_CONV_GENERAL, _try_conv3d_spatial_pad_and_slice

order = sys.argv[1] if len(sys.argv) > 1 else "A"
WARM, REPS = 10, 60
FAMILIES = {
    "family2_54x66_inT3": (1, 3, 54, 66, 512),     # the one being added (dominant, census)
    "family2_54x66_inT4": (1, 4, 54, 66, 512),     # boundary key
    "family1_108x132_inT4": (1, 4, 108, 132, 512), # reference (already shipped)
}
mx.random.seed(0)

def bench(fn):
    for _ in range(WARM): mx.eval(fn())
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); mx.eval(fn()); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)

out = {}
for name, ish in FAMILIES.items():
    x = mx.random.normal(ish).astype(mx.float16)
    w = mx.random.normal((512, 3, 3, 3, ish[4])).astype(mx.float16)
    p3, p6 = (0, 1, 1), (0, 0, 1, 1, 1, 1)
    native = lambda: _ORIGINAL_CONV_GENERAL(x, w, stride=(1, 1, 1), padding=p3,
                                            kernel_dilation=1, input_dilation=1, groups=1, flip=False)
    spad = lambda: _try_conv3d_spatial_pad_and_slice(x, w, p6)
    on = native(); os_ = spad(); mx.eval(on)
    engaged = os_ is not None
    if engaged:
        mx.eval(os_)
        a = np.array(on.astype(mx.float32)).ravel(); b = np.array(os_.astype(mx.float32)).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        maxabs = float(np.abs(a - b).max())
        if order == "A": nm, sm = bench(native), bench(spad)
        else: sm, nm = bench(spad), bench(native)
        ratio = nm / sm
    else:
        cos = maxabs = ratio = None; nm = bench(native); sm = None
    out[name] = {"in_shape": list(ish), "engaged": engaged, "native_ms": round(nm, 4),
                 "spatial_pad_ms": round(sm, 4) if sm else None,
                 "ratio_native_over_spatialpad": round(ratio, 3) if ratio else None,
                 "cos_vs_native": cos, "max_abs_vs_native": maxabs}
    print(f"{name} [{order}]: engaged={engaged} native={nm:.4f}ms spatial={sm and round(sm,4)}ms "
          f"ratio={ratio and round(ratio,3)}x cos={cos and round(cos,7)} maxabs={maxabs}")
meta = {"order": order, "mlx": mx.__version__, "mlx_mfa": mlx_mfa.__version__,
        "warm": WARM, "reps": REPS, "families": out}
json.dump(meta, open(f"benchmarks/results/conv3d_spatial_pad_family2_order{order}.json", "w"), indent=1)
