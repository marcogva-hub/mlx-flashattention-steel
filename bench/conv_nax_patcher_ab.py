#!/usr/bin/env python3
"""Sprint D Track C.2 — patch_seedvr2_vae A/B sanity bench.

Loads a mock 3-Conv3d VAE block (production-shape-ish), times un-patched
vs patched forward. Sanity check: patched must show ≥1.2× speedup on at
least one of the test layers' representative shapes.

Single-session (sanity, not a shipping-grade sweep). Phase 1.5 numbers
remain the canonical perf record. Smoke gate per Phase 1.1 lesson:
correctness check before timing.
"""
import time, statistics
import mlx.core as mx
import mlx.nn as nn
from mlx_mfa.integrations.seedvr2_vae import patch_seedvr2_vae


class TestVAEBlock(nn.Module):
    """Mock VAE block with production-shape Conv3d layers.

    Two 3×3×3 layers + one 1×1×1 layer (eligible for the fast path).
    Channel counts match a SeedVR2 VAE mid-resnet-like profile.
    """
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv_b = nn.Conv3d(512, 512, kernel_size=1, padding=0)
        self.conv_c = nn.Conv3d(512, 512, kernel_size=3, padding=1)

    def __call__(self, x):
        x = self.conv_a(x)
        x = self.conv_b(x)
        x = self.conv_c(x)
        return x


def cast_f16(model):
    for _, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            m.weight = m.weight.astype(mx.float16)
            if m.bias is not None:
                m.bias = m.bias.astype(mx.float16)


def bench_forward(model, x, n_runs=5):
    # Warmup
    for _ in range(3):
        y = model(x); mx.async_eval(y); mx.synchronize()
    times = []
    for _ in range(n_runs):
        mx.synchronize()
        t0 = time.perf_counter()
        y = model(x); mx.async_eval(y); mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    # mid_resnet-ish input shape (channels-last)
    B, T, H, W, C = 1, 5, 64, 64, 512
    mx.random.seed(0)
    x = (mx.random.uniform(shape=(B, T, H, W, C)) * 0.1).astype(mx.float16)
    mx.async_eval(x); mx.synchronize()

    # Build TWO model instances — one for unpatched, one for patched
    m_unpatched = TestVAEBlock()
    cast_f16(m_unpatched)
    m_patched = TestVAEBlock()
    cast_f16(m_patched)
    # Copy weights from unpatched to patched so they're identical
    for (n1, m1), (n2, m2) in zip(
        m_unpatched.named_modules(), m_patched.named_modules()):
        if isinstance(m1, nn.Conv3d):
            m2.weight = m1.weight
            m2.bias = m1.bias

    # Smoke gate: outputs match within FP16 noise
    patch_seedvr2_vae(m_patched, verbose=False)
    y_u = m_unpatched(x); mx.async_eval(y_u); mx.synchronize()
    y_p = m_patched(x); mx.async_eval(y_p); mx.synchronize()
    err = mx.abs(y_u.astype(mx.float32) - y_p.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(err*err)))
    mag = float(mx.max(mx.abs(y_u.astype(mx.float32))))
    rel = rmse / mag if mag > 0 else 0
    print(f"[smoke] rel_err patched vs unpatched: {rel:.4e}")
    assert rel < 1e-3, f"smoke failed: rel={rel}"

    t_unp = bench_forward(m_unpatched, x, n_runs=7)
    t_pat = bench_forward(m_patched,  x, n_runs=7)
    speedup = t_unp / t_pat if t_pat > 0 else 0
    print(f"[A/B] unpatched: {t_unp:>7.2f} ms  patched: {t_pat:>7.2f} ms  speedup: {speedup:.2f}×")
    assert speedup >= 1.2, (
        f"patcher A/B sanity fail: speedup={speedup:.2f}× < 1.2× bar"
    )
    import json
    out = {
        "shape": {"B": B, "T": T, "H": H, "W": W, "C": C, "block": "3x Conv3d (3x3x3, 1x1x1, 3x3x3)"},
        "unpatched_ms": t_unp,
        "patched_ms": t_pat,
        "speedup": speedup,
        "smoke_rel_err": rel,
        "n_runs": 7,
        "single_session": True,
    }
    with open("docs/conv-nax/conv-nax-prod-patcher-ab.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"data -> docs/conv-nax/conv-nax-prod-patcher-ab.json")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()
