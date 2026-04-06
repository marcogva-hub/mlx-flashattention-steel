"""Audit mlx-mfa dispatch for DiT/UNet/VSR shapes.

Tests self-attention and cross-attention shapes used by 11 MLX VSR models
and reports dispatch correctness (MFA vs SDPA routing) against wall-clock
measurements.

Usage:
    .venv/bin/python scripts/audit_dit_dispatch.py
"""

import json
import os
import sys
import time

import mlx.core as mx

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mlx_mfa import flash_attention  # noqa: E402
from mlx_mfa.dispatch_policy import should_use_mfa  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_WARMUP = 3
N_ITERS = 10

# === Self-attention shapes (non-causal) ===
SELF_ATTN_SHAPES = [
    # (B, H, N, D, label)
    # DiT models (D=128)
    (1, 20, 26730, 128, "SeedVR2 DiT bs=29 (27x33x30)"),
    (1, 20, 111375, 128, "SeedVR2 DiT bs=497 (27x33x125)"),
    (1, 30, 70200, 128, "CogVideoX (DOVE/STAR/SparkVSR/Vivid-VR)"),
    (1, 40, 100000, 128, "Wan2.1 (FlashVSR) approx"),
    # UNet models (D=64 and D=128)
    (1, 8, 4096, 64, "UNet SD low-res (64x64) D=64"),
    (1, 8, 4096, 128, "UNet SD low-res (64x64) D=128"),
    (1, 8, 16384, 64, "UNet SD mid-res (128x128) D=64"),
    (1, 8, 1024, 64, "UNet SD high-res (32x32) D=64"),
    (2, 8, 4096, 64, "UNet SD B=2 (CFG) D=64"),
    # Small shapes (should they go to SDPA?)
    (1, 8, 256, 128, "UNet tiny (16x16)"),
    (1, 8, 64, 128, "UNet smallest (8x8)"),
]

# === Cross-attention shapes (non-causal, N_q >> N_kv) ===
CROSS_ATTN_SHAPES = [
    # (B, H_q, N_q, H_kv, N_kv, D, label)
    (1, 30, 70200, 30, 226, 128, "CogVideoX text cross-attn (226 text tokens)"),
    (1, 40, 100000, 40, 512, 128, "Wan2.1 text cross-attn"),
    (1, 8, 4096, 8, 77, 64, "SD/SDXL text cross-attn D=64"),
    (1, 8, 4096, 8, 77, 128, "SD/SDXL text cross-attn D=128"),
    (1, 30, 70200, 30, 77, 128, "DiT x CLIP-77 extreme ratio"),
    # LTX-2 audio-video cross-attention
    (1, 32, 14000, 32, 2000, 64, "LTX-2 video->audio cross-attn"),
    (1, 32, 2000, 32, 14000, 64, "LTX-2 audio->video cross-attn"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median(vals):
    s = sorted(vals)
    return s[len(s) // 2]


def _bench_fn(fn, n_warmup, n_iters):
    """Warmup + timed iterations. Returns median ms.

    mx.eval() triggers GPU dispatch; mx.synchronize() waits for completion.
    Without mx.eval(), MLX lazy evaluation means we only time graph building.
    """
    for _ in range(n_warmup):
        o = fn()
        mx.eval(o)
        mx.synchronize()

    times = []
    for _ in range(n_iters):
        mx.synchronize()
        t0 = time.perf_counter()
        o = fn()
        mx.eval(o)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return _median(times) * 1000


def _get_is_m3_plus():
    """Detect M3+ from the extension if available."""
    try:
        from mlx_mfa._ext import _get_device_info
        info = _get_device_info()
        return info.get("is_m3_plus", False)
    except (ImportError, AttributeError):
        return False


IS_M3_PLUS = _get_is_m3_plus()


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_self_attn(B, H, N, D, label):
    q = mx.random.normal((B, H, N, D), dtype=mx.float16)
    k = mx.random.normal((B, H, N, D), dtype=mx.float16)
    v = mx.random.normal((B, H, N, D), dtype=mx.float16)
    mx.eval(q, k, v)
    mx.synchronize()

    # Dispatch decision
    decision = should_use_mfa(D, N, causal=False, is_m3_plus=IS_M3_PLUS,
                              dtype=mx.float16)

    # Benchmark MFA
    try:
        med_mfa = _bench_fn(
            lambda: flash_attention(q, k, v, backend="mfa"),
            N_WARMUP, N_ITERS,
        )
    except Exception as e:
        med_mfa = float("inf")
        print(f"  MFA ERROR: {e}")

    # Benchmark SDPA
    med_sdpa = _bench_fn(
        lambda: flash_attention(q, k, v, backend="sdpa"),
        N_WARMUP, N_ITERS,
    )

    ratio = med_sdpa / med_mfa if med_mfa > 0 and med_mfa != float("inf") else 0
    winner = "MFA" if med_mfa < med_sdpa else "SDPA"
    correct = (decision and winner == "MFA") or (not decision and winner == "SDPA")

    mark = "\u2705" if correct else "\u274c"
    print(f"[{mark}] {label}")
    print(f"    dispatch={'MFA' if decision else 'SDPA'} | "
          f"MFA={med_mfa:.1f}ms | SDPA={med_sdpa:.1f}ms | "
          f"ratio={ratio:.2f}x | winner={winner}")
    return {
        "label": label, "B": B, "H": H, "N": N, "D": D,
        "dispatch_mfa": decision, "mfa_ms": round(med_mfa, 2),
        "sdpa_ms": round(med_sdpa, 2), "ratio": round(ratio, 3),
        "winner": winner, "correct": correct,
    }


def bench_cross_attn(B, H_q, N_q, H_kv, N_kv, D, label):
    q = mx.random.normal((B, H_q, N_q, D), dtype=mx.float16)
    k = mx.random.normal((B, H_kv, N_kv, D), dtype=mx.float16)
    v = mx.random.normal((B, H_kv, N_kv, D), dtype=mx.float16)
    mx.synchronize()

    # MFA
    try:
        med_mfa = _bench_fn(
            lambda: flash_attention(q, k, v, backend="mfa"),
            N_WARMUP, N_ITERS,
        )
    except Exception as e:
        med_mfa = float("inf")
        print(f"  MFA ERROR: {e}")

    # SDPA
    med_sdpa = _bench_fn(
        lambda: flash_attention(q, k, v, backend="sdpa"),
        N_WARMUP, N_ITERS,
    )

    ratio = med_sdpa / med_mfa if med_mfa > 0 and med_mfa != float("inf") else 0
    winner = "MFA" if med_mfa < med_sdpa else "SDPA"

    mark = "\u2705" if winner == "MFA" else "\u26a0\ufe0f"
    print(f"[{mark}] {label}")
    print(f"    Nq={N_q} Nkv={N_kv} ratio={N_q / N_kv:.0f}:1 | "
          f"MFA={med_mfa:.1f}ms | SDPA={med_sdpa:.1f}ms | "
          f"speedup={ratio:.2f}x | winner={winner}")
    return {
        "label": label, "B": B, "H_q": H_q, "N_q": N_q,
        "H_kv": H_kv, "N_kv": N_kv, "D": D,
        "mfa_ms": round(med_mfa, 2), "sdpa_ms": round(med_sdpa, 2),
        "ratio": round(ratio, 3), "winner": winner,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print(f"DiT/UNet dispatch audit -- M{'3+' if IS_M3_PLUS else '1/M2'}")
    print("=" * 70)

    print("\n--- Self-attention (non-causal) ---\n")
    self_results = []
    for B, H, N, D, label in SELF_ATTN_SHAPES:
        try:
            r = bench_self_attn(B, H, N, D, label)
            self_results.append(r)
        except Exception as e:
            print(f"[!!] {label}: {e}")

    print("\n--- Cross-attention (non-causal, N_q >> N_kv) ---\n")
    cross_results = []
    for B, Hq, Nq, Hkv, Nkv, D, label in CROSS_ATTN_SHAPES:
        try:
            r = bench_cross_attn(B, Hq, Nq, Hkv, Nkv, D, label)
            cross_results.append(r)
        except Exception as e:
            print(f"[!!] {label}: {e}")

    # Save results
    all_results = {
        "hardware": f"M{'3+' if IS_M3_PLUS else '1/M2'}",
        "self_attn": self_results,
        "cross_attn": cross_results,
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "docs", "audit_dit_dispatch_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    misrouted = [r for r in self_results if not r.get("correct", True)]
    if misrouted:
        print(f"\n!! {len(misrouted)} misrouted self-attention shapes:")
        for r in misrouted:
            d = "MFA" if r["dispatch_mfa"] else "SDPA"
            print(f"   {r['label']}: dispatch={d} but {r['winner']} is faster")

    sdpa_wins_cross = [r for r in cross_results if r["winner"] == "SDPA"]
    if sdpa_wins_cross:
        print(f"\n!! {len(sdpa_wins_cross)} cross-attn shapes where SDPA wins:")
        for r in sdpa_wins_cross:
            print(f"   {r['label']}: SDPA={r['sdpa_ms']:.1f}ms vs MFA={r['mfa_ms']:.1f}ms")

    # Count correct/total
    total_self = len(self_results)
    correct_self = sum(1 for r in self_results if r.get("correct", True))
    print(f"\nSelf-attention dispatch accuracy: {correct_self}/{total_self}")
