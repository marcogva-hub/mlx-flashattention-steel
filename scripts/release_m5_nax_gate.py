#!/usr/bin/env python3
"""Pre-tag M5/NAX validation gate (audit H-05 / CLAUDE_V6_NAX.md §AA.6).

MANDATORY before any release tag.  It:
  1. asserts the M5+ NAX fast path is genuinely LIVE (`has_nax()` True, not a
     silent SDPA fallback) — else the release's NAX perf/correctness claims are
     unverified;
  2. captures which-binary **byteΔ fingerprints** for the key NAX-tier dispatch
     cells (dense D128→NAX, dense D64→SDPA, sparse-sym→NAX, V6-split backward
     engagement) — proving the intended kernels actually run;
  3. ARCHIVES the fingerprints (MLX / hardware / macOS / date stamped) for the
     release record.
Exit non-zero — which BLOCKS the tag — if NAX is absent or any fingerprint check
fails.

Rationale (on record): the repo is PUBLIC, so a self-hosted M5 CI runner is a
security risk.  This required pre-release M5 run with archived fingerprints gives
NAX coverage without exposing the machine (audit H-05 option b).

Usage:
    .venv/bin/python scripts/release_m5_nax_gate.py
"""
from __future__ import annotations

import datetime
import json
import os
import sys

import mlx.core as mx

import mlx_mfa
from mlx_mfa import flash_attention, flash_attention_sparse, get_device_info, make_causal_block_mask


def _delta(a, b) -> float:
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _gen(B, H, N, D):
    f = lambda h: (mx.random.uniform(-1, 1, (B, h, N, D)) * 0.1).astype(mx.float16)
    q, k, v = f(H), f(H), f(H)
    mx.eval(q, k, v)
    return q, k, v


def main() -> int:
    print("=== Pre-tag M5/NAX validation gate (§AA.6) ===")

    # 1) NAX must be LIVE (not a silent SDPA fallback).
    ok, code = mlx_mfa.has_nax(reason=True)
    if not ok:
        print(f"❌ GATE FAILED: has_nax() is False (reason={code!r}). NAX is not live — "
              f"refusing to tag a release whose NAX claims are unverified. Run on an M5+ host "
              f"in the venv whose Python matches the built _ext.", file=sys.stderr)
        return 1
    print(f"✓ NAX live (has_nax reason={code})")

    mx.random.seed(0)
    fps: dict[str, float] = {}
    failures: list[tuple[str, float, str]] = []

    # 2a) dense D=128 auto → NAX matmul2d (a real kernel: 1e-7 < byteΔ < 3e-2 vs SDPA).
    q, k, v = _gen(1, 4, 2048, 128); sc = 1.0 / (128 ** 0.5)
    d = _delta(flash_attention(q, k, v, scale=sc, causal=False),
               mx.fast.scaled_dot_product_attention(q, k, v, scale=sc))
    fps["dense_D128_auto_nax"] = d
    if not (1e-7 < d < 3e-2):
        failures.append(("dense_D128_auto_nax", d, "expected NAX byteΔ in (1e-7, 3e-2)"))

    # 2b) dense D=64 auto → SDPA (byteΔ == 0: NAX loses at D=64, must stay SDPA).
    q, k, v = _gen(1, 4, 2048, 64); sc = 1.0 / (64 ** 0.5)
    d = _delta(flash_attention(q, k, v, scale=sc, causal=False),
               mx.fast.scaled_dot_product_attention(q, k, v, scale=sc))
    fps["dense_D64_auto_sdpa"] = d
    if d != 0.0:
        failures.append(("dense_D64_auto_sdpa", d, "expected SDPA byteΔ == 0"))

    # 2c) sparse symmetric D=128 → real NAX sparse (byteΔ > 0 vs masked SDPA).
    N, D = 2048, 128; sc = 1.0 / (D ** 0.5)
    q, k, v = _gen(1, 4, N, D)
    mask = make_causal_block_mask(N, head_dim=D)
    NQ, NK = mask.shape[-2], mask.shape[-1]
    em = mx.repeat(mx.repeat(mask.astype(mx.float32), N // NQ, -2), N // NK, -1)
    bias = mx.where(em > 0, mx.array(0.0), mx.array(-1e9, mx.float32)).astype(mx.float16)
    d = _delta(flash_attention_sparse(q, k, v, mask, scale=sc, causal=False),
               mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask=bias[None, None]))
    fps["sparse_D128_sym_nax"] = d
    if not (d > 1e-7):
        failures.append(("sparse_D128_sym_nax", d, "expected real NAX-sparse byteΔ > 0 vs masked SDPA"))

    # 2d) V6-split backward engagement (D=64 default-on): grad differs from SDPA-vjp.
    qd, kd, vd = _gen(1, 4, 2048, 64); scd = 1.0 / (64 ** 0.5)
    def loss(fn):
        def _l(q_, k_, v_):
            return mx.sum(fn(q_, k_, v_))
        g = mx.grad(_l, argnums=(0, 1, 2))(qd, kd, vd)
        mx.eval(*g)
        return g
    gk = loss(lambda q_, k_, v_: flash_attention(q_, k_, v_, scale=scd, causal=True))
    gs = loss(lambda q_, k_, v_: mx.fast.scaled_dot_product_attention(q_, k_, v_, scale=scd, mask="causal"))
    d = max(_delta(gk[i], gs[i]) for i in range(3))
    fps["v6_split_backward_D64_vs_sdpavjp"] = d
    if not (d > 1e-7):
        failures.append(("v6_split_backward_D64_vs_sdpavjp", d,
                         "expected V6-split backward to differ from SDPA-vjp (engaged)"))

    # ── Correctness fingerprints for the audit RC-A / RC-B causal defect class ──
    # (oracle-error, NOT byteΔ: a wrong-but-engaged kernel must still be caught).
    # Bottom-right causal additive mask for the N<S decode/cross convention:
    # query i (abs pos S-N+i) attends keys [0 .. S-N+i].
    def _br_causal_mask(N, S):
        qi = mx.arange(N).reshape(N, 1) + (S - N)
        ki = mx.arange(S).reshape(1, S)
        return mx.where(ki <= qi, mx.array(0.0), mx.array(-1e9, mx.float32))

    def _oracle_err(q16, k16, v16, sc, mask):
        out = flash_attention(q16, k16, v16, scale=sc, causal=True)
        ref = mx.fast.scaled_dot_product_attention(
            q16.astype(mx.float32), k16.astype(mx.float32), v16.astype(mx.float32),
            scale=sc, mask=mask[None, None])
        return _delta(out, ref)

    # 2e) DEFAULT-PATH CRITICAL: D=128 causal fp16 S=4096, N=4095 (qL_off%32=1).
    # Pre-fix err was ~3 (silent-wrong on backend="auto"); post-fix ~1e-3.
    Nc, Sc, Dc = 4095, 4096, 128; scc = 1.0 / (Dc ** 0.5)
    q, k, v = _gen(1, 4, Sc, Dc)  # gen S rows then slice q to N
    qN = q[:, :, :Nc, :]
    d = _oracle_err(qN, k, v, scc, _br_causal_mask(Nc, Sc))
    fps["rca_critical_D128_S4096_N4095_oracle"] = d
    if not (d < 1.5e-2):
        failures.append(("rca_critical_D128_S4096_N4095_oracle", d,
                         "RC-A default-path CRITICAL: expected oracle err < 1.5e-2 (was ~3 pre-fix)"))

    # 2f) RC-B decode-tail (odd-NK) must not NaN: D=128 causal fp16 N=1 S=257.
    Nt, St, Dt = 1, 257, 128; sct = 1.0 / (Dt ** 0.5)
    q, k, v = _gen(1, 4, St, Dt); qN = q[:, :, :Nt, :]
    d = _oracle_err(qN, k, v, sct, _br_causal_mask(Nt, St))
    fps["rcb_decode_tail_D128_N1_S257_oracle"] = d
    if not (d < 1.5e-2):  # NaN -> _delta returns nan -> comparison False -> fails (caught)
        failures.append(("rcb_decode_tail_D128_N1_S257_oracle", d,
                         "RC-B decode-tail: expected finite oracle err < 1.5e-2 (was NaN pre-fix)"))

    # 3) Archive the stamped fingerprints (audit record; off the tracked tree).
    dev = get_device_info()
    stamp = {
        "release_version": mlx_mfa.__version__,
        "mlx_version": mx.__version__,
        "device": dev.get("device_name", "?"),
        "chip": dev.get("chip_name", "?"),
        "is_m5_plus": bool(dev.get("is_m5_plus", False)),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "has_nax": ok,
        "nax_reason": code,
        "fingerprints": fps,
        "gate": "PASS" if not failures else "FAIL",
    }
    archive_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "devnotes", "release-fingerprints")
    os.makedirs(archive_dir, exist_ok=True)
    path = os.path.join(archive_dir, f"m5-nax-fingerprints-{mlx_mfa.__version__}.json")
    with open(path, "w") as fh:
        json.dump(stamp, fh, indent=2)
    print(f"✓ archived fingerprints → {os.path.relpath(path)}")
    for nm, val in fps.items():
        print(f"    {nm}: byteΔ={val:.3e}")

    if failures:
        print("\n❌ GATE FAILED — fingerprint mismatches:", file=sys.stderr)
        for nm, val, exp in failures:
            print(f"    {nm}: byteΔ={val:.3e} — {exp}", file=sys.stderr)
        return 1
    print("\n✓ M5/NAX GATE PASSED — NAX live, all fingerprints as expected, archived.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
