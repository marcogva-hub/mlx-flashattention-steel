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
import hashlib
import json
import os
import subprocess
import sys

import mlx.core as mx

import mlx_mfa
from mlx_mfa import flash_attention, flash_attention_sparse, get_device_info, make_causal_block_mask

# Engagement-fingerprint shape — SINGLE SOURCE for the gate (2c below) AND the
# anti-recurrence lock (tests/test_release_gate_engage_shape_lock.py).  The CENTER
# of a proven non-causal NAX-sparse winning region (07-13 re-map): D=128, N=8192,
# B·H=12, density~0.15 (density is the mid of the [0, 0.30] non-causal ceiling; N is
# in the measured [4096, 8192]).  The lock asserts this shape satisfies
# _nax_sparse_route_viable, so any future routing-narrowing fails the lock BEFORE a
# Phase-3 gate run (with a message saying to re-choose inside the measured envelope).
SPARSE_ENGAGE_SPEC = {"B": 1, "H": 12, "N": 8192, "D": 128, "BT": 32,
                      "density": 0.15, "causal": False}

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_sha() -> str:
    """HEAD commit the gate is validating (volet D — the fingerprint's authenticity
    anchor; the publish precondition checks no csrc//mlx_mfa/ source changed since)."""
    try:
        return subprocess.run(
            ["git", "-C", _REPO, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def _delta(a, b) -> float:
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _gen(B, H, N, D):
    f = lambda h: (mx.random.uniform(-1, 1, (B, h, N, D)) * 0.1).astype(mx.float16)
    q, k, v = f(H), f(H), f(H)
    mx.eval(q, k, v)
    return q, k, v


def _masked_sdpa(q, k, v, block_mask, scale):
    """Dense SDPA with a block_mask expanded to an element additive bias — the
    reference the sparse fingerprints compare against (byteΔ 0 ⇒ same path = SDPA)."""
    N = int(q.shape[2]); NQ, NK = block_mask.shape[-2], block_mask.shape[-1]
    em = mx.repeat(mx.repeat(block_mask.astype(mx.float32), N // NQ, -2), N // NK, -1)
    bias = mx.where(em > 0, mx.array(0.0), mx.array(-1e9, mx.float32)).astype(mx.float16)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias[None, None])


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

    # 2c) sparse D=128 ENGAGEMENT → real NAX sparse (byteΔ in the NAX window vs masked
    # SDPA).  Shape = SPARSE_ENGAGE_SPEC (center of a proven non-causal winning region,
    # 07-13 re-map: D=128, N=8192, B·H=12, density~0.15 ≤ 0.30 ceiling, N∈[4096,8192]).
    # The old N=2048/density-0.51 cell was measured-losing → it is now the DELEGATION
    # fingerprint (2d).  byteΔ==0 here would mean a silent SDPA fallback (the exact
    # failure the 2.62.0 Phase-3 caught); the anti-recurrence lock guards the shape.
    S = SPARSE_ENGAGE_SPEC
    scE = 1.0 / (S["D"] ** 0.5); NBe = S["N"] // S["BT"]
    q, k, v = _gen(S["B"], S["H"], S["N"], S["D"])
    mx.random.seed(20260713)  # deterministic engagement mask (~0.15 density)
    ii = mx.arange(NBe).reshape(NBe, 1); jj = mx.arange(NBe).reshape(1, NBe)
    maskE = ((jj <= ii) & (mx.random.uniform(shape=(NBe, NBe)) < 0.30)) | (ii == jj)
    densE = float(mx.mean(maskE.astype(mx.float32)).item())
    d = _delta(flash_attention_sparse(q, k, v, maskE, scale=scE, causal=False),
               _masked_sdpa(q, k, v, maskE, scE))
    fps["sparse_D128_nc_engage_nax"] = d
    if not (1e-7 < d < 3e-2):
        failures.append(("sparse_D128_nc_engage_nax", d,
                         f"expected NAX-sparse engagement byteΔ ∈ (1e-7, 3e-2) at density={densE:.3f} "
                         f"(shape {S}); byteΔ==0 ⇒ silent SDPA fallback — re-check routing/window"))

    # 2d) sparse D=128 DELEGATION → the measured-losing cell (N=2048, density~0.51,
    # non-causal) MUST delegate to dense SDPA (byteΔ == 0.0).  07-13 re-map found this
    # cell 0/36 winning (density 0.51 > every ceiling).  Recorded so the receipt PROVES
    # the delegation — turning the Phase-3 incident into permanent coverage.
    Nd, Dd = 2048, 128; scD = 1.0 / (Dd ** 0.5)
    q, k, v = _gen(1, 4, Nd, Dd)
    maskD = make_causal_block_mask(Nd, head_dim=Dd)
    d = _delta(flash_attention_sparse(q, k, v, maskD, scale=scD, causal=False),
               _masked_sdpa(q, k, v, maskD, scD))
    fps["sparse_D128_nc_delegate_sdpa"] = d
    if d != 0.0:
        failures.append(("sparse_D128_nc_delegate_sdpa", d,
                         "expected measured-losing sparse cell (N=2048/density~0.51) to DELEGATE to "
                         "SDPA (byteΔ == 0.0); nonzero ⇒ it engaged a kernel it should not"))

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
    gate_result = "PASS" if not failures else "FAIL"
    # Content hash binds the receipt to the exact fingerprint values.
    fp_hash = hashlib.sha256(
        json.dumps(fps, sort_keys=True).encode()).hexdigest()
    stamp = {
        "release_version": mlx_mfa.__version__,
        "git_sha": _git_sha(),
        "mlx_version": mx.__version__,
        "device": dev.get("device_name", "?"),
        "chip": dev.get("chip_name", "?"),
        "is_m5_plus": bool(dev.get("is_m5_plus", False)),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "has_nax": ok,
        "nax_reason": code,
        "fingerprints": fps,
        "fingerprints_sha256": fp_hash,
        "gate": gate_result,
    }
    # (a) full record → gitignored archive (off the tracked tree, journal policy).
    archive_dir = os.path.join(_REPO, "devnotes", "release-fingerprints")
    os.makedirs(archive_dir, exist_ok=True)
    path = os.path.join(archive_dir, f"m5-nax-fingerprints-{mlx_mfa.__version__}.json")
    with open(path, "w") as fh:
        json.dump(stamp, fh, indent=2)
    print(f"✓ archived fingerprints → {os.path.relpath(path)}")
    # (b) TRACKED receipt → release-gate/ (volet D): the publish.yml precondition
    # reads THIS (a fresh GitHub checkout cannot see the gitignored archive).
    # Commit it as a release-prep step; check_m5_gate_fingerprint.py verifies it
    # at publish time (git_sha freshness + PASS + NAX live).  Tracked but
    # sdist-excluded (does not ship to users).
    receipt_dir = os.path.join(_REPO, "release-gate")
    os.makedirs(receipt_dir, exist_ok=True)
    receipt = os.path.join(receipt_dir, f"m5-gate-{mlx_mfa.__version__}.json")
    with open(receipt, "w") as fh:
        json.dump(stamp, fh, indent=2)
    print(f"✓ wrote tracked release receipt → {os.path.relpath(receipt)} "
          f"(git add + commit this before dispatching publish.yml)")
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
