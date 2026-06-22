"""Dense STEEL family correctness + dispatch LOCK (audit Phase B2, 2026-06-17).

Two locks for the `backend="mfa"` STEEL variants (V1/V2/V3/V4/V5/split-K/dsplit/
flash_decode):

1. **Forced-variant correctness** — each variant, engaged via its env knob +
   shape, validated vs an INDEPENDENT manual fp32 oracle (not SDPA, not another
   STEEL variant — lesson #11). Catches a variant breaking.

2. **Source-predicate threshold lock** — the STEEL variants are BYTE-IDENTICAL to
   each other (Δ=0.0, verified) so byte-identity cannot distinguish which runs,
   and timing is too flaky for CI. Instead this locks the dispatch THRESHOLDS in
   source (like the KD-5 / Gate-9 source locks): a change to `v3_min_N`,
   `m3_prefers_v1`, or the flash_decode gate trips CI, forcing a deliberate
   dispatch-map update (Phase F). The variant SELECTION map is sentinel-confirmed
   (env-toggle timing) in phase-B2-dense-steel-report.md; this lock guards its
   source predicates from silent drift.

M5+-gated (the routes under test are M5-specific).
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import flash_attention
from mlx_mfa.attention import _get_is_m5_plus_cached

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(),
    reason="dense STEEL family lock asserts M5+ backend=mfa routes",
)

mx.random.seed(0)
_TOL = 3e-2
_ATTN_CPP = Path(__file__).resolve().parent.parent / "csrc" / "mfa_attention.cpp"

# T2-1 (audit H-10, 2026-06-21): these locks ran ONLY at 0.1 input scale — the
# regime that hid the II-6 fused-dKdV corruption.  Every correctness cell now
# runs at BOTH 0.1 (kept) AND realistic unit scale (std≈1.0, normal), validated
# vs the SAME independent fp32 oracle.  Toy keeps the original ABSOLUTE bound;
# unit uses a scale-invariant RELATIVE bound (fp16 attention rel-err is ≲1e-2).
# A unit-scale failure is a BUG-DISCOVERY signal — investigate which-binary;
# do NOT loosen it without confirming the kernel is correct.
_REL_TOL = 5e-2
_MAG = {"mode": "toy"}


def _gen(shape):
    if _MAG["mode"] == "unit":
        return mx.random.normal(shape).astype(mx.float16)         # std ≈ 1.0
    return (mx.random.uniform(-1, 1, shape) * 0.1).astype(mx.float16)


def _fp32_oracle(q, k, v, scale, causal):
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    Hq, Hk = q.shape[1], k.shape[1]
    if Hq != Hk:
        r = Hq // Hk; kf = mx.repeat(kf, r, axis=1); vf = mx.repeat(vf, r, axis=1)
    s = (qf @ kf.swapaxes(-1, -2)) * scale
    N, S = q.shape[2], k.shape[2]
    if causal:
        # T2-1 ORACLE FIX (audit, 2026-06-21): the prior formula
        # `i >= j + (S-N)` had the (S-N) sign FLIPPED — for N<S (the flash-decode
        # cell, N=1,S=2048) it masked EVERY key → a uniform-softmax reference.
        # Toy scale hid it (tiny scores → the kernel also ≈ uniform, so they
        # agreed); at unit scale the kernel's correctly-peaked softmax diverged
        # (rel 1.6).  Query i sits at absolute position i+(S-N) and attends key j
        # iff i+(S-N) >= j.  Kernel verified CORRECT vs this fixed oracle (4.3e-5)
        # and vs non-causal-full (the N=1 query is the last position → attends
        # all keys).  Dormant-same-formula in sibling locks only test N==S.
        cm = (mx.arange(N)[:, None] + (S - N) >= mx.arange(S)[None, :]).astype(mx.float32)
        s = mx.where(cm > 0, s, mx.array(-1e30, mx.float32))
    return mx.softmax(s, axis=-1) @ vf


def _qkv(B, H, N, D, Hk=None, Sk=None):
    Hk = Hk or H; Sk = Sk or N
    fq = _gen((B, H, N, D))
    fk = _gen((B, Hk, Sk, D))
    fv = _gen((B, Hk, Sk, D))
    mx.eval(fq, fk, fv); return fq, fk, fv


def _assert_engaged(q, k, v, scale, causal, o):
    """Prove the forced MFA Metal kernel ACTUALLY ran (which-binary), not a
    silent SDPA fallback (volet-A / CC-07).  The STEEL/V2/V3 variants are
    byte-identical to *each other* (so no runtime lock distinguishes which
    variant ran — that is the source-predicate lock's job below), but the
    family as a whole is byte-DISTINCT from Apple SDPA: a real MFA forward
    differs from `scaled_dot_product_attention` by the kernel's own rounding.
    byteΔ == 0.0 ⇒ the call silently produced the SDPA bytes ⇒ green-on-wrong-
    binary; assert byteΔ > 0 so that can never pass unnoticed."""
    sdpa = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=("causal" if causal else None))
    mx.eval(sdpa)
    byte_delta = float(mx.max(mx.abs(
        o.astype(mx.float32) - sdpa.astype(mx.float32))).item())
    assert byte_delta > 0.0, (
        f"ENGAGEMENT FAILURE: backend='mfa' byteΔ-vs-SDPA == 0.0 — the MFA "
        f"kernel did not run; this is a silent SDPA fallback (the correctness "
        f"assert below would pass vacuously against SDPA-as-oracle).")


def _assert_correct(q, k, v, scale, causal):
    o = flash_attention(q, k, v, scale=scale, causal=causal, backend="mfa")
    ref = _fp32_oracle(q, k, v, scale, causal)
    mx.eval(o, ref)
    assert bool(mx.all(mx.isfinite(o.astype(mx.float32))).item()), "non-finite"
    # which-binary first, then correctness (so a vacuous SDPA-vs-SDPA pass is
    # impossible — engagement is proven before the oracle comparison).
    _assert_engaged(q, k, v, scale, causal, o)
    d = float(mx.max(mx.abs(o.astype(mx.float32) - ref)).item())
    if _MAG["mode"] == "unit":
        denom = float(mx.max(mx.abs(ref)).item()) + 1e-6
        rel = d / denom
        assert rel < _REL_TOL, f"unit-scale rel_err {rel:.3e} exceeds {_REL_TOL} (abs={d:.3e})"
    else:
        assert d < _TOL, f"toy-scale max_abs_err {d} exceeds {_TOL}"


# ── forced-variant correctness (vs independent fp32 oracle) ──────────────────
class TestSteelVariantCorrectness:
    # Run each cell below at BOTH input regimes (T2-1).
    @pytest.fixture(autouse=True, params=["toy", "unit"])
    def _regime(self, request):
        _MAG["mode"] = request.param
        yield
        _MAG["mode"] = "toy"

    def test_v3_d64_causal(self):  # default V3 (causal large-N)
        q, k, v = _qkv(2, 8, 4096, 64); _assert_correct(q, k, v, 1 / math.sqrt(64), True)

    def test_v3_d128_causal(self):
        q, k, v = _qkv(2, 8, 4096, 128); _assert_correct(q, k, v, 1 / math.sqrt(128), True)

    def test_v1_d128_causal_smallN(self, monkeypatch):  # below v3_min_N → V1
        q, k, v = _qkv(2, 8, 512, 128); _assert_correct(q, k, v, 1 / math.sqrt(128), True)

    def test_v2_d128_noncausal(self):  # V3 needs causal → V2
        q, k, v = _qkv(2, 8, 4096, 128); _assert_correct(q, k, v, 1 / math.sqrt(128), False)

    def test_v2_forced_disable_v3(self, monkeypatch):
        monkeypatch.setenv("MFA_DISABLE_V3", "1")
        q, k, v = _qkv(2, 8, 4096, 128); _assert_correct(q, k, v, 1 / math.sqrt(128), True)

    @pytest.mark.skip(reason="STEEL V4/V5 retired from build (Lot-2 chore); opt-in kernels removed")
    def test_v4_forced(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V4", "1")
        q, k, v = _qkv(2, 8, 4096, 128); _assert_correct(q, k, v, 1 / math.sqrt(128), False)

    @pytest.mark.skip(reason="STEEL V4/V5 retired from build (Lot-2 chore); opt-in kernels removed")
    def test_v5_forced(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V5", "1")
        q, k, v = _qkv(2, 8, 4096, 128); _assert_correct(q, k, v, 1 / math.sqrt(128), False)

    def test_splitk_forced(self, monkeypatch):
        monkeypatch.setenv("MFA_FORCE_SPLITK", "1")
        q, k, v = _qkv(2, 8, 4096, 128); _assert_correct(q, k, v, 1 / math.sqrt(128), False)

    def test_dsplit_d256(self):
        q, k, v = _qkv(2, 8, 2048, 256); _assert_correct(q, k, v, 1 / math.sqrt(256), False)

    def test_flash_decode(self):  # N_q=1, S>=256
        q, k, v = _qkv(2, 8, 1, 128, Sk=2048); _assert_correct(q, k, v, 1 / math.sqrt(128), True)

    def test_gqa_d128_causal(self):
        q, k, v = _qkv(2, 8, 4096, 128, Hk=2); _assert_correct(q, k, v, 1 / math.sqrt(128), True)


# ── source-predicate threshold lock (catches selection drift) ────────────────
class TestSteelDispatchThresholdsLocked:
    """Lock the dispatch thresholds in source. A change forces a deliberate
    dispatch-map update (the variants are byte-identical so no runtime lock can
    catch selection drift; this source lock does — see B2 report).

    # RUNTIME-INDISTINGUISHABLE: byte-identical variants, source-predicate is
    # the only available proof.  The STEEL V1/V2/V3/split-K/dsplit variants
    # produce byteΔ == 0 against each other (same math, different tiling), so a
    # reroute *among them* cannot be caught by a byteΔ fingerprint — only by
    # asserting the selection predicate still matches csrc/mfa_attention.cpp
    # below.  A reroute to an UNLISTED variant (or to SDPA) IS byte-distinct and
    # is caught by `_assert_engaged` in the correctness cells above.  (volet-A /
    # CC-07: this limitation is now explicit, not silently source-trusted.)"""

    def _src(self):
        return _ATTN_CPP.read_text(encoding="utf-8")

    def test_v3_min_N_threshold(self):
        import re
        assert re.search(r"v3_min_N\s*=\s*\(D\s*==\s*64\)\s*\?\s*4096\s*:\s*2048", self._src()), (
            "v3_min_N threshold changed — re-validate the V3 auto-route crossover "
            "(B2 report) and update the dispatch map before changing this lock"
        )

    def test_flash_decode_gate(self):
        assert "N <= 4 && S >= 256" in self._src(), "flash_decode gate changed — update dispatch map"

    def test_m3_prefers_v1_predicate(self):
        s = self._src()
        assert "m3_prefers_v1" in s and "is_m3_plus_steel && D <= 128 && params_.causal" in s, (
            "m3_prefers_v1 predicate changed — the M5 causal→V1/V3 routing moved; update the map"
        )
