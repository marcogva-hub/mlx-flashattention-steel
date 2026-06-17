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


def _fp32_oracle(q, k, v, scale, causal):
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    Hq, Hk = q.shape[1], k.shape[1]
    if Hq != Hk:
        r = Hq // Hk; kf = mx.repeat(kf, r, axis=1); vf = mx.repeat(vf, r, axis=1)
    s = (qf @ kf.swapaxes(-1, -2)) * scale
    N, S = q.shape[2], k.shape[2]
    if causal:
        cm = (mx.arange(N)[:, None] >= mx.arange(S)[None, :] + (S - N)).astype(mx.float32)
        s = mx.where(cm > 0, s, mx.array(-1e30, mx.float32))
    return mx.softmax(s, axis=-1) @ vf


def _qkv(B, H, N, D, Hk=None, Sk=None):
    Hk = Hk or H; Sk = Sk or N
    fq = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    fk = (mx.random.uniform(-1, 1, (B, Hk, Sk, D)) * 0.1).astype(mx.float16)
    fv = (mx.random.uniform(-1, 1, (B, Hk, Sk, D)) * 0.1).astype(mx.float16)
    mx.eval(fq, fk, fv); return fq, fk, fv


def _assert_correct(q, k, v, scale, causal):
    o = flash_attention(q, k, v, scale=scale, causal=causal, backend="mfa")
    ref = _fp32_oracle(q, k, v, scale, causal)
    mx.eval(o, ref)
    assert bool(mx.all(mx.isfinite(o.astype(mx.float32))).item()), "non-finite"
    d = float(mx.max(mx.abs(o.astype(mx.float32) - ref)).item())
    assert d < _TOL, f"max_abs_err {d} exceeds {_TOL}"


# ── forced-variant correctness (vs independent fp32 oracle) ──────────────────
class TestSteelVariantCorrectness:
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

    def test_v4_forced(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V4", "1")
        q, k, v = _qkv(2, 8, 4096, 128); _assert_correct(q, k, v, 1 / math.sqrt(128), False)

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
    catch selection drift; this source lock does — see B2 report)."""

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
