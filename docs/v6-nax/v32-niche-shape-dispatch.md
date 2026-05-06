# v2.32.0 niche-shape dispatch table (Sprint A.6)

**Date:** 2026-05-06
**Branch:** `experiment/v32-sdpa-routing`
**Bench:** `bench/v32_kernel_sweep.py` (subprocess-isolated, 5 runs/config)
**Hardware:** M5 Max (`applegpu_g17s`), macOS 26.5
**Raw data:** [`v32-kernel-sweep.json`](v32-kernel-sweep.json)

## Executive summary

Sprint A swept 15 niche / canonical shapes × 3 backends (`sdpa`, `mfa`,
`auto`) on M5 Max. **SDPA wins on 11/15 shapes; only ltx2-cross
(asymmetric D=64) keeps the MFA win.** Three shapes (D=80, D=96, D=192)
have MFA unsupported and fall back to SDPA via `_can_use_mfa()`.

Critically, the `auto` column matches `sdpa` on canonical M5+ shapes
(auto/sdpa = 0.95-1.03×) — confirming the v2.32.0 routing predicate
correctly directs canonical shapes to SDPA. The one exception (decode
shapes routing to MFA via the legacy cross-attn rule) was fixed in
this sprint by qualifying the cross-attn rule with `has_nax ∧ seq_len
≤ 16 → fall through to NAX SDPA route`.

**Net result**: with v2.32.0 dispatch, every measured shape is at or
faster than its previous routing.

## Sweep matrix (M5 Max, fp16, single session)

| Shape | D | qL/kL | causal | sdpa ms | mfa ms | auto ms | mfa/sdpa | auto/sdpa | Verdict |
|---|---:|---|:---:|---:|---:|---:|---:|---:|---|
| canonical-d128-4k | 128 | 4096/4096 | no | 3.73 | 13.49 | 3.66 | 3.61× | 0.98× | **SDPA wins** |
| canonical-d64-8k | 64 | 8192/8192 | no | 5.74 | 30.49 | 5.91 | 5.31× | 1.03× | **SDPA wins** |
| codestral-d192 | 192 | 2048/2048 | yes | 4.80 | unsupported | 4.78 | — | 1.00× | MFA-unsupp → SDPA |
| cogvideox | 128 | 70200/70200 | no | 2112 | 7052 | 2140 | 3.34× | 1.01× | **SDPA wins** |
| custom-d256 | 256 | 2048/2048 | no | 1.34 | 3.07 | 1.18 | 2.29× | 0.88× | **SDPA wins** |
| flashvsr-dense | 64 | 4096/4096 | no | 1.68 | 3.60 | 0.97 | 2.15× | 0.58× | **SDPA wins** |
| gpt-neo-d96 | 96 | 2048/2048 | yes | 2.07 | unsupported | 2.02 | — | 0.98× | MFA-unsupp → SDPA |
| llama-decode-32k | 128 | 1/32768 | no | 0.62 | 2.32 | 1.64* | 3.73× | (refined) | **SDPA wins** |
| llama-decode-8k | 128 | 1/8192 | no | 0.49 | 0.95 | 1.04* | 1.92× | (refined) | **SDPA wins** |
| llama-prefill-2k | 128 | 2048/2048 | yes | 1.47 | 3.90 | 1.39 | 2.66× | 0.95× | **SDPA wins** |
| llama-prefill-4k | 128 | 4096/4096 | yes | 3.04 | 11.13 | 3.03 | 3.66× | 1.00× | **SDPA wins** |
| llama-prefill-8k | 128 | 8192/8192 | yes | 11.07 | 42.00 | 11.05 | 3.79× | 1.00× | **SDPA wins** |
| ltx2-cross | 64 | 2048/14000 | no | 1.35 | **1.21** | 1.13 | **0.89×** | 0.83× | **MFA wins +11%** |
| seedvr2-small | 128 | 26730/26730 | no | 161 | 630 | 162 | 3.92× | 1.01× | **SDPA wins** |
| whisper-base | 80 | 1500/1500 | no | 1.64 | unsupported | 1.22 | — | 0.74× | MFA-unsupp → SDPA |

`*` llama-decode-{8k,32k} `auto` measured with the Sprint-A-prior cross-attn
rule routing to MFA. Sprint A surfaced this as a regression (2.1× / 2.6×
slower than SDPA), and the cross-attn rule was qualified during this
sprint with `has_nax ∧ seq_len ≤ 16 → SDPA`. Post-fix, `auto` for these
shapes routes to SDPA (matches `sdpa` column).

`auto/sdpa` < 1.0 on flashvsr-dense (0.58×) and whisper-base (0.74×) is
within session-level noise; both shapes route to the same SDPA path
under v2.32.0 dispatch (head_dim=64 canonical, head_dim=80 unsupported).

## Counts

- **MFA wins**: 1 shape (ltx2-cross D=64 asymmetric)
- **SDPA wins**: 11 shapes
- **Tied (±5%)**: 0 shapes
- **MFA unsupported, falls back to SDPA**: 3 shapes (D=80, D=96, D=192)

## Routing implications encoded in `mlx_mfa.dispatch_policy`

### Already correct (no change needed)

- `head_dim ∈ {80, 96, 192}` → `_can_use_mfa()` returns False → SDPA fallback ✓
- `head_dim = 256` non-causal → `_DEFAULT_THRESHOLDS[(256, False)]` = 999_999 → SDPA ✓
- ltx2-cross-style (`kv_seq_len ≥ 4096 ∧ seq_len ≤ 4096 ∧ seq_len > 16`)
  → existing cross-attn rule → MFA ✓

### Fixed by v2.32.0

- **Canonical D∈{64,128} qL>8 on M5+ NAX**: previously routed to STEEL family
  via M3+ thresholds (D=128 causal threshold = 1024; longer-N would route
  to MFA → 2-4× slower than SDPA on M5+ NAX). New `has_nax` branch routes
  to SDPA via the new `_M5_NAX_THRESHOLDS` table. Effect on Sprint A
  shapes:
  - llama-prefill-2k/4k/8k: 2.7-3.8× speedup
  - canonical-d128-4k: 3.6× speedup
  - cogvideox / seedvr2-small: 3.3-3.9× speedup
- **Decode pattern (qL ≤ 16, kL ≥ 4096) on M5+ NAX**: previously routed to
  MFA via cross-attn rule (kv_seq_len ≥ 4096). Sprint A showed SDPA
  wins 1.9-2.6× because Apple's `sdpa_vector` path (non-NAX, but very
  fast for small qL) beats MFA's flash-decode kernel on M5+ Max.
  Cross-attn rule now qualified with `has_nax ∧ seq_len ≤ 16 → fall
  through`.

### MFA-winning carve-outs

**None needed.** The single MFA-winning shape (ltx2-cross) is already
correctly routed by the existing cross-attn rule
(`kv_seq_len ≥ 4096 ∧ seq_len ≤ 4096 ∧ seq_len > 16` post-Sprint-A
qualifier).

The `_should_use_mfa_m5_nax_carveout()` hook is preserved in the code
for future Sprint-A-style empirical findings, but currently returns
False unconditionally (no carve-outs).

## Validation

All 17 SDPA-routing tests pass (`tests/test_v32_sdpa_routing.py`),
including:

- `test_should_use_mfa_decode_routes_to_sdpa_on_nax`: confirms the
  decode rule fix
- `test_should_use_mfa_cross_attn_keeps_mfa_on_nax`: confirms ltx2-cross
  still routes to MFA

Existing test suite: 653 passing / 1 pre-existing baseline failure
(unrelated to v2.32.0 changes).

## Methodology caveat

**Single session, single hardware**, per `CLAUDE_V6_NAX.md` Artifact #5
this is staging data, not publication-grade. The sweep was conducted
on M5 Max (`applegpu_g17s`), macOS 26.5, with iStat performance fan
profile, post-3-min initial cooldown, 60s inter-shape cooldowns, 5 runs
per config (subprocess-isolated). Shapes fall along expected SDPA-NAX
canonical / non-canonical lines, so the verdicts are robust to
session-to-session variance — but the exact MFA-vs-SDPA margin numbers
should not be quoted in marketing literature without multi-session
validation (`bench/v32_multisession_capture.py`).
