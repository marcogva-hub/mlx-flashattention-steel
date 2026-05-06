# v2.32.0 niche-shape dispatch table (Sprint A.6)

**Date:** 2026-05-06
**Branch:** `experiment/v32-sdpa-routing`
**Bench:** `bench/v32_kernel_sweep.py` (subprocess-isolated, 5 runs/config)
**Raw data:** [`v32-kernel-sweep.json`](v32-kernel-sweep.json)

## Executive summary

For shapes where MLX SDPA NAX is **not** the optimal forward path on M5+
(non-canonical D, exotic features, decode patterns), this table records
the empirically-best mlx-mfa kernel from the v2.32.0 niche-shape sweep.

The Sprint B routing predicate (`mlx_mfa.dispatch_policy.should_use_mfa`)
encodes these as carve-outs in `_should_use_mfa_m5_nax_carveout()`.

## Sweep matrix

15 shapes × 3 backends (`sdpa`, `mfa`, `auto`) on M5 Max, fp16.

[Final results to be inserted from `bench/v32_kernel_sweep_analyze.py`
output once the sweep completes.]

```
[Sprint A.6 will fill this section with the analyzer table:
shape, sdpa ms, mfa ms, auto ms, mfa/sdpa, auto/sdpa, verdict.]
```

## Verdict per shape (template)

| Shape | D | qL/kL | causal | Best backend | Margin vs SDPA | Carve-out? |
|---|---:|---|:---:|---|---:|:---:|
| whisper-base | 80 | 1500/1500 | no | SDPA (mfa unsupported) | — | no — handled by `_can_use_mfa` |
| gpt-neo-d96 | 96 | 2048/2048 | yes | SDPA (mfa unsupported) | — | no |
| codestral-d192 | 192 | 2048/2048 | yes | SDPA (mfa unsupported) | — | no |
| custom-d256 | 256 | 2048/2048 | no | [TBD] | [TBD] | [TBD] |
| llama-decode-8k | 128 | 1/8192 | no | [TBD — likely MFA via cross-attn rule] | [TBD] | covered by cross-attn rule |
| llama-decode-32k | 128 | 1/32768 | no | [TBD] | [TBD] | covered by cross-attn rule |
| flashvsr-dense | 64 | 4096/4096 | no | [TBD — Sprint 4 history says MFA wins] | [TBD] | [TBD] |
| llama-prefill-2k | 128 | 2048/2048 | yes | [TBD] | [TBD] | [TBD] |
| llama-prefill-4k | 128 | 4096/4096 | yes | [TBD] | [TBD] | [TBD] |
| llama-prefill-8k | 128 | 8192/8192 | yes | [TBD] | [TBD] | [TBD] |
| ltx2-cross | 64 | 2048/14000 | no | [TBD] | [TBD] | covered by cross-attn rule |
| seedvr2-small | 128 | 26730/26730 | no | [TBD] | [TBD] | [TBD] |
| cogvideox | 128 | 70200/70200 | no | [TBD] | [TBD] | [TBD] |
| canonical-d128-4k | 128 | 4096/4096 | no | [TBD] | [TBD] | control — expect SDPA |
| canonical-d64-8k | 64 | 8192/8192 | no | [TBD] | [TBD] | control — expect SDPA |

## Known-already-correct dispatch

These shapes are already routed correctly without v2.32.0 changes:

- `head_dim ∈ {80, 96, 192}` — `_can_use_mfa()` returns False ⇒ SDPA fallback.
- `qL ≤ 4096 ∧ kL ≥ 4096` (cross-attn small-Q large-K) — existing
  cross-attn rule routes to MFA.
- `kL ≤ 512 ∧ qL > 8192` — existing cross-attn rule routes to SDPA.

The v2.32.0 SDPA routing layer adds:

- Canonical D ∈ {64, 128} on M5+ NAX (qL > 8) → SDPA (was MFA via M3+
  thresholds for some shapes).

## Carve-outs to encode

[After sweep analysis, list the specific (head_dim, seq_len, kv_seq_len,
causal) tuples where MFA wins by > 5%. These get added as predicate
clauses in `_should_use_mfa_m5_nax_carveout()`.]

## Methodology caveats

- Single-session bench (per `CLAUDE_V6_NAX.md` Artifact #5, this is
  staging data, not publication-grade data).
- Cross-session re-validation needed before any carve-out becomes
  permanent. Multi-session protocol available via
  `bench/v32_multisession_capture.py`.
- The sweep on canonical shapes serves as a control: SDPA is expected
  to win or match on D=128 long-N self-attn and D=64 long-N self-attn.
  If it does, that confirms the strategic shift; if a control shape
  unexpectedly favors MFA, that's a methodology red flag.
