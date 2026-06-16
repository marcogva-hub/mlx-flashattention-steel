# Sprint III-12b — TQ Claim Archaeology + Correction (pre-v2.55.0)

**Date:** 2026-06-16 · **Executor:** Opus 4.8 High · HEAD `508cf37`, macOS 26.6, M5 Max, mlx 0.31.2.

## Verdict: CASE 2 — the TQ "6–14× faster" claim is REAL. III-12's "inverted" finding was WRONG
## (it measured the wrong baseline). The README claim only lost its baseline ("vs the fused TQ kernel").

### R.1 — Archaeology (the record, not memory)
`docs/PERF_CLAIMS.md:47` + `sprint-III-2-report.md` (the claim's origin, §AA.5 FULL_INVERSION) state
the claim PRECISELY, with absolutes:

| metric (S=4K → S=16K) | III-2 (original OS) |
|---|---|
| attend-only **vs fused TQ kernel** | 13.8× (4.11→0.298 ms) / 22.1× (15.07→0.683 ms) |
| full `step()` **vs fused** | 5.99× (4.65→0.78 ms) / 14.42× (16.68→1.16 ms) |

The "6–14× faster" is the new per-step **gather/dequant (`tq_decode.py`) + Apple SDPA** path
(`TurboQuantPagedInferenceContext.step()`, default-on, opt-out `MFA_DISABLE_TQ_DECODE_SDPA=1`)
**vs the OLD fused TQ attend kernel** it replaced (the II-7 14×-dense decode floor). Correctly
directional, with ms absolutes. The baseline is the **fused TQ kernel**, NOT fp16.

### Why III-12 went wrong
III-12 benched `flash_attention_paged_varlen_turboquant` (the FUSED kernel) vs **fp16 paged decode**
— neither arm of the claim. It then read the (fused/fp16) ratio as the claim and called it
"inverted." The fused kernel is the SLOW baseline the claim BEATS; comparing it to fp16 is a
different question. Lesson: read the claim's exact path+baseline from the record before measuring.

### R.2 — Re-measured the REAL path on 26.6 (Hq=32 Hkv=8 D=128 tq3b, 2 runs)
| S | new path (`tq_decode_attend`) | fused TQ | **fused/new** | new vs fp16-dense |
|---|---|---|---|---|
| 4096 | 0.32–0.67 ms | ~4.32 ms | **6.5–13.5×** | 1.4–2.95× slower |
| 16384 | ~0.74 ms | ~16.84 ms | **~22.5×** (stable ~2%) | ~2.3× slower |

The claim **HOLDS on 26.6**: the new path is ~6.5–23× faster than the fused TQ kernel (S=16K matches
III-2's 22.1× attend-only spot-on; S=4K is 6.5–13.5×, variance from the new path being sub-ms →
canonical-regime clock-state bimodality). Provenance + raw: `iii12b_tq_claim_26.6_run{1,2}.log`.

### R.3 — promotion / honest framing
- The §AA.5 promotion (new path replaces fused for N_q=1 decode) is CONFIRMED — the new path wins
  6.5–23× vs the fused kernel on 26.6. Promotion holds.
- HONEST nuance for the README: the new path is still **~1.4–3× slower than fp16 *dense* decode**
  (the "gap to dense floor" — III-2 logged 1.66×/2.4×; 26.6 ~1.4–3×). So "6–14× faster" is faster
  **than the fused TQ kernel it replaced**, NOT faster than fp16 dense. TQ's net value remains the
  **~4–5× KV-memory reduction at cos ~0.96** (enables longer context / higher concurrency); the
  decode path now pays only a ~1.4–3× latency tax vs fp16 dense (down from the old ~14–52× fused floor).

### Institutional lesson (codified)
The README's bare "6–14× faster" with no baseline is what let it be misread (by a reader AND by
III-12). **Every perf claim must state numerator, denominator, direction, and an absolute (ms):**
"new path 0.75 ms vs fused 16.8 ms → 22× faster than the fused kernel (still ~2.3× slower than fp16
dense)." A bare ratio is ambiguous by construction. Added to the pattern catalogue (no-bare-ratios).

## Correction to III-12
`sprint-III-12-report.md`'s "the TQ claim is INVERTED" headline is RETRACTED — it measured the
fused kernel vs fp16, not the claim's path. The claim is real; only its baseline was undisclosed.
