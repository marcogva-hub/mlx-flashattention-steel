# Experimental Path Status Matrix (v2.9.2)

Date: 2026-03-12
Device: Apple M1 Max (gen 13, non-M3+)
Primary artifacts:
- `notes/experimental_path_triage_latest.json`
- `notes/sage_decode_matrix_post_bwd_latest.json`
- `notes/paged_sharedprefix_matrix_latest.json`

## Experimental Forward Paths (V3/V4/V5)

| Path | Evidence snapshot | Hardware dependency | Recommendation |
|---|---|---|---|
| V3 | `clear_win=3`, `losing=13` across D={64,128}, N={2048,8192}, causal on/off; wins are narrow (mostly causal N=2048) | Works on M1/M2 for D=64/128, but unstable outside narrow regimes | **Keep (research-only)**. Do not auto-promote. Keep opt-in for targeted experiments only. |
| V4 | `ineligible=16` on current M1 hardware (falls through to V2); simulated M3 probe: `ratio_vs_v2=0.39x` (losing) | M3+ only in real dispatch | **Park / de-emphasize** on M1/M2. Keep behind explicit opt-in and hardware gate. |
| V5 | `clear_win=1`, `losing=15`; one narrow win, broad regressions | All gens, but benchmark evidence still weak | **Keep (experimental opt-in)**. Not production candidate; no broad promotion. |

## Specialized Runtime Paths (existing evidence)

| Path | Existing benchmark evidence | Recommendation |
|---|---|---|
| Sage decode (QuantizedKVCache) | `sage_win=13`, `maybe=4`, `losing=223` | Keep as **specialized decode backend** only; narrow auto policy remains correct. |
| Paged decode step | `clear_win=0`, `maybe_win=1`, `no_win=1`, `losing=28` | Keep **explicit-first / narrow**; no broad auto promotion. |
| Shared-prefix | `clear_win=4`, `no_win=3`, `losing=1` | Keep as targeted runtime optimization where prefix reuse is real. |
| Splitfuse | `clear_win=3`, `losing=5` | Keep reachable via runtime helper; shape-sensitive, no broad claims. |

## Selective AOT Candidate Probes (cold-start)

| Candidate kernel probe | First call | Steady-state | First/steady | Recommendation |
|---|---:|---:|---:|---|
| `sage_decode_d128_gqa2` | 7.66 ms | 2.71 ms | 2.8x | **AOT candidate: YES** (hot decode-specialized path). |
| `paged_gather_d128` | 5.39 ms | 0.66 ms | 8.2x | **AOT candidate: YES** (paged decode gather hot path). |
| `paged_steel_d128` | 19.55 ms | 4.92 ms | 4.0x | **AOT candidate: DEFER** (not the dominant decode path when `N_q<=4`). |

## Keep / Park Summary

- **Keep production default**: V2 dense.
- **Keep specialized**: Sage decode, shared-prefix, splitfuse.
- **Keep experimental opt-in**: V3, V5.
- **Park/de-emphasize on current hardware**: V4 (M3+ dependent, no current-hardware win evidence).
- **Selective AOT next step**: Sage decode hot kernels + paged gather hot kernel(s); do not broaden AOT coverage beyond benchmark-backed kernels.
