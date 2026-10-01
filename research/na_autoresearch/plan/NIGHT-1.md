# NIGHT-1 plan — dense D64+D128 exhaustive screening (dimensioned)

Submitted for Marco's GO. Nothing runs at night without it. Base master `b1e7ac2` (v2.62.1),
branch `research/na-autoresearch`, venv `.venv`, MLX 0.31.2, M5 Max, Automatic + recalibrated curve.

## Queue (generated, reproducible)
`benchmarks/results/autoresearch/night1/night1_queue.jsonl` — **13,940 cells**, all dense, all valid
(engagement + fp32/SDPA oracle enforced per cell; 0 raises in calibration).

| block | what | cells | GPU h (measured cost model) |
|---|---|---|---|
| **A** (priority) | 40 valid tiles + default × D{64,128} × dtype{fp16,bf16} × N{2048,4096,8192,16384,32768} × bh{2×8,1×4,1×12} | 2,460 | ~0.9 h |
| **B** | orthogonal knob product {RELAXED_PRECISION=0, UNROLL={none,2,4}, MAX_THREADS={256,512}, RELAXED×UNROLL} × full tile grid × D×dtype×N × bh{2×8,1×12} | 11,480 | ~4.1 h |
| **base total** | | **13,940** | **~5.05 h** |
| + auto-zoom | top-K tiles → valid neighbours, until dry | dynamic | ~1–2 h |
| + backlog | **sparse-BT32 (Phase-1: needs sparse executor)** | — | (deferred) |

Cost model (measured, Phase-0 calibration `rep_run`): per-cell wall s/N = {2048:0.344, 4096:0.220,
8192:0.852, 16384:1.70⁺, 32768:3.40⁺} (⁺ extrapolated ~2×). Effective screening rate **~7,000 cells/h**.

## Honest sizing finding (surfaced for decision)
**Dense screening is cheap** — the entire dense *tile* space (block A) screens exhaustively in **<1 h**;
even tiles × the orthogonal-knob product (block A+B) is **~5 h**. It cannot fill a padded 8 h alone
without diminishing marginal value. The night reaches sunrise via **base (5 h) + auto-zoom + the 07:00
deadline cap**: launched ~23:00, the 5 h base + zoom runs to dry or 07:00, and stops clean (GPU measured
usefully — the whole dense envelope covered). A genuinely-full **multi-family** 8 h night is the NIGHT-2
shape, once the Phase-1 **sparse / backward / decode** screening executors exist (each ~1 executor +
its own oracle; the census sized them: sparse 288, backward+decode 750, varlen/GNA/conv3d 202, frontiers
480 coarse cells). **Decision for Marco**: (a) run NIGHT-1 as dimensioned (dense-exhaustive ~5 h + zoom),
or (b) hold NIGHT-1 until the sparse executor lands so the backlog is real and the night is multi-family.

## Head block (day-floors) — judged against
Measured first, into `day_floors.json`: dense D128 default (~3.0 ms), dense D64 default (~1.35 ms), each
with cos vs SDPA. All night verdicts are deltas vs THESE. The sentinel (fixed D128 spot) re-measures every
25 cells; drift > 15 % over the floor → PAUSE.

## PAUSE / STOP criteria (watchdog)
- **PAUSE** (backoff 120 s, ≤6): sentinel drift > `NAAR_DRIFT_FRAC` (15 %) over the day-floor; sustained
  GPU freq < 1300 MHz over the window (macmon).
- **STOP** (clean, reason logged): RAM free < 8 GB; >6 pauses; `STOP` file; 07:00 deadline; queue+zoom+
  backlog exhausted. No degraded number is ever written as clean.

## Launch
```bash
research/na_autoresearch/launch_night.sh \
  benchmarks/results/autoresearch/night1/night1_queue.jsonl
# deadline auto = next 07:00; runs detached under caffeinate; survives session end.
# morning: benchmarks/results/autoresearch/night-<ts>/morning_report.md
```

## Morning → next session
The report ranks tiles/knobs per shape vs the day-floor default and lists Stage-2 candidates
(screen-margin > 3 %). Those go to **in-session hardened confirmation** (≥10 fresh processes, 2 orders,
outliers, fp32 oracle, fingerprints, verdict vs day-floor) — cost **~1 min (canonical, sub-1.5 ms cells)
to ~11 min (§4-strict, ≥1.5 ms cells)** per candidate, so a 2–3 h session confirms ~15–100 candidates.
Only survivors reach a Stage-3 promotion checkpoint (WM2 rule; 2.63.0 vehicle).

## Early live signals (Stage-1 only — NOT decisions)
- D128/N4096 tile **32·32·2** screens ~5–7 % under the default; D64 default-NAX (2.06 ms) loses to its own
  32·32·2 (1.50 ms). **BK=128** (M5 doctrine target) untested vs current default BK=32 — in the queue.
- RELAXED_PRECISION=0 / UNROLL=none showed ratio-vs-SDPA flips at D128/N4096 — but within screening noise
  (~0.3 ms run-to-run); flagged for Stage-2, claimed by nothing.
- These are exactly why the funnel exists: screening ranks, hardened confirms, Marco promotes.
