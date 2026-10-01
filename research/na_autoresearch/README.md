# NA-autoresearch — M5 Max V6-NAX tuning campaign (infrastructure)

Redo the autoresearch on **everything** — every kernel, every tile/layout axis — with
modern measurement infrastructure (clean-regime floors, outlier protocol, freq gate),
like the M1 Max campaign (V3 pattern, +47%). Promotion vehicle: **2.63.0**, published
when gains justify it. Base: master `b1e7ac2` (v2.62.1). Branch: `research/na-autoresearch`.

## Location (consigned per Marco's "research/ or the conventional place you judge")
- **Infra code** → `research/na_autoresearch/` (runner + generators + launch). Marco's named
  option; keeps the campaign self-contained.
- **Dated evidence** → `benchmarks/results/autoresearch/` (repo convention: dated evidence under
  `benchmarks/results/`). Night runs land here (`night-<ts>/{results.jsonl, day_floors.json,
  macmon.jsonl, run_summary.json, morning_report.md}`).
- **Plan + census docs** → `research/na_autoresearch/{plan,census}/` and `devnotes/` (untracked).
- Nothing committed/pushed without a Marco checkpoint (zéro push).

## The funnel (worker is the SCRIPT, not the session)
1. **Stage 1 — screening** (`runner/`): in-process, 1 order, few reps → RANK. The detached night
   runner does ONLY this. Each cell forces the NAX kernel (`_ext.v6_nax_forward`) so tile knobs
   engage at any D, proves engagement (byteΔ-vs-SDPA > 0) and correctness (cos-vs-SDPA ≥ 0.999),
   and records `median_ms` (NAX@tile) + `sdpa_ms` (production competitor).
2. **Stage 2 — hardened confirmation** (in-session, NOT at night): ≥10 fresh processes, 2 orders,
   outliers declared, fp32 oracle, fingerprints, verdict vs the day-floor — only for candidates
   with screening margin above threshold.
3. **Stage 3 — promotion** (in-session, Marco checkpoint): WM2 rule (gain > floor both orders AND
   zero regression on the envelope; three axes; atomic commit). NEVER at night.

## Operate
```bash
# build a queue (dense D64/D128 tile grid × prod shapes)
.venv/bin/python -m research.na_autoresearch.runner.gen_queue \
    --out night1.jsonl --dtypes float16,bfloat16 --Ns 2048,4096,8192 --bh 2x8,1x4,1x12 [--fine]
# launch detached (deadline = next 07:00; runs under caffeinate; survives session end)
research/na_autoresearch/launch_night.sh night1.jsonl [backlog.jsonl]
# watch / stop / report
tail -f benchmarks/results/autoresearch/night-*/runner.log
touch  benchmarks/results/autoresearch/night-*/STOP     # clean stop after current cell
cat    benchmarks/results/autoresearch/night-*/morning_report.md
```

## Runner properties (validated Phase 0)
- **Serial** (one GPU job at a time, always). **Crash-safe** append-only `results.jsonl` (fsync per
  cell). **Idempotent resume** by `cell_id` (kill/resume test: 6/30 killed → resumed → 30 unique, 0
  dupes). **Watchdog**: sentinel dense spot every N cells (drift > day-floor `NAAR_DRIFT_FRAC` →
  PAUSE), macmon on a time cadence, freq gate, RAM refuse. **Auto-zoom**: top-K tiles → valid
  neighbours until dry. **Stops only** on queue+zoom+backlog exhausted, watchdog STOP, deadline, or
  `STOP` file. Every stop leaves a legible reason; degraded numbers are never written as clean.

## Phase-0 findings that shape the search (read at source)
- Dense tiles `MFA_V6_NAX_{BQ,BK,WM,D_SUBTILE}` are **runtime knob-driven** (getenv per dispatch,
  `mfa_v6_nax_primitive.cpp:354-356`), no rebuild. Valid domain: `BK %32==0`, `BQ %(WM*16)==0`,
  else **loud-fail** (Rule 8). Tiles are a **pure perf axis** (cos = 1.0 across configs).
- Public `flash_attention` routes **D=64 dense → SDPA**, so D64 tiles are inert via the public path
  → the runner forces NAX via `_ext.v6_nax_forward(...,force_v6nax=True)`.
- Live (unconfirmed, Stage-1) signals: D128/N4096 `32·32·2` screens ~5–7% under the default;
  D64/N4096 `32·32·2` ~1.50 ms vs default-NAX 2.06 ms (still > SDPA 1.35 ms — a routing question).
- Codegen axes (simdgroup 2×2, Morton walk, buffering, barriers) are **not** env-swept — they need a
  source-generator variant + recompile (mapped in `census/`, not implemented in Phase 0).
