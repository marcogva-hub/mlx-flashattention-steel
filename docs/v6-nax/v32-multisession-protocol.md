# v2.32.0 multi-session bench protocol

**Status:** READY for execution. Approved by Marco at end of Phase A
diagnostic session (2026-05-06). Goal: characterize natural cross-session
variance and try to reproduce v2.31.0's slower legacy regime to confirm
or refute the "deep-overnight-idle" hypothesis.

## Goal

Determine whether v2.31.0's published perf claims (legacy at 275/3669/6780
ms on D=128 shapes) represent a reproducible regime, or a one-time
measurement under conditions that cannot be triggered on demand.

If reproducible → both v2.31.0 and Phase 0 numbers are valid for their
respective conditions; the publication needs a "depending on system state"
caveat.

If not reproducible → v2.31.0's perf claims are session-locked artifacts
and need a CHANGELOG/PyPI correction.

## Sessions to run

Run **at minimum 3 sessions** under 3 distinct conditions. Each session
takes ~30 min (5 shapes × A/B/A with 3 runs each, including 90s initial
+ 60s/30s cooldowns).

| Session | Conditions | Cache | Why |
|---|---|---|---|
| **S1** | Cold-boot morning, < 1h uptime | warm (don't clear) | Closest natural reproduction of v2.31.0's likely state — system fresh out of long idle |
| **S2** | Same morning, 30 min after S1 | warm (don't clear) | Test how quickly post-boot state stabilizes |
| **S3** | Afternoon, > 4h sustained activity | warm (don't clear) | Replicates Phase 0's regime — control |
| **S4 (optional)** | Cold-boot + cleared cache + S1 conditions | cleared | Combines all "cold" factors; orthogonal to PSO test from Phase A.1 |
| **S5 (optional)** | Late-night session after long evening idle | warm (don't clear) | Different time of day; tests time-of-day vs uptime causation |

## How to run one session

```bash
cd /Users/marcomarcelino/code/mlx-mfa-v2
git checkout experiment/v32-drift-diagnostic   # or whichever branch you're investigating from

# Run a session
.venv/bin/python bench/v32_multisession_capture.py \
    --label "S1-cold-boot-morning-2026-05-07" \
    [--clear-cache]   # optional — only for S4-style cold-cache conditions
```

Each invocation:
1. Captures conditions (sw_vers, uptime, Metal cache size + age range, time-of-day bucket)
2. (optional) Clears the macOS 26 Python Metal PSO cache at
   `/var/folders/c2/<user-hash>/C/org.python.python/com.apple.metal/`
3. 90s initial cooldown
4. A/B/A bench across all 5 production shapes with 60s inter-round /
   30s inter-shape cooldowns
5. Appends a record to `docs/v6-nax/v32-multisession-data.json`

## How to aggregate across sessions

```bash
.venv/bin/python bench/v32_multisession_aggregate.py
```

Prints per-shape median across sessions, range, variance, and flags any
session that reproduced v2.31.0's slow regime within ±10%.

## Pre-flight per session

Before running each session, confirm:

1. **iStat performance fan profile active** (Marco: manual)
2. **System otherwise idle** — close browsers, no recording, no other
   inference jobs
3. **Memory pressure normal** — `memory_pressure | head -5` shows no
   compression activity
4. **No competing Python processes** — `ps aux | grep python` shows
   nothing GPU-heavy

## Reference values (from v2.31.0 + Phase A)

For analytical comparison:

| Shape | v2.31.0 legacy | v2.31.0 V34 | Phase 0 + A legacy | Phase 0 + A V34 |
|---|---:|---:|---:|---:|
| FlashVSR-dense | 1.115 | 1.55 (regression) | ~0.93 | ~0.95 |
| LTX2-cross | 1.65 | 1.42 | 1.63 | 1.30 |
| SeedVR2-small | 275.6 | 170.9 | 167-185 | 184.7 |
| CogVideoX | 3669 | 2399 | 2333-2370 | 2162 |
| SeedVR2-large | 6780 | 4042.7 | 3886-3982 | 3878 |

## Decision rules after multi-session collection

After 3+ sessions complete:

- **If 0 sessions reproduce v2.31.0's slow regime** → conclude v2.31.0 was
  a non-reproducible artifact. Recommend v2.31.0 PyPI/CHANGELOG addendum.
- **If 1+ session reproduces it** → look at conditions. If correlated with
  cold-boot/morning, document as a regime-dependent perf characteristic.
  Both v2.31.0 and steady-state numbers are valid for their conditions.
- **If sessions split unpredictably** → there's an uncontrolled variable.
  Add more discrimination (more sessions, more conditions, possibly
  powermetrics with sudo).

## What this protocol does NOT do

- **Does not test V34 ramp-up specifically** — A.3.1 already rejected
  that hypothesis at the SeedVR2-small scale; if multi-session reveals
  ramp-up matters at other shapes, add focused tests then.
- **Does not vary macOS background activity** — that would need to be
  manually controlled per session; the protocol's "system idle" pre-flight
  is the assumed baseline.
- **Does not run thermal monitoring (powermetrics)** — needs sudo. Add
  manually if a session shows interesting drift you want to correlate
  with GPU freq.

## Output

`docs/v6-nax/v32-multisession-data.json` — append-only across sessions.
Each session record has:
- `session_label`, `timestamp_iso`, `time_of_day_bucket`
- `uptime_raw`, `sw_vers`
- `metal_cache_size_before`, `metal_cache_oldest_iso`,
  `metal_cache_newest_iso`, `metal_cache_file_count`
- `cache_cleared_pre_bench`
- `protocol` (runs/cooldowns)
- `bench[shape]` = list of 3 rounds with (mode, v6_runs_ms, v6_median_ms,
  sdpa_runs_ms, sdpa_median_ms, rmse, correctness_ok)

Aggregate analysis is reproducible from this file alone.
