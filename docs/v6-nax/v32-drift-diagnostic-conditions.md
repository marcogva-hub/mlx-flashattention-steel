# Phase A.0 — Measurement conditions (v2.31.0 vs Phase 0 vs today)

**Sprint:** v2.32.0 drift diagnostic
**Date:** 2026-05-06 18:10
**Branch:** `experiment/v32-drift-diagnostic` @ `224d039`

## Today's system state

```
macOS:           26.5 (Tahoe), build 25F5068a
Hardware:        M5 Max (`applegpu_g17s`)
Uptime:          6h 06min (boot at ~2026-05-06 12:04)
Memory free:     ~67 GB (4.14M pages × 16 KB)
Compressor:      0 pages (no swap pressure)
Top CPU:         claude (51%), sysmond (20%), Activity Monitor (9%)
Background GPU:  Unknown (didn't powermetric — Phase A.2)
```

## Critical finding — Metal PSO cache moved on macOS 26

The original Phase A.1 plan prescribed clearing
`~/Library/Caches/com.apple.metal/*`. **That path is empty and obsolete on
macOS 26.5**. The actual Python-process Metal PSO cache is at:

```
/var/folders/c2/pwjb45v12rl4tf2k56vvh_300000gn/C/org.python.python/com.apple.metal/
```

(per-application, under the user's `DARWIN_USER_CACHE_DIR`, partitioned by
`bundle-id` — for our `.venv/bin/python` that's `org.python.python`).

Current cache contents — **155 MB total**:

| File | Size | mtime | Meaning |
|---|---:|---|---|
| `16777235_355/functions.data` | 38 MB | 2026-05-06 11:49 | MSL functions, current bucket — built today post-boot |
| `16777235_355/functions.list` | 230 KB | 2026-05-06 13:49 | Cache index — touched during Phase 0 bench |
| `16777235_355/functions1.data` | 1.3 MB | 2026-05-05 11:56 | Older/overflow bucket — yesterday's leftover |
| `16777235_355/functions1.list` | 10 KB | 2026-05-06 09:34 | Older index — touched at v2.31.0 release time |
| `32024/libraries.data` | 118 MB | 2026-05-06 11:49 | MSL pipeline libraries — built today post-boot |
| `32024/libraries.list` | 166 KB | 2026-05-06 13:49 | Library index — touched during Phase 0 bench |
| `32024/libraries1.data` | 3.2 MB | 2026-05-05 01:54 | Older library bucket |
| `32024/libraries1.list` | 4.3 KB | 2026-05-06 09:34 | Older library index — v2.31.0 release time |

Two-bucket structure (`*.data` current + `*1.data` overflow) is how macOS
manages cache rotation — when `*.data` exceeds threshold, older entries
roll into `*1.data`. The `*.list` files are write-through indices, so
their mtime tracks every cache lookup.

## Reconstructed timeline

| Time | Event | Cache state |
|---|---|---|
| 2026-05-05 01:54 | older `libraries1.data` mtime | accumulated cache from prior session |
| 2026-05-05 11:56 | older `functions1.data` mtime | accumulated cache, 11h old |
| **2026-05-06 02:48** | **Sprint 4 / v2.31.0 bench (commit `0efe95f`)** | warm-from-yesterday cache |
| 2026-05-06 09:34 | v2.31.0 release commit (`e0e581f`) — pip install/first import bumped `functions1.list` mtime | partial refresh after reinstall |
| ~2026-05-06 12:04 | OS reboot (uptime says 6h06m at 18:10) | in-memory cleared; disk persists |
| 2026-05-06 11:49 | `functions.data` and `libraries.data` mtimes — main cache rebuilt | **post-reboot warmup** |
| **2026-05-06 13:24-13:52** | **Phase 0 cross-session bench** (commit `224d039`) | 155 MB warm cache loaded |
| 2026-05-06 14:20 | Phase 0 commit | — |
| 2026-05-06 18:10 | Now (Phase A.0) | 155 MB cache still present |

## Variable differences identified

### v2.31.0 bench (02:48) vs Phase 0 (13:24)

| Variable | v2.31.0 bench | Phase 0 bench | Delta |
|---|---|---|---|
| Cache state | Yesterday's cache + post-release `functions1.list` refresh | 155 MB main cache built post-boot | **DIFFERENT** — different on-disk content + likely different in-memory state |
| Reboot between | — | Yes, at ~12:04 | **YES** |
| macOS version | 26.5 25F5068a (today; haven't checked v2.31.0 time, but unlikely changed mid-day) | 26.5 25F5068a | likely same |
| Hardware | M5 Max | M5 Max | same |
| iStat performance fan | per Marco's protocol | confirmed by Marco at Phase 0 launch | same (assumed) |
| Cooldown protocol | wrapper 90/60/30s same | wrapper 90/60/30s same | same |
| Background load | unknown | unknown (claude was running) | UNKNOWN — could differ |

### Inferred causal candidates

The largest controlled-variable difference is the **reboot + cache state
between v2.31.0 measurement and Phase 0**. Two paths flow from this:

1. **PSO cache theory** (audit's primary hypothesis): the v2.31.0
   measurement happened on a "yesterday's cache" state. The Phase 0
   measurement happened on a "fresh-post-boot cache" state. If the
   cache state materially affects bench timing, this could account
   for the drift. Phase A.1 tests this directly by clearing the cache
   to force cold, then re-measuring.

2. **GPU power-state theory**: a 6h uptime vs an unknown-uptime v2.31.0
   measurement could mean different baseline P-states. M5's Dynamic
   Caching may also play in. Phase A.2 (powermetrics) discriminates.

### Variables NOT reconstructable

- Whether v2.31.0 bench was preceded by sustained GPU activity (GPU
  pre-warmed) or idle (GPU cold).
- Hardware uptime at v2.31.0 measurement time.
- Background-load profile (browsers, Spotlight, etc.) at v2.31.0 time.
- Whether any kernel module was loaded/unloaded between sessions.

## Implications for Phase A.1

The corrected Phase A.1 procedure must:

1. Clear the **actual** Python Metal cache path (not the `~/Library/Caches/com.apple.metal/` no-op).
2. Verify the clear took effect (cache size = 0).
3. Run a cold legacy bench on D=128 shapes (SeedVR2-small + CogVideoX + SeedVR2-large).
4. Run a warm legacy bench immediately after.
5. Compare to v2.31.0 (275.6/3669/6780 ms) and Phase 0 (167.75/2344/3982 ms).

Hypothesis to test:

| Outcome | Interpretation |
|---|---|
| Cold ≈ v2.31.0 (slow), Warm ≈ Phase 0 (fast) | **PSO cache CONFIRMED** — v2.31.0 bench was effectively cold |
| Both fast (≈ Phase 0) | **PSO cache REJECTED** — cache state is not the driver |
| Both slow (≈ v2.31.0) | Cache *is* the driver but Phase 0 already paid for warmup; legacy is genuinely slower today than yesterday — different culprit (e.g., macOS update via `softwareupdated` background, GPU driver state) |
| Cold and Warm both intermediate | Mixed contribution — partial PSO + other factor |

## Files

- `outputs/diagnostic/system-state-now.txt` — raw system state (sw_vers, uptime, vm_stat, top, memory_pressure)
- `outputs/diagnostic/git-history.txt` — commits & timestamps
