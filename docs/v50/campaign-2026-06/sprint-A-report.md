# Campaign 2026-06 — Sprint A report: systematic cache-key correctness audit

**Date**: 2026-06-12 · **Status**: COMPLETE · **Commits**: `5f8bdf4` (fixes),
`4937a8b` (tests), + cache-audit docs

## Headline

**5 new cache-key-invariant findings beyond the 2026-05 four** (C1/C2/C5/C7),
confirming the sprint's premise that four was not the complete set.  One is
CRITICAL (`V6Key.cfg_axis_flags` uint8_t truncation), and fixing it produced
a **discovery cascade**: the `MFA_V6_MATMUL_EXEC_SG` experiment knob turned
out to be a Pattern #8-style ghost — a silent no-op since its introduction
(its key bits truncated → every override aliased to the default pipeline),
and statically ILLEGAL to compile on current MetalPerformancePrimitives
headers once the key was fixed.  The knob is removed.

Phase A.5 ships permanent enforcement: a static parser test asserting every
field of all 11 C++ key structs participates in both `operator==` and the
hash — verified to FAIL on the pre-review V6Key, locking the C1/C5/C6
omission classes out of future code.

## Test-suite truth table

| | Before | After |
|---|---|---|
| passed | 1346 | **1366** (+20: 9 behavioral + 11 invariant) |
| xfailed / xpassed / flakes | 0 / 0 / 0 | 0 / 0 / **0 across 3 consecutive runs** |

## Findings & fixes

| ID | Severity | Site | Finding | Fix |
|---|---|---|---|---|
| A-1 | **CRITICAL** | V6Key (mfa_v6_nax_primitive.cpp) | `cfg_axis_flags` uint8_t truncated bits 8-9 (`MFA_V6_MAX_THREADS` buckets 0x100/0x180/0x200) — distinct pipeline configs aliased to one key. Pre-dates the 2026-05 fix (old `<<24` packing overflowed identically) | uint16_t field + hash + cast |
| A-1b | (cascade) | same | `MFA_V6_MATMUL_EXEC_SG` (bits 10-11, v2.30 experiment): silent no-op ghost pre-fix; statically illegal MPP source post-fix (`matmul2d` cooperative-tensor patterns require single-SG scope) | knob REMOVED (substitution + encoding) with mechanism comments |
| A-2 | DEFENSIVE | MFASteelBwdDQ::is_equivalent | `has_block_mask` missing (selects different compiled kernel). Severity refined: dense=7 vs sparse=8 inputs — CSE can't conflate differing arity today | field added |
| A-3 | DEFENSIVE | MFASteelBwdDKV::is_equivalent | same | field added |
| A-4 | **HIGH** | MFAPagedVarlenTQForward::is_equivalent | `tq_wht_enabled` missing — changes numerical output with IDENTICAL inputs → mx.compile CSE conflates WHT-on/off nodes | field added |
| A-5 | MEDIUM | dispatch_policy._load_custom_table + decision cache | `MLX_MFA_DISPATCH_TABLE` is a DOCUMENTED runtime override but was frozen at first read (process-lifetime flag) AND absent from the decision-cache key | reload-on-path-change + keyed |
| A-8 | LOW | conv_nax legacy Python path | embedded matmul2d source hardcodes `device half` casts; bf16 inputs bitwise type-punned → silently wrong values | loud ValueError (Rule 8); fp16-only matches the C++ production path (KD-7) |

## Pre-review failure verification (validation item 6)

Old `csrc/` + `dispatch_policy.py` checked out (`aa5741c`), rebuilt, new
tests run:
- `test_table_reload_on_path_change` **FAILS** on pre-review code ✅ (A-5)
- `test_key_struct_fields_in_eq_and_hash[V6Key]` **FAILS** on the
  pre-review V6Key ✅ (catches the original C5/C6 state)
- Honest nuance: the RUNTIME axis-flags test passes on old code — the
  max-threads alias changes pipeline occupancy, not numerics, so output
  comparison alone cannot detect it.  The STATIC invariant test is the
  load-bearing guard for that class (both ship).
- C1 (scale) pre-review behavioral repro requires old attention.py + old
  csrc simultaneously (the current C7 scale gate redirects non-default
  scales before the old key is exercised).  C1's mechanism was verified at
  fix time by source reading (scale baked via DOT_SCALE #define, absent
  from all 9 keys); the behavioral test discriminates on current paths and
  the static test guards the omission class generically.  [verified/deduced
  labeled accordingly]

## Perf (no-regression gate)

Headline paths post-fix: output diff vs SDPA = **0.0** at all 4 probed
shapes; dispatch ratios 0.98–1.01× (added env-key lookups are noise-level;
D=64/4096 improved vs the review's snapshot).

## Invariant enforcement shipped (Phase A.5)

- `tests/test_campaign_2026_06_sprint_a_key_invariants.py`: static parser
  over all 11 key structs (incl. KernelKey's out-of-line ==/hash in
  shader_cache.mm) — every declared field must appear in == AND hash.
  Runs in CI as plain pytest (0.01s).
- Larger options evaluated: a `CacheKey` base-class refactor (repo-wide
  surface change → **Marco-gated candidate**, escalated not deferred);
  debug-mode recompute-on-hit assertion (requires hooks into 11 hot caches
  — cost/benefit poor vs the static guard + behavioral tests; declined
  with rationale).

## Investigated and declined (evidence attached)

- `_verbose` import-time bake (logging only, zero dispatch effect).
- `MFA_FORCE_SPLITK` removal from the decision key (kept defensively —
  one dict lookup vs silent-staleness risk if a Python read appears).
- conv_nax `_KERNEL_CACHE` dtype key component: defensively correct as-is
  (source hardcodes half; key separates dtypes; the REAL bug was the
  missing input guard → fixed as A-8).

## Candidates surfaced for Sprint C

1. Gate-#9 automation harness (generator-constant introspection script) —
   partially anticipated by the Phase A.5 static test; full
   dispatch-vs-generator value diffing still open.
2. V3/V4/V5 M5 re-bench (carried from Sprint B).
3. `MFA_FORCE_NATIVE_BWD` reconsideration (Marco-gated, carried).
4. CacheKey base-class refactor (Marco-gated, new).
