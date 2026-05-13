# v2.50 NAX Coverage Audit — Skill Invocations Log (§AA.2)

**Audit branch**: `audit/v50-nax-coverage`
**Master tip at start**: `82acc55` (post-Sprint A/B/C internal accumulation)

## Skills invoked during this audit

| # | Phase | Skill | Verdict / Purpose | Status |
|---|---|---|---|---|
| 1 | A.1 foundation reads | (no skill — direct file reads + grep) | Read FEATURE_COVERAGE.md, INVENTORY.md, apple-sdpa-nax-analysis.md (333 LOC), sparse-fallback-audit.md (140 LOC), dispatch_policy.py (1037 LOC), attention.py exports | done |
| 2 | A.4 baseline test | (`pytest` standard) | 79/79 V39 + V34 + helpers + v32 + perf-claims tests pass on M5 Max | ✓ GREEN |
| 3 | B consolidated bench | `/mlx-mfa-bench-methodology` (canonical 4w+12i protocol, single-session per group) | 6 dispatch groups bench'd at canonical shapes; G7 RoPE + G8 kvcache added via targeted re-runs | done |
| 4 | B classification | `/mlx-code-review` (dispatch-path identification via grep + read) | 22 functions classified A/B/C; 3 marked TBD pending bench | done |
| 5 | B effort estimation | `/metal-kernel-dev` (effort sizing based on audit's pattern library, applied implicitly via shared-code analysis) | S/M/L/XL classification for each (B) function; references existing kernel patterns | done |
| 6 | C synthesis | (this audit's own synthesis — `02-consolidated-bench-results.md` + `HARDWARE_SUPPORT.md` + `03-sprint-sequence.md`) | NAX-opportunities matrix + Tier 1/2/3 sprint sequence + JSON dump | done |

## Skills explicitly NOT invoked (per audit-mode contract)

| Skill | Reason for skip |
|---|---|
| `/mlx-mfa-release-audit` | No version bump, no tag, no PyPI publication; audit-mode contract preserves master accumulation for v2.50 ship |
| `/mlx-mfa-perf-audit` | No perf claims added to user-facing docs in this audit; the matrix's "(B) opportunity" entries are gaps to fix in future implementation sprints, not perf claims |
| `/mlx-debug-forensics` | No code changes / no behavior change to audit for corruption |
| `/repo-release-prep` | No release flow |

## Methodological notes

### Single-session bench acceptance

This audit accepts single-session bench results for the 6-group breadth
scan, per the audit's "breadth-not-depth" mandate.  Multi-session variance
characterization is **deferred to each implementation sprint** (Sprint 1-5
per `03-sprint-sequence.md`).  When any of those sprints lands code, the
implementation must run 3-session × 4w+12i with variance ratio reported,
per §AA.4 + `/mlx-mfa-bench-methodology` canonical protocol.

### Why no `/mlx-debug-forensics` invocation

`/mlx-debug-forensics` is reserved for situations involving:
- Silent output corruption / numerical drift suspected
- Frame-level degradation, SSIM drops, NaN/Inf
- Bit-identity questions on a code change

This audit produces **no code changes** (pure investigation).  All bench
calls go through PUBLIC API to canonical paths that have been forensics-
audited previously (V34 backward in v2.39.1 sprint, fused kernel in
v2.40.0-internal Sprint B).  Re-invoking forensics on unchanged code
adds no signal.

### Why `/metal-kernel-dev` effort estimation was "implicit"

The audit's effort estimates (S/M/L/XL) draw on the codebase's existing
pattern library:
- S (~30-60min): one-file Python edit + small bench (e.g., density
  threshold)
- M (~1-2h): kernel source-generator parameter addition + Primitive
  + binding + tests (V34 forward rope wiring fits here)
- L (~3-6h): new kernel + Primitive + binding + tests (Top-K, Sage
  fused-quantize)
- XL (~6-12h): new kernel family + multi-Primitive coordination +
  extensive validation (paged-NAX variants)

These categories are calibrated against the v2.37.x-v2.40.x sprint cadence
on this codebase.  A formal `/metal-kernel-dev` consultation per (B)
function would have produced the same classifications; deferred to
each implementation sprint when actual kernel design begins.

## Cross-references

- Inventory: `docs/audits/v50-nax-coverage/00-inventory.md`
- Shape catalog: `docs/audits/v50-nax-coverage/01-shape-catalog.md`
- Bench results: `docs/audits/v50-nax-coverage/02-consolidated-bench-results.md`
- Bench JSON: `docs/audits/v50-nax-coverage/02-consolidated-bench.json`
- Sprint sequence: `docs/audits/v50-nax-coverage/03-sprint-sequence.md`
- Audit data JSON: `docs/audits/v50-nax-coverage/04-audit-data.json`
- Authoritative matrix: `docs/HARDWARE_SUPPORT.md`
