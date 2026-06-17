# Proposal — `CLAUDE_V6_NAX.md` Artifact #5

**Status:** DRAFT awaiting Marco approval
**Source:** `v32-drift-diagnostic-report.md` Phase B.3
**Action:** insert as new subsection in `CLAUDE_V6_NAX.md` §2 ("Méthodologie obligatoire — quatre artifacts déjà rencontrés"), updating the section title to reflect five artifacts. Renaming the section to "Méthodologie obligatoire — cinq artifacts" or similar.

## Diff

```markdown
### Artifact #5 — Cross-session perf claims publishable only after multi-condition repro

**Findings v2.31.0 → v2.32.0 cross-session diagnostic** (2026-05-06):
36-43% drift on legacy D=128 path (and inverse-direction drift on V6NAX
on SeedVR2-small) between v2.31.0 release-time bench (02:48 AM,
post-overnight-idle) and Phase 0 re-bench at 13:24 PM same day. PSO
cache hypothesis tested and **rejected** (cold-cache and warm-cache
benches produce identical timings within ±2%). GPU ramp-up / P-state
hypothesis tested and **rejected** (post-30s-aggressive-warmup bench
matches no-warmup bench within ±2%). The drift is **not a transient
manipulable artifact** — it's a steady-state offset between v2.31.0
measurement context and current sessions, beyond session-feasible
discrimination.

**Sub-rule 5a — Metal PSO cache path on macOS 26+**

Cache moved from `~/Library/Caches/com.apple.metal/` (empty/obsolete
on macOS 26) to per-application:

```
$DARWIN_USER_CACHE_DIR/<bundle-id>/com.apple.metal/
```

For our `.venv/bin/python` bench process: `org.python.python` bundle.
Resolve via `getconf DARWIN_USER_CACHE_DIR`. Any "clear cache" step in
diagnostic scripts must use this path, not the legacy `~/Library/Caches`
location.

**Sub-rule 5b — Marketing-grade benchmark publication discipline**

Before publishing perf claims to PyPI / CHANGELOG / README:

1. **Cross-session repro across 3+ sessions** with different times of
   day and different pre-bench states (cold-boot morning vs mid-day
   sustained vs after long idle).
2. **Document each session's conditions**: time of day, hardware
   uptime at bench start, Metal cache size before clear, macOS
   version (`sw_vers`), `GPU Active` percentage from `sudo powermetrics`
   in idle (must be < 5% — confirms no background GPU consumer).
3. **Use median of session medians**, not within-session statistics.
4. **Single-session bench results are STAGING data**, not publication
   data. Always pair with at least one re-bench in a different session.

**Why**: a single well-controlled within-session A/B/A is sufficient
for *engineering decisions* (e.g., dispatch choice within the project),
but insufficient for *external publication*. v2.31.0's perf claims —
which were based on a single A/B/A session — turned out to depend on
measurement-time conditions we cannot reproduce on demand.
```

## Where to insert

After the existing Artifact #4 (Env var change without kernel cache invalidation), as a new sub-section.

## Pre-conditions for merging this proposal

Marco approval, since `CLAUDE_V6_NAX.md` is project-level guardrail
documentation that affects all future agents (Claude + Codex) working
on V6 NAX. The change is non-trivial — it expands the methodology
discipline from "subprocess isolation within a session" to "multi-session
discipline for publication".
