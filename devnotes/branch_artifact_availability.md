# Branch Artifact Availability (Freeze Prep)

Date: 2026-03-13

This note records branch-hygiene checks performed during final cleanup.

## What was done

- Inspected local branch list to identify tracks with potential archival notes.
- Archived/moved locally available `notes/*` artifacts into track folders under
  `devnotes/`.
- Did **not** merge non-retained branches.
- Did **not** perform branch deletion/surgery in this pass.

## Local branch tracks observed

- `codex/native-backward-winning-shapes`
- `codex/d256-design-track`
- `codex/d256-design-track-post-bwd`
- `codex/d512-decision-pass`
- `codex/sage-decode-productionization`
- `codex/runtime-unification-pass`
- `codex/paged-shared-prefix-productionization`
- `codex/experimental-triage-aot`
- `codex/end-to-end-profiling-pass`
- `codex/paged-packed-varlen-unification`
- `codex/paged-continuous-batching`
- `codex/chunked-prefill`
- `codex/prefix-caching-automation`
- `codex/speculative-decode-runtime`
- `codex/hybrid-kv-cache-abstraction`
- `codex/hybrid-kv-cache-behavior`
- `codex/final-serving-completion`

## Notes

- `devnotes/final-profiling/` remains a placeholder in this checkout because no
  dedicated `notes/*` artifact matching that track was present locally during
  migration.
- Additional artifacts may exist in other clones/worktrees; this archive covers
  this local checkout only.
