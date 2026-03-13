# Development Notes Archive (`devnotes/`)

This folder contains historical development artifacts (design notes, benchmark
JSON snapshots, validation notes) collected during iterative R&D branches.

Purpose:
- preserve experiment history and decision rationale;
- keep active docs (`README.md`, `docs/*`, `RESULTS.md`) focused on current
  project state;
- provide future restart context for newer hardware generations (for example,
  M5+ follow-up work).

## Important scope notes

- `devnotes/` is archival/informative material, not active API documentation.
- Many branch tracks were exploratory; not all experiments were promoted to the
  retained production path.
- Artifacts were included from locally available branch work only.
- Missing local artifacts were not invented.

## Track folders

- `native-backward-pass/`
- `d256-design-track/`
- `d512-decision-pass/`
- `sage-decode-productionization/`
- `runtime-unification/`
- `paged-shared-prefix-productionization/`
- `experimental-triage/`
- `final-profiling/`
- `paged-packed-varlen-unification/`
- `paged-continuous-batching/`
- `chunked-prefill/`
- `prefix-caching-automation/`
- `speculative-decode-runtime/`
- `hybrid-kv-cache-abstraction/`
- `hybrid-kv-cache-behavior/`
- `final-serving-completion/`
- `final-stabilization-release/`
- `archive/`

## Availability notes

- `final-profiling/` is currently a placeholder folder in this checkout; no
  dedicated profiling-note file was found under the previous `devnotes/` root to
  migrate into it during this pass.
- Additional branch-local artifacts may exist in other worktrees or clones;
  this archive includes what was locally available in this checkout.
- Branch scan details are recorded in `devnotes/branch_artifact_availability.md`.
