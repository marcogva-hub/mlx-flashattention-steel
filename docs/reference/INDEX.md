# mlx-mfa Documentation Index

The tracked tree carries **current-state documentation only**. The campaign journal
(phase reports, sprint decisions, design docs, diagnostics, the audit ledger, devnotes)
is retained off the tracked tree — see **Provenance** below.

## Published (ships in the sdist — the published artifact on PyPI)
- [`README.md`](../../README.md) — overview, install, quick start, current perf, known issues
- [`CHANGELOG.md`](../../CHANGELOG.md) — version history
- `LICENSE`, `LICENSE-DRAWTHINGS`, `THIRD_PARTY_LICENSES`

## Root current-state / institutional
- [`RESULTS.md`](../../RESULTS.md) — verified-M5 perf summary (full tables → `BENCHMARKS.md`)
- [`ENV_VARS.md`](../../ENV_VARS.md) — environment-variable reference
- [`NAMING.md`](../../NAMING.md) — V6/NAX nomenclature glossary
- [`CLAUDE.md`](../../CLAUDE.md), [`CLAUDE_V6_NAX.md`](../../CLAUDE_V6_NAX.md) — agent/institutional rules

## Current-state reference (this directory, `docs/reference/`)

### Verified runtime + per-kernel (audit-locked)
- [`dispatch-map.md`](dispatch-map.md) — which kernel actually runs per entry × input (locked by `tests/test_dispatch_map_lock.py`)
- [`sparse-family-spec.md`](sparse-family-spec.md), [`dense-steel-family-spec.md`](dense-steel-family-spec.md), [`backward-family-spec.md`](backward-family-spec.md), [`b4-family-spec.md`](b4-family-spec.md) — per-kernel specs (fp32/oracle-locked)
- [`doc-claim-lock-map.md`](doc-claim-lock-map.md) — every load-bearing claim → its executable lock

### Guides & references
- [`API_MANUAL.md`](API_MANUAL.md) — public API
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system architecture
- [`FEATURE_COVERAGE.md`](FEATURE_COVERAGE.md) — feature matrix
- [`INVENTORY.md`](INVENTORY.md) — export inventory
- [`SERVING_GUIDE.md`](SERVING_GUIDE.md) — serving/runtime
- [`TRAINING_QUICKSTART.md`](TRAINING_QUICKSTART.md) — training quick start
- [`HARDWARE_SUPPORT.md`](HARDWARE_SUPPORT.md) — M1–M5 support matrix
- [`HOOK_TELEMETRY.md`](HOOK_TELEMETRY.md) — auto-hook telemetry
- [`PERF_CLAIMS.md`](PERF_CLAIMS.md) — perf-claim registry (synced by `tests/test_perf_claims_doc_sync.py`)
- [`RELEASE_PHILOSOPHY.md`](RELEASE_PHILOSOPHY.md) — auto-default principle
- [`BENCHMARKS.md`](BENCHMARKS.md) — full benchmark tables
- [`MIGRATION_v2.39.1_to_v2.50.0.md`](MIGRATION_v2.39.1_to_v2.50.0.md), [`MIGRATION_v2.50.0_to_v2.50.1.md`](MIGRATION_v2.50.0_to_v2.50.1.md) — upgrade notes

## Provenance — the journal
The full campaign journal is **retained but off the public tracked tree**: it lives in
**git history** (every file at its prior path) and in the gitignored **`.doc-archive/`**
snapshot. It is intentionally excluded from both the published sdist and the public repo
surface — enforced by `tests/test_publish_surface_guard.py` (the built sdist's
publication surface **and** the tracked-tree allowlist). Links throughout the current-state docs that point into `.doc-archive/...`
are provenance pointers into that archive.
