# Round-3 audit remediation — enumeration artifacts (durable, tracked)

The completeness proofs for the systematic A–E (+CX-06) remediation of the
round-3 dual no-priors audit (branch `fix/audit-remediation`, base `dafdbce`,
v2.61.0, M5 Max / macOS 26.6 / MLX 0.31.2). Relocated here from gitignored
`devnotes/` so the round-4 re-audit and future sessions can verify completeness.

**Tracked but sdist-excluded** (same model as `release-gate/`): present on the
git tree, never shipped to PyPI users. Enforced by
`tests/test_publish_surface_guard.py` (0 `audit/` members in the built sdist) and
allowed by the tracked-tree journal guard.

| volet | artifact | enumeration deliverable |
|---|---|---|
| A | `engagement_proof_audit.md` | every `_dispatch_trace` terminal site + every engagement test, classified |
| C | `validation_matrix.md` | every entry point × edge-input class → raises-cleanly / silent-wrong / N-A |
| B | `buffer_read_audit.md` | every kernel device-buffer read → guarded-correctly / mis-ordered / unguarded |
| E | `claims_reconciliation.md` | every perf claim, routing claim, knob vs runtime/source ground truth |
| CX-06 | `stream_param_surgery.md` | every binding exposing a stream/device param → functional? removed? |

Each artifact ends with its bite-proven validation. Commit history on
`fix/audit-remediation` (volets A→C→B→D→E→CX-06) carries the full record; volet
D's gate-enforcement report remains in `devnotes/release_gate_enforcement.md`
(git history + the gitignored snapshot).
