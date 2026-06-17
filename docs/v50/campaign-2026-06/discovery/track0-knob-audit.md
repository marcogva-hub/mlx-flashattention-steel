# Track 0 — Ghost-knob / dead-env-var sweep (2026-06-12)

83 knobs enumerated and classified (read-site trace + mechanism derivation;
full per-knob table preserved in the sweep agent record, dispositions
implemented in commit `docs(track0)`).

## Status summary

| Status | Count | Items |
|---|---|---|
| LIVE | 79 | dispatch gates, V2/V3/V5/V6/V6NAX tile overrides (all flow into cache keys — Sprint A verified), Python dispatch knobs, diagnostics |
| GHOST-DEAD (removed) | 1 | `MFA_V6_MATMUL_EXEC_SG` (Sprint A A-1b: no-op since v2.30 via key truncation; statically illegal on current MPP) |
| GHOST-ALIASED | 1 | `MFA_TOPK_BISECT` — never read by any code path (comment-only); ENV_VARS row corrected |
| STALE-DOCUMENTED | 2 | `MFA_REQUIRE_MSL4` (source sentinel, not an env var — docs corrected); `MFA_V6_NAX_DISABLE_ALIGN` (documented, never wired — row removed; re-add only if implemented) |
| UNDOCUMENTED-LIVE | 10 | `MFA_V6BWD_BQ/BK`, `MFA_V6BWDKV_*`, `MFA_V6BWDF_*`, `MFA_V6_SENTINEL_FILL`, `MFA_V6_DUMP_SOURCE`, `MFA_V6BWD_DUMP_SOURCE` — added to ENV_VARS.md |

## Interaction notes surfaced (documented in ENV_VARS.md)

- `MFA_V6_EXEC_SG` has no effect on the V6NAX path (`v6nax_WM` overrides).
- `MFA_V6_BYPASS_TGP=0` is a no-op when single-Otile auto-fires.
- `MFA_V6_FORCE_DYNAMIC_K` is live but fragile (MPP static_asserts may
  reject dynamic_length with cooperative-left operands at some configs).

## Historical benches invalidated

`docs/v6-nax/v2-30-thermal-rebench.md` EXEC_SG sweep rows measured the
SG=1 pipeline in EVERY configuration (the knob was a ghost when those
numbers were taken).  Invalidation note added to the doc.  The
MAX_THREADS dimensions of the same sweeps remain valid.  No downstream
TUNING DECISION consumed the EXEC_SG sweep (the knob never entered any
default config) — no dispatch re-validation required beyond the note.
