# Sprint III-8 — RESOLUTION: `backend="mfa"` non-causal divergence

**Date:** 2026-06-15
**Executor:** Claude Opus 4.8 (1M context)
**Outcome:** **RESOLVED.** Root cause was the precompiled `async_v2.metallib`
(`simdgroup_async_copy` DMA, removed from the macOS-26 AIR runtime) being loaded
ahead of the JIT path for `SteelForwardV2` keys — **not** the JIT source the
five prior sprints (III-7 → III-8e) instrumented. Fix: gate the async fast-path
off on macOS 26+ in `csrc/shader_cache.mm`. Mechanism understood and fixed at
source (no route-around), per Marco's standing directive.

## Root cause (one line)
On macOS 26, `try_async_pipeline()` served a pipeline built from
`async_v2.metallib`; its broken `simdgroup_async_copy` DMA loads only
`~(qb+1)·BQ` keys per Q-tile → truncated non-causal output. Causal survives
(mask zeroes unloaded keys); default dispatch routes non-causal dense → SDPA, so
only `backend="mfa"` (expert) reaches it.

## Why it took five sprints (the meta-lesson, #14)
III-8e's uniform-P oracle correctly measured the *behavior* (q-dependent key
truncation) but **mis-attributed the location** to JIT codegen. The tell, missed:
"the `(qb+1)·BQ` signature matches NO obvious source line." A signature that fits
no source line is evidence the **source isn't running**. All register dumps,
differential reading, and oracle-bisection of `generate_steel_v2_source` were
inert — they instrumented a binary that does not execute on this machine.
Codified as lesson #14 (`audit-framing-inversions.md`): **confirm which binary
runs before debugging source** (sentinel write / dispatch-env toggle / enumerate
key loaders), a ≤10-min check that would have pre-empted the entire marathon.

## How it was confirmed (not guessed)
| Probe | Result | Inference |
|---|---|---|
| `MFA_DISABLE_ASYNC=1` differential | MAE ~0.12 → bit-exact vs fp32 | an env that bypasses source compilation fixes it ⇒ bug is in async pipeline selection, not source |
| sentinel-777 write in JIT source | never appears in output | JIT source ≠ running binary for that key |

## The fix
`csrc/shader_cache.mm::try_async_pipeline()` — after the `SteelForwardV2`-only
guard and the `MFA_DISABLE_ASYNC` check, return `nullptr` on macOS 26+:
```objc
if ([[NSProcessInfo processInfo] operatingSystemVersion].majorVersion >= 26)
  return nullptr;  // simdgroup_async_copy broken on macOS 26 → JIT path
```
All V2 dispatch then uses `generate_steel_v2_source` (per-lane device reads),
correct on macOS 26 all along.

## Validation ladder
| Gate | Criterion | Result |
|---|---|---|
| R.4 O-vs-fp32 | D∈{64,128}×{fp16,bf16}×causal/non-causal×N{32..4096}×B·H{4..16} vs fp32 SDPA | all correct; deterministic (maxdiff 0.0) |
| R.5 generalization | fix disables broken metallib for ALL its keys on macOS 26 (only served `SteelForwardV2`); full suite | **1728 passed, 2 skipped** |
| R.6 re-bench + promotion | JIT V2 fwd ~3–4× slower than Apple SDPA on M5+; async only reachable via `backend="mfa"` | correctness-only, **no perf impact, no promotion** (consistent with default→SDPA) |
| R.7 lock | `tests/test_iii8_backend_mfa_noncausal.py`: fp32-parity sweep + forced-single-pass non-causal known-answer (each Q-tile attends ALL keys) | 57 passed; closes v1.4.0 coverage illusion |

## ⚠ Separate pre-existing bug found (flagged, NOT fixed)
V2 **split-K** non-causal at **partial N** (N∈{127,160,191,224}, D=128) →
MAE ~7–16 vs fp32. **Not** caused by the async metallib (`try_async_pipeline`
only ever served single-pass; split-K always used JIT). Distinct defect in
split-K partial-N range/reduction handling. Non-default-reachable
(`backend="mfa"` + split-K occupancy heuristic + partial N). Lock scopes itself
to single-pass (`MFA_FORCE_SPLITK=0`) to stay precise. **Follow-up required.**

## Release disposition
Bundle the V2 non-causal correctness fix with the two III-7 `quantize_model`
fixes (already on master) into **v2.52.2** (Marco-gated). Correctness-only;
no perf claim, no API change.

## Skill invocations (§AA.2)
| Checkpoint | Skill | When | Outcome |
|---|---|---|---|
| Kernel debugging (multi-gate dispatch suspected) | sentinel-write / kernel-debugging.md | which-binary check | isolated async metallib as running binary (vs 5 sprints of source reading) |
| Independent-reference validation (lesson #11) | mlx-debug-forensics §11 | R.4 | fp32 SDPA oracle throughout |
| Perf discovery / no-promotion verdict | /mlx-mfa-bench-methodology | R.6 | sub-ms variance handled; no promotion (SDPA wins) |
| Pre-release gate | /mlx-mfa-release-audit | deferred to v2.52.2 tag (Marco-gated) | pending |
