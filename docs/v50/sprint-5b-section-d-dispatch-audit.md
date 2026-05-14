# Prompt 5b Section D — dispatch-chain audit (Pattern #5)

**Mandate**: lift `_v34_backward_carveout` D=128 hard-gate; broaden V34
backward to D=128 split kernels via AUTO path (preserves dV residual
fix from Prompt 4 Section B and Sprint B v2.40.0-internal D=128 split
empirical parity finding).

**Pattern #5 application**: identify ALL dispatch gates that produce
or consume V34 backward inputs for D=128 BEFORE the broadening fix.
Each gate is verified individually; the broadening either lifts the
gate or confirms it's already permissive.

## Multi-gate audit table

| # | Gate | Location | D=128 state pre-fix | Action |
|---|---|---|---|---|
| G1 | `_v34_backward_carveout` (flash_attention()-level carve-out) | `mlx_mfa/dispatch_policy.py:373-380` | **BLOCKING**: `head_dim == 64` | **LIFT** → `head_dim in (64, 128)` |
| G2 | `_v34_eligible` (closure-level second-line check) | `mlx_mfa/attention.py:3743` | PERMISSIVE: `head_dim in (64, 128)` | Verified; no change |
| G3 | `_v34_backward_vjp` routing | `mlx_mfa/attention.py:3827-3859` | PERMISSIVE: `head_dim not in (64, 128)` → ValueError for `fused`; AUTO routes D=128 → `split` per Sprint B outcome γ | Verified; no change |
| G4 | `MFAV6Forward::eval_gpu` causal-routing gate | `csrc/mfa_v6_nax_primitive.cpp:625` (post-Prompt 4 fix) | PERMISSIVE: causal+D=128 routes to V34 forward (natural-log lse) | Verified; no change |
| G5 | `MFAV6Backward::eval_gpu` D-handling | `csrc/mfa_v6_nax_primitive.cpp` (Sprint B v2.40.0-internal) | PERMISSIVE: D=128 split kernels emit correct gradients | Verified; no change |
| G6 | `compile_v34_backward_pipeline` cache keys | C++ Primitive cache | PERMISSIVE: D=128 cache keys constructed | Verified; no change |
| G7 | `get_supported_configs` (docs surface) | `mlx_mfa/__init__.py:325-335` | LISTED but only D=64 entries | **ADD** D=128 entries for discoverability |
| G8 | `flash_attention()` body backend-resolution short-circuit | `mlx_mfa/attention.py:468-502` | PERMISSIVE: passes head_dim through to G1 | Verified; no change |

## Sentinel-write verification (per `docs/methodology/kernel-debugging.md` §2)

Verified by inspection (no kernel changes required):

- **G4 verification**: Prompt 4 Section B already lifted the causal routing
  gate to enable V34 forward causal (natural-log lse) for D ∈ {64, 128}.
  D=128 causal forward currently produces natural-log lse; V34 backward
  D=128 split kernels consume it correctly per Sprint B v2.40.0-internal
  empirical parity (RMSE ~2e-5 vs SDPA-vjp).
- **G5 verification**: Sprint B v2.40.0-internal Phase C.1.b added
  D=128 split kernel source generators. `attention.py:3827-3859` AUTO
  routes D=128 → split (preserves v2.38.1 D=128 behavior). Fused for
  D=128 opens via opt-in `MFA_V34_BWD_KERNEL=fused`.

## LSE consistency check (per `docs/methodology/kernel-debugging.md` §4)

V34 forward D=128 emits natural-log lse (post-Prompt 4 multi-gate fix).
V34 backward D=128 split kernels expect natural-log lse. **Convention
match confirmed**.

## Risk register

| Risk | Mitigation |
|---|---|
| D=128 backward perf marginal vs SDPA-vjp (Sprint B finding: parity) | Documented in CHANGELOG: "coverage extension for D=128 training; perf gain not guaranteed" per outcome γ |
| Sprint 5 sparse D=128 needs D=128 dense backward as foundation | Section A depends on D ready; sequenced D → A explicitly |
| B.5 xfails `TestNativeBackwardRouting[128-2048, 128-4096]` were marked because forced-native bwd zeroed out tail blocks | These xfails will be re-investigated post-D-fix; if real fix lands, unmark; if still failing, preserve with accurate rationale |

## Conclusion

**Sole code change**: G1 broadening `head_dim == 64` → `head_dim in (64, 128)`.

**G7 collateral**: extend `get_supported_configs()` discovery entries for D=128.

Three-axis validation per Phase D.3 below.
