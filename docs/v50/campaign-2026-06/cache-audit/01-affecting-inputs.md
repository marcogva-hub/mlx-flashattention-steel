# Sprint A Phase A.1/A.2 — per-site derivation summary + collision analysis

Three parallel sub-agent audits (C++ pipeline keys / is_equivalent / Python),
each deriving the affecting-input set from kernel source + generator reads,
then diffing against the actual key.  Detailed agent outputs preserved in
the session record; consolidated verdicts:

## DIFFs found (fixed this sprint)

| ID | Site | Missing/colliding | Verdict |
|---|---|---|---|
| A-1 | V6Key.cfg_axis_flags | uint8_t truncates bits 8-9 (MAX_THREADS buckets) and 10-11 (EXEC_SG) | CRITICAL — widened to uint16_t; EXEC_SG knob removed (statically illegal on current MPP — discovered BY the fix) |
| A-2 | MFASteelBwdDQ::is_equivalent | has_block_mask | DEFENSIVE (arity blocks today's CSE) — added |
| A-3 | MFASteelBwdDKV::is_equivalent | has_block_mask | DEFENSIVE — added |
| A-4 | MFAPagedVarlenTQForward::is_equivalent | tq_wht_enabled | HIGH (identical inputs, different output) — added |
| A-5 | dispatch decision cache + _load_custom_table | MLX_MFA_DISPATCH_TABLE (documented runtime override frozen at first read + absent from key) | MEDIUM — reload-on-path-change + keyed |
| A-8 | conv_nax legacy path | (kernel bug class, not key): `device half` cast accepts bf16 type-pun | LOW — loud ValueError |

## Collision analysis (Phase A.2)

- V6Key: post-2026-05 fix, no bit-packing remains EXCEPT axis_flags
  (a true flags field).  Max accumulated value pre-Sprint-A: 0xFC00-ish
  (bits 0-11); field was 8-bit → overflow.  Post-fix: bits 0-9 in a
  16-bit field; EXEC_SG bits retired.  Max value 0x3FF < 2^16. PROVEN.
- KernelKey: no packed fields; 18 scalar fields, FNV-1a hash over all.
  PROVEN by the Phase A.5 static invariant test.
- 9 V34 keys: scalar fields only. PROVEN by the same test.
- Remaining `<<` constructs in csrc: hash combiners only (not key
  encodings) — verified non-aliasing by the == completeness guarantee
  (hash collisions are perf-only when == is complete).

## Sites verified CLEAN (no DIFF)

- All 7 Python lru factories: every closure-affecting input is a factory
  arg; all env steering reads are LIVE inside closures (verified line-level
  for _make_mfa_custom: MFA_ENABLE_V34_BACKWARD, MFA_V34_BWD_KERNEL,
  MFA_V34BWD_USE_FUSED, MFA_V34BWD_WM; MFA_FORCE_NATIVE_BWD was also a
  live read here until v2.56.0 removed the knob).
- STEEL forward/backward scale: runtime params-struct field, NOT baked
  in source → correctly absent from KernelKey.
- MFAEnvConfig invalidate() interaction: V2/V3/V5 block overrides flow
  into computed BQ/BK which ARE keyed; force_gen flows into is_m3_plus
  which IS keyed; no_padding is frozen-static (immune to invalidate, by
  design).  Invalidation-safe.
- 22 of 25 is_equivalent sites complete (3 fixed); MFAttention compares
  all 13 Params fields.
- turboquant centroids, _M5_PLUS_CACHE, mlx_lm globals, lcsa_nax,
  compile_metallib, masks, integrations: clean (rationales in agent
  reports).

## INFO-grade (recorded, not fixed)

- `dispatch_policy._verbose` baked at import — logging only, zero
  dispatch effect.  Changing it would alter the documented monkeypatch
  semantics for no correctness gain.
- `MFA_FORCE_SPLITK` in the decision-cache key is defensively redundant
  (should_use_mfa doesn't read it) — kept with an explanatory comment;
  removal saves one dict lookup but reintroduces silent-staleness risk
  if a future Python-side read appears.
