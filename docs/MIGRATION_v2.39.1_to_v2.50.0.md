# Migration guide: v2.39.1 → v2.50.0

**Audience**: existing mlx-mfa v2.39.1 users upgrading to v2.50.0.
**TL;DR**: zero source changes required.  All v2.50 work landed
behind the existing public API; defaults route to the best-measured
path automatically.

---

## Breaking changes

**None.**  Internal-mode accumulation since v2.39.1 (Sprints A/B/C
internal + Prompts 1-5f) preserved every public API signature.  All
new behavior is either an automatic default upgrade (better perf,
same correctness) or opt-in via env vars.

---

## What v2.50.0 delivers

### Sprint 1 — Sparse density threshold + RoPE NAX (Prompts 1-2)

- **Sparse density auto-route**: `flash_attention_sparse` at low
  density (BT-block mask, M5+ NAX) automatically routes to the
  fused-NAX sparse forward (Sprint 1 ~6× speedup at audit shape).
- **RoPE unified fused NAX**: `flash_attention_rope_unified` discovers
  `mx.fast.rope` and dispatches through it instead of the STEEL RoPE
  helper (Sprint 2 ~4× speedup; Pattern #2 inversion catalogued).

### Sprint 3 — Top-K (Prompts 3-4)

- **Top-K bisection AUTO**: `flash_attention_topk` routes through
  Architecture B (bisection over score thresholds) by default at VSR
  shapes (3.85× over the dense baseline).  Opt-out via
  `MFA_DISABLE_TOPK_BISECT=1`.

### Sprint 4 + Section D — V34 forward + backward broadening

- **V34 forward causal D=64 + D=128**: NAX-direct kernels engaged
  automatically when `flash_attention(..., causal=True)` shape qualifies.
- **V34 backward causal D=64 + D=128**: production NAX-direct backward
  via `mx.grad(flash_attention(...))` on supported shapes.
  Replaces SDPA-vjp for the matching shape envelope.

### Section C + A (Prompts 5b-5d) — Sparse backward

- **`flash_attention_sparse` backward**: production routing per
  Pattern #6 empirical finding.  Defaults to dense SDPA-vjp with bias
  mask (Apple NAX optimization, fastest at all VSR densities).
  Opt-in V34 sparse hybrid via `MFA_ENABLE_V34_BACKWARD=1` for
  research / specific shape envelopes.
- **`flash_attention_sparse` forward (LSE-aware)**: returns
  `(O, L)` internally to support efficient sparse backward.  Public
  API unchanged (still returns just `O`).

### Prompt 5b Section C — `attn_bias` correctness

- Native Metal `attn_bias` kernel (modes 1/2) with causal fix per
  `qL_off` accounting for prefill+decode mismatch.

---

## New env vars (all opt-in / opt-out)

| Env var | Default | Purpose |
|---|---|---|
| `MFA_ENABLE_V34_BACKWARD` | unset (= 0) | Opt-in to V34 sparse backward hybrid orchestrator (research; production default is SDPA-vjp dense per Pattern #6 finding). |
| `MFA_V34_BWD_SPARSE_NATIVE` | unset (= 0) | Opt-in to full-native V34 sparse backward (all 4 gradients via native sparse kernels).  Typically slower than hybrid on M5+ per Pattern #6; research/benchmark only. |
| `MFA_DISABLE_TOPK_BISECT` | unset (= AUTO on) | Disable Top-K Architecture B bisection AUTO default.  Falls back to materialized weights path. |
| `MFA_DISABLE_TOPK_NATIVE` | unset | Disable the native Top-K Metal kernel path.  Falls back to Python composition. |
| `MFA_DISABLE_ROPE_NAX` | unset | Disable `mx.fast.rope` auto-discovery in `flash_attention_rope_unified`.  Routes through legacy STEEL RoPE helper. |

---

## Deprecations

> **Superseded (v2.56.0):** `MFA_FORCE_NATIVE_BWD` was REMOVED in v2.56.0 (the env var
> is now inert), and `MFA_ENABLE_V34_BACKWARD` is default-on for D=64 causal. The notes
> below are accurate for the v2.39.1→v2.50.0 migration; see CHANGELOG [2.56.0] for the
> current state.

### `MFA_FORCE_NATIVE_BWD=1` → emits `DeprecationWarning` (target removal v2.51+)

This env var forces routing through **legacy STEEL backward kernels**
which have a known correctness bug at D=128 N≥2048 (KD-5: zeroed output
blocks for query rows ≥ 1024).  V34 backward NAX-direct (production
default since Section D Prompt 5b) is unaffected and is the recommended
path for all backward use cases.

**Action required**: if you have any pipeline setting
`MFA_FORCE_NATIVE_BWD=1`, remove that setting.  The default routing
already gives you V34 backward where appropriate, and SDPA-vjp dense
elsewhere.

---

## Known-debt closure (KD-1..KD-5)

| ID | Severity | v2.50.0 disposition |
|---|---|---|
| KD-1 | HIGH | **FIXED** Phase A — V34 backward sparse mask shape conversion + C++ shape validation |
| KD-2 | MEDIUM | **FIXED** Phase B — forward recompute eliminated via custom_function `outputs` parameter (~1.33ms saving at VSR shape d=0.1) |
| KD-3 | LOW | **FIXED** Phase C — explicit `elif head_dim == 128` + defensive `else: raise` |
| KD-4 | LOW | **FIXED** (Prompt 5e Phase 1 fix + Prompt 5f Phase D regression coverage) |
| KD-5 | research-only | **DEPRECATED** (env var emits `DeprecationWarning`; target removal v2.51+) |

See `docs/v50/known-debt-v2.50.md` for full disposition details and
links to test coverage.

---

## Performance summary (VSR shapes, M5 Max)

Cumulative gains relative to v2.39.1 baseline:

| Operation | Baseline (v2.39.1) | v2.50.0 | Speedup |
|---|---|---|---|
| Sparse forward d≈0.02 | (Sprint 1 baseline) | 6× | 6× |
| RoPE unified forward (D=128) | STEEL helper | mx.fast.rope dispatch | 4.07× |
| Top-K @ k=64 / S=4096 (VSR) | dense baseline | Architecture B bisection AUTO | 3.85× |
| V34 forward causal D=64 qL=4096 | SDPA fallback | V34 NAX-direct | 1.82× |
| V34 backward causal D=64 qL=4096 | SDPA-vjp | V34 NAX-direct | 1.81× |
| V34 sparse backward d=0.1 (Phase B) | Prompt 5d snapshot | KD-2 fix | -1.33ms |

See `docs/PERF_CLAIMS.md` for the canonical perf-claim table with
verification test references.

---

## Upgrade procedure

```bash
.venv/bin/pip install --upgrade mlx-mfa==2.50.0
```

Or via wheel:

```bash
.venv/bin/pip install dist/mlx_mfa-2.50.0-*.whl
```

No code changes required.  Optionally:

1. **Audit your pipeline for `MFA_FORCE_NATIVE_BWD=1`** and remove it
   (see Deprecations above).
2. **Consider enabling `MFA_ENABLE_V34_BACKWARD=1`** if your sparse
   training workload matches the documented shape envelope (D ∈ {64,
   128}, qL ≥ 2048, fp16/bf16, M5+ hardware).  Currently opt-in
   pending broader perf validation.
3. **Verify your tests still pass** — v2.50.0 preserves 1274 tests
   green relative to v2.39.1's 920 (the additional ~354 tests are new
   coverage from v2.40-v2.50 sprint work).
