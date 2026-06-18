# mlx-mfa perf claims registry

**Canonical** registry of perf claims documented in user-facing docs
(README, TRAINING_QUICKSTART, CHANGELOG, release notes).  Per
`CLAUDE_V6_NAX.md` §Z (public API path testing rule), every claim
here must be reproducible via the public API path with the documented
env vars.

This doc is the human-readable counterpart to
`tests/test_release_notes_perf_claims.py` (the executable §Z
enforcement).  Every entry here has a corresponding `PERF_CLAIMS`
list entry in that file.  Pre-tag `/mlx-mfa-release-audit` Check 4
verifies all active claims are still REACHABLE.

---

## Active claims (as of v2.58.1)

### NAX forward tile autotune (M5 Max, `research/nax-autotune-m5`, 2026-06-18)

Autoresearch over the V6 NAX **forward** kernel (`v6_nax_forward`, the production-dispatched
`NAAttentionKernel`). Phase 0 (runtime knob-map) found the ONLY live tuning axis is the tile
triple `NAX_BQ/NAX_BK/NAX_WM` (jointly constrained `BQ%(WM*16)==0`, `BK%32==0`); `BLOCK_R/
BLOCK_C/EXEC_SG` are vestigial (fed the F-3-removed simdgroup path), and `UNROLL_MODE/
RELAXED_PRECISION/BLOCK_D/FORCE_DYNAMIC_K/MAX_THREADS` are NO-OP for the NAX path. All TFLOPS
≤49 (gate 51.8 ✓); 3-replicate median; correctness err ≤9e-6 vs independent fp32.

- **D=64 default `BK` 64→32 (tuned this branch).** Robustly faster across 6 shapes × {fp16,bf16}
  (−2% to −15%; larger N gains most). Absolute ms, fp16, B=1 H=8: N=8192 **2.97→2.50 ms**
  (now ~parity with SDPA 2.45 ms, was ~1.21× slower); N=16384 **10.81→9.44 ms** (SDPA 9.19);
  bf16 N=8192 **2.93→2.49 ms**. Benefits the D=64 NAX paths that run (the V6 backward-recompute,
  bare `_ext`); does NOT re-open the F-2 D=64-dense→SDPA auto-route (NAX BK=32 reaches ~parity,
  not a robust win over SDPA). Reproduce: `bench/nax_autotune.py baseline`. Lock:
  `tests/test_nax_tuned_defaults_lock.py` (config fingerprint BK=32).
- **D=128 default BQ=64/BK=32/WM=4 confirmed optimal — NO change.** The nearest valid alternative
  (32/32/2) regressed **+31.9%** at N=2048 (within noise elsewhere) → not a robust win. Baseline
  (fp16, B=1 H=8): N=4096 NAX **1.65 ms** vs SDPA 1.65 ms (parity); N=8192 NAX **5.73 ms** vs SDPA
  5.98 ms (NAX faster). This is the production `backend="auto"` dense-D=128 route (F-2).

### NAX backward tile autotune (M5 Max, `research/nax-backward-autotune-m5`, 2026-06-18)

Autoresearch over the default-on **D=64 native backward** (the split dQ kernel,
`MFAV6NAXBwdQuery`). The backward has its OWN dedicated per-kernel tile knobs
(`MFA_V6BWD_*` dQ, `MFA_V6BWDV_*`/`MFA_V6BWDK_*` split dV/dK, `MFA_V6BWDKV_*`/`MFA_V6BWDF_*`
fused) — re-proved live from scratch (Lesson #14), NOT the forward's `MFA_V6_NAX_*`. All
TFLOPS ≤ 51.8 effective gate (measured 7.5–20.1). Gradients verified vs an independent fp32
vjp oracle (Lesson #11). Both fp16 and bf16 (the VSR training dtype); routing confirmed clean
(bf16 reaches the native backward, symmetric with fp16).

- **D=64 dQ default `BK` 64→32 (tuned this branch).** The dQ kernel had inherited the stale
  pre-tune `BK=64` (its comment said "matches forward defaults", but the forward had moved D=64
  to `BK=32`). `BK 64→32` is robustly faster on the FULL backward (dQ+dV+dK) across 6 shapes ×
  {fp16,bf16}: −4 % to −14 %, grad-IDENTICAL (BK is perf-only for dQ). fp16 N=8192 causal
  **16.3→14.3 ms**; fp16 N=2048 non-causal **1.87→1.34 ms**; bf16 N=4096 causal **4.46→4.01 ms**.
  dQ D=64 is now `32/32/2` (= the forward's tuned D=64 config). The D=64 native backward already
  beats Apple SDPA-vjp **1.4–2.7×** (N=8192 fp16 causal 14.3 vs 43.3 ms). Reproduce:
  `mx.grad(flash_attention(q,k,v,causal=...))` at D=64, N≥2048 (default-on). Lock:
  `tests/test_nax_backward_tuned_defaults_lock.py` (dQ tile fingerprint BK=32 + fp32-oracle grad).
- **D=128 backward NOT tuned — architectural floor confirmed.** The default D=128 backward is
  Apple SDPA-vjp (the native D=128 backward is opt-in via `MFA_ENABLE_V6_BACKWARD=1` and measured
  slower — 0.46–0.58× per the v2.50 carve-out record). Measured at the default: D=128 N=2048
  **2.71 ms**, N=4096 **9.94 ms** (= SDPA-vjp, the production choice). No headroom; not a tuning target.

v2.39.1 outcome α: H1 register pressure root-caused + fixed.  Fused
kernel default `BK` lowered 32 → 16 in Sprint v2.39.1 investigation.
Auto-default routes D=64 V6NAX backward to fused-BK16 (modest improvement
over v2.38.1 split path).  D=128 unchanged (carve-out hard-gated to D=64).

| Claim ID | Version intro | Description | Reproduction |
|---|---|---|---|
| `v2.39.1_d64_qL4096_fused_bk16_engages_via_auto` | v2.39.1 | D=64 qL=4096 V6NAX backward 2.00× vs SDPA-vjp (was 1.91× in v2.38.1, wall-time -2.9%) | `MFA_ENABLE_V6_BACKWARD=1` + `mx.grad(flash_attention(..., backend="auto"))` |
| `v2.39.1_d64_qL8192_fused_bk16_engages_via_auto` | v2.39.1 | D=64 qL=8192 V6NAX backward 1.95× vs SDPA-vjp (was 1.87×, wall-time -1.4%) | same |
| `v2.39.1_d64_qL16384_fused_bk16_engages_via_auto` | v2.39.1 | D=64 qL=16384 V6NAX backward 1.72× (3-session median; fresh-machine 1.89×; thermal drift across back-to-back sessions) | same |

Full investigation evidence + skill invocations log:
`.doc-archive/docs/v6-nax/v39-1-investigation-synthesis.md`.



| Claim ID | Version intro | Description | Env required | Public-API reproduction | Latest /mlx-mfa-perf-audit verdict |
|---|---|---|---|---|---|
| `v2.38.1_d64_qL4096_v6nax_dvec_engages_via_auto` | v2.38.1 | D=64 qL=4096 V6NAX backward **1.91×** vs SDPA-vjp (was 1.75× v2.37.3 under identical conditions; D_vec precompute saves 2 in-kernel rowsums) | `MFA_ENABLE_V6_BACKWARD=1` | `mx.grad(mlx_mfa.flash_attention(q,k,v))` with `(B=2,H=8,qL=4096,D=64) fp16 non-causal` | REACHABLE (2026-05-13, /mlx-mfa-perf-audit verified, 3-session median 1.91× variance 1.03) |
| `v2.38.1_d64_qL8192_v6nax_dvec_engages_via_auto` | v2.38.1 | D=64 qL=8192 V6NAX backward **1.87×** vs SDPA-vjp (was 1.79× v2.37.3) | `MFA_ENABLE_V6_BACKWARD=1` | Same with `qL=8192` | REACHABLE (3-session median 1.87× variance 1.10) |
| `v2.38.1_d64_qL16384_v6nax_dvec_engages_via_auto` | v2.38.1 | D=64 qL=16384 V6NAX backward **1.80×** vs SDPA-vjp (was 1.75× v2.37.3) | `MFA_ENABLE_V6_BACKWARD=1` | Same with `qL=16384` | REACHABLE (3-session median 1.80× variance 1.10) |
| `v2.37.2_d64_qL4096_v6nax_engages_via_auto` | v2.37.2 | D=64 qL=4096 V6NAX backward 1.82× faster than SDPA-vjp (preserved historical baseline; superseded by v2.38.1 1.91× under identical bench conditions) | `MFA_ENABLE_V6_BACKWARD=1` | `mx.grad(mlx_mfa.flash_attention(q,k,v))` with `q,k,v` of shape `(1,4,4096,64) fp16` | REACHABLE (2026-05-13, audit v2.37.x) |
| `v2.37.2_d64_qL8192_v6nax_engages_via_auto` | v2.37.2 | D=64 qL=8192 V6NAX backward 1.81× faster than SDPA-vjp (preserved historical baseline) | `MFA_ENABLE_V6_BACKWARD=1` | Same as above with `qL=8192` | REACHABLE (2026-05-13) |
| `v2.50.0_prompt5b_d128_qL8192_auto_engages_v6nax_split_at_parity` | v2.50 Prompt 5b | D=128 qL=8192 V6NAX backward engages via AUTO (split kernels, Sprint B v2.40.0-internal outcome γ) at parity with SDPA-vjp (~RMSE 2e-5).  Coverage extension; no speedup claim | `MFA_ENABLE_V6_BACKWARD=1` | `mx.grad(mlx_mfa.flash_attention(q,k,v))` with `(1,4,8192,128) fp16` | REACHABLE (parity engagement, v2.50 Prompt 5b Section D) |
| `ii12_d64_qL8192_default_on_v6nax` | II-12 (2026-06) | D=64 backward (causal + non-causal) default-on via the clean V6NAX split kernel, 1.7-2.7x vs SDPA-vjp | env unset | B=1 H=4 qL=8192 D=64 fp16 | REACHABLE (default) |
| `ii12_d64_qL8192_optout_sdpa` | II-12 (2026-06) | `MFA_DISABLE_V6_BACKWARD=1` restores SDPA-vjp bit-exactly | opt-out env | Same shape | REACHABLE (opt-out) |
| `ii9_conv3d_t16_64x64_c128_fp16_mpp_default` | II-9 (2026-06; row added III-1) | conv3d via the MPP convolution2d primitive, default-on: 2.3-2.5x vs the materialized-im2col path (T8/T16 64x64 C128) | env unset (opt-out `MFA_DISABLE_CONV3D_MPP=1`) | `install_hooks(); mx.conv3d(x, w)` with x `(1,16,64,64,128)` w `(128,3,3,3,128)` fp16, pad (1,1,1) | REACHABLE (default; telemetry-verified) |
| `iii1_conv3d_t16_64x64_c128_bf16_mpp_default` | III-1 (2026-06, KD-7 lift) | bf16 conv3d via MPP: 1.4-2.7x vs the pre-lift public bf16 path (Apple mx.conv3d fallback) at the II-9 cells | env unset (opt-out `MFA_DISABLE_CONV3D_MPP=1`) | Same shapes in bf16 | REACHABLE (default; telemetry-verified) |
| `iii2_tq_paged_decode_step_default` | III-2 (2026-06; re-confirmed III-12b on 26.6; reframed III-12c) | **User-facing trade-off (the headline): TQ paged decode trades ~1.4-3x decode-step latency for a ~4-5x KV-cache memory reduction at cos ~0.96, vs fp16 dense decode** (`step()` `0.75 ms vs 0.33 ms` @S=16K; KV `32 MB → ~6.5 MB` @S=8K). Opt-in (`TurboQuantPagedInferenceContext`), not auto-routed — the user chooses the trade-off. _Secondary / internal-perf history (NOT the user choice — the fused kernel is gone, so it is not a selectable baseline): the gather/dequant+SDPA path is 6.5-23x faster than the fused TQ attend kernel it replaced (`0.75 ms vs 16.8 ms` @S=16K)._ Lesson #15 + III-12c: lead with the actionable denominator (fp16 dense), not the biggest-number one. | env unset (opt-out `MFA_DISABLE_TQ_DECODE_SDPA=1`) | `TurboQuantPagedInferenceContext.step(q, k, v)` N_q=1, B=1 Hq=32 Hkv=8 D=128 tq3b; reproduce: `benchmarks/methodology/iii12b_tq_claim_26.6_run{1,2}.log` (script `tq_claim.py`) | REACHABLE (default; kernel-cache-verified) |

### Internal claims (v2.39.2-internal — below-public-floor coverage)

v2.39.2-internal lowered the V6NAX backward carve-out floor from `qL≥4096`
to `qL≥2048` after the v2.39.1 BK=16 fused kernel achieved parity with
SDPA-vjp at qL=2048 (3-session variance 1.004; see
`.doc-archive/docs/v6-nax/v39-2-internal-decisions.md`).  These claims preserve §Z
coverage for the new floor boundary.  Internal-mode only — not promoted
to user-facing release notes because no speedup at qL=2048 (parity-only
engagement preserves contract honesty per env-var opt-in).

| Claim ID | Version intro | Description | Env required | Public-API reproduction | Latest /mlx-mfa-perf-audit verdict |
|---|---|---|---|---|---|
| `v2.39.2_internal_d64_qL2048_auto_engages_v6nax_at_parity` | v2.39.2-internal | D=64 qL=2048 V6NAX backward engages via AUTO at parity with SDPA-vjp (3-session variance 1.004; no speedup but contract-honest engagement) | `MFA_ENABLE_V6_BACKWARD=1` | `mx.grad(mlx_mfa.flash_attention(q,k,v))` with `(1,4,2048,64) fp16 non-causal` | REACHABLE (parity engagement, v2.39.2-internal) |
| `v2.39.2_internal_d64_qL1024_auto_falls_back_to_sdpa` | v2.39.2-internal | D=64 qL=1024: below v2.39.2-internal carve-out floor (regresses ~15% vs SDPA-vjp empirically); carve-out correctly does not engage | `MFA_ENABLE_V6_BACKWARD=1` (still sdpa_fallback) | Same shape with `qL=1024` | REACHABLE (correct fallback below new floor) |

### Reproduce snippet template (per §Z)

```python
import os
os.environ["MFA_ENABLE_V6_BACKWARD"] = "1"
import time, statistics
import mlx.core as mx
import mlx_mfa

mx.random.seed(0)
q = mx.random.normal((1, 4, 8192, 64), dtype=mx.float16) * 0.1
k = mx.random.normal((1, 4, 8192, 64), dtype=mx.float16) * 0.1
v = mx.random.normal((1, 4, 8192, 64), dtype=mx.float16) * 0.1
mx.synchronize()

def loss(q, k, v):
    return mlx_mfa.flash_attention(q, k, v).sum()  # default backend="auto"

grad_fn = mx.grad(loss, argnums=(0, 1, 2))
for _ in range(4):
    g = grad_fn(q, k, v); mx.synchronize()

ts = []
for _ in range(12):
    t0 = time.perf_counter()
    g = grad_fn(q, k, v); mx.synchronize()
    ts.append((time.perf_counter() - t0) * 1000)
print(f"V6NAX backward: {statistics.median(ts):.2f} ms")
# M5 Max, fp16: ~9.78-9.91 ms (vs SDPA-vjp ~17.67-18.10 ms = 1.81× faster)
```

---

## Retracted claims (historical record — DO NOT REINSTATE)

| Claim ID | Version intro | Version retracted | Reason |
|---|---|---|---|
| `v2.37.1_d64_qL2048_v6nax_wins_1.44x` | v2.37.1 | v2.37.3 | Overstated.  Current canonical-methodology bench shows 1.15× kernel-level / ~1.06× end-to-end win, within measurement noise.  v2.37.2 carve-out correctly does not engage at qL=2048.  See `.doc-archive/docs/v6-nax/v2.37.x-perf-claim-audit.md`. |

## Reclassified claims (kernel characterization, not user-facing)

These remain in research / methodology docs but were REMOVED from
user-facing release notes / README / TRAINING_QUICKSTART because the
AUTO path doesn't engage their measured kernel.

| Claim ID | Version intro | Reclassified in | Reason |
|---|---|---|---|
| `v2.37.0_d128_v6nax_22_24x_slower` | v2.37.0 | v2.37.3 | D=128 V6NAX backward 2.2-2.4× slower than SDPA-vjp at kernel level.  AUTO path correctly never engages D=128 V6NAX (architectural-floor research only).  Numbers reproducible via `backend="mfa"` forced path; not user-facing perf. |
| `v2.37.3_d128_qL8192_auto_falls_back_to_sdpa` | v2.37.3 | v2.50 Prompt 5b | Superseded by `v2.50.0_prompt5b_d128_qL8192_auto_engages_v6nax_split_at_parity`.  Sprint B v2.40.0-internal Phase C.1.b validated D=128 split kernels at parity with SDPA-vjp (RMSE ~2e-5); Prompt 5b Section D lifted the `_v6nax_backward_carveout` D=128 gate.  D=128 now ENGAGES at parity (not fallback). |
| `v2.37.3_d64_qL2048_auto_falls_back_to_sdpa` | v2.37.3 | v2.39.2-internal | Superseded by `v2.39.2_internal_d64_qL2048_auto_engages_v6nax_at_parity` (Internal claims table).  The v2.39.2-internal carve-out floor was lowered from `qL≥4096` → `qL≥2048` after BK=16 fused kernel achieved parity at qL=2048.  qL=2048 now ENGAGES at parity (not fallback).  Below-floor fallback coverage preserved by `v2.39.2_internal_d64_qL1024_auto_falls_back_to_sdpa`. |
| `v2.50.0_pattern6_v6nax_sparse_bwd_falsified_at_vsr` | v2.50 Prompt 5d | v2.50 Prompt 5d | EMPIRICAL FALSIFICATION record: V6NAX native sparse backward projected 10× at d=0.1; empirical bench at VSR shape (B=1 H=12 qL=4096 D=128 fp16) shows native is 0.09×-0.77× SDPA-vjp dense across all densities.  Apple SDPA NAX on M5+ is empirically optimal for sparse backward — Pattern #6 inversion catalogued in `.doc-archive/docs/v50/audit-framing-inversions.md`.  Production routing reverted to Prompt 5c hybrid.  Documented per §Z institutional discipline. |
| `v2.50.0_prompt5c_topk_bisection_auto_3_85x_phase3a` | v2.50 Prompt 5c | v2.50 Prompt 5e | Top-K bisection kernel (Architecture B) AUTO production default delivers 3.85× speedup over Phase 3a `mx.topk` at audit shape (42.91 ms → 11.15 ms).  Reclassified as documentation-grade: bench is documented but not executable via the §Z PERF_CLAIMS test harness (top-K is not `mx.grad`-routed; engagement detection via differential gradient comparison doesn't apply).  Reproduce via opt-out flag: `MFA_DISABLE_TOPK_BISECT=1` vs default.  See `.doc-archive/docs/v50/phase-3b-approach-5-decision.md` for full bench data. |

---

## How to add a new claim

Per `CLAUDE_V6_NAX.md` §Z + §AA mandatory blocking:

1. **Discovery** — bench shows "X× speedup" or "Y% faster"
2. **Invoke `/mlx-mfa-perf-audit`** with the claim's documented API
   call + env vars + shape regime.  Verdict must be REACHABLE.
3. **Add an entry to `tests/test_release_notes_perf_claims.py`**
   `PERF_CLAIMS` list with `id`, `env`, `shape`, `dtype`, `expected`,
   `documented_in`, `documented_perf_claim`.  Tests must pass.
4. **Add a row to this doc's "Active claims" table** with the
   `claim_id` matching step 3.
5. **CHANGELOG entry** for the release must include the Reproduce
   snippet (template above).
6. **Pre-tag `/mlx-mfa-release-audit` Check 4** verifies the test
   passes; doc audit verifies row presence here.

If a claim is later overstated (per re-bench under canonical
methodology) or reclassified (kernel-only, not user-facing), MOVE
it to the appropriate historical-record table — do NOT delete.
Audit trail preservation is the §Z institutional discipline.

---

## Cross-references

- `CLAUDE_V6_NAX.md` §Z (public API path testing rule)
- `CLAUDE_V6_NAX.md` §AA.2 (skill invocation evidence)
- `CLAUDE_V6_NAX.md` §AA.4 (pre-tag enforcement via /mlx-mfa-release-audit)
- `tests/test_release_notes_perf_claims.py` (executable enforcement)
- `.doc-archive/docs/v6-nax/v2.37.x-perf-claim-audit.md` (the audit that drove §Z creation)
- `.doc-archive/docs/skills/README.md` (/mlx-mfa-perf-audit skill)
- `~/.claude/skills/mlx-mfa-perf-audit/SKILL.md` (skill definition)
