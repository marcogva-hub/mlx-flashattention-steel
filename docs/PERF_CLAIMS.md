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

## Active claims (as of v2.39.1)

v2.39.1 outcome α: H1 register pressure root-caused + fixed.  Fused
kernel default `BK` lowered 32 → 16 in Sprint v2.39.1 investigation.
Auto-default routes D=64 V34 backward to fused-BK16 (modest improvement
over v2.38.1 split path).  D=128 unchanged (carve-out hard-gated to D=64).

| Claim ID | Version intro | Description | Reproduction |
|---|---|---|---|
| `v2.39.1_d64_qL4096_fused_bk16_engages_via_auto` | v2.39.1 | D=64 qL=4096 V34 backward 2.00× vs SDPA-vjp (was 1.91× in v2.38.1, wall-time -2.9%) | `MFA_ENABLE_V34_BACKWARD=1` + `mx.grad(flash_attention(..., backend="auto"))` |
| `v2.39.1_d64_qL8192_fused_bk16_engages_via_auto` | v2.39.1 | D=64 qL=8192 V34 backward 1.95× vs SDPA-vjp (was 1.87×, wall-time -1.4%) | same |
| `v2.39.1_d64_qL16384_fused_bk16_engages_via_auto` | v2.39.1 | D=64 qL=16384 V34 backward 1.72× (3-session median; fresh-machine 1.89×; thermal drift across back-to-back sessions) | same |

Full investigation evidence + skill invocations log:
`docs/v6-nax/v39-1-investigation-synthesis.md`.



| Claim ID | Version intro | Description | Env required | Public-API reproduction | Latest /mlx-mfa-perf-audit verdict |
|---|---|---|---|---|---|
| `v2.38.1_d64_qL4096_v34_dvec_engages_via_auto` | v2.38.1 | D=64 qL=4096 V34 backward **1.91×** vs SDPA-vjp (was 1.75× v2.37.3 under identical conditions; D_vec precompute saves 2 in-kernel rowsums) | `MFA_ENABLE_V34_BACKWARD=1` | `mx.grad(mlx_mfa.flash_attention(q,k,v))` with `(B=2,H=8,qL=4096,D=64) fp16 non-causal` | REACHABLE (2026-05-13, /mlx-mfa-perf-audit verified, 3-session median 1.91× variance 1.03) |
| `v2.38.1_d64_qL8192_v34_dvec_engages_via_auto` | v2.38.1 | D=64 qL=8192 V34 backward **1.87×** vs SDPA-vjp (was 1.79× v2.37.3) | `MFA_ENABLE_V34_BACKWARD=1` | Same with `qL=8192` | REACHABLE (3-session median 1.87× variance 1.10) |
| `v2.38.1_d64_qL16384_v34_dvec_engages_via_auto` | v2.38.1 | D=64 qL=16384 V34 backward **1.80×** vs SDPA-vjp (was 1.75× v2.37.3) | `MFA_ENABLE_V34_BACKWARD=1` | Same with `qL=16384` | REACHABLE (3-session median 1.80× variance 1.10) |
| `v2.37.2_d64_qL4096_v34_engages_via_auto` | v2.37.2 | D=64 qL=4096 V34 backward 1.82× faster than SDPA-vjp (preserved historical baseline; superseded by v2.38.1 1.91× under identical bench conditions) | `MFA_ENABLE_V34_BACKWARD=1` | `mx.grad(mlx_mfa.flash_attention(q,k,v))` with `q,k,v` of shape `(1,4,4096,64) fp16` | REACHABLE (2026-05-13, audit v2.37.x) |
| `v2.37.2_d64_qL8192_v34_engages_via_auto` | v2.37.2 | D=64 qL=8192 V34 backward 1.81× faster than SDPA-vjp (preserved historical baseline) | `MFA_ENABLE_V34_BACKWARD=1` | Same as above with `qL=8192` | REACHABLE (2026-05-13) |
| `v2.50.0_prompt5b_d128_qL8192_auto_engages_v34_split_at_parity` | v2.50 Prompt 5b | D=128 qL=8192 V34 backward engages via AUTO (split kernels, Sprint B v2.40.0-internal outcome γ) at parity with SDPA-vjp (~RMSE 2e-5).  Coverage extension; no speedup claim | `MFA_ENABLE_V34_BACKWARD=1` | `mx.grad(mlx_mfa.flash_attention(q,k,v))` with `(1,4,8192,128) fp16` | REACHABLE (parity engagement, v2.50 Prompt 5b Section D) |
| `v2.37.3_d64_qL8192_env_unset_no_v34` | v2.37.3 | Without `MFA_ENABLE_V34_BACKWARD=1`, V34 backward NEVER engages | env unset | Same shape, env clear | REACHABLE (correct fallback) |

### Internal claims (v2.39.2-internal — below-public-floor coverage)

v2.39.2-internal lowered the V34 backward carve-out floor from `qL≥4096`
to `qL≥2048` after the v2.39.1 BK=16 fused kernel achieved parity with
SDPA-vjp at qL=2048 (3-session variance 1.004; see
`docs/v6-nax/v39-2-internal-decisions.md`).  These claims preserve §Z
coverage for the new floor boundary.  Internal-mode only — not promoted
to user-facing release notes because no speedup at qL=2048 (parity-only
engagement preserves contract honesty per env-var opt-in).

| Claim ID | Version intro | Description | Env required | Public-API reproduction | Latest /mlx-mfa-perf-audit verdict |
|---|---|---|---|---|---|
| `v2.39.2_internal_d64_qL2048_auto_engages_v34_at_parity` | v2.39.2-internal | D=64 qL=2048 V34 backward engages via AUTO at parity with SDPA-vjp (3-session variance 1.004; no speedup but contract-honest engagement) | `MFA_ENABLE_V34_BACKWARD=1` | `mx.grad(mlx_mfa.flash_attention(q,k,v))` with `(1,4,2048,64) fp16 non-causal` | REACHABLE (parity engagement, v2.39.2-internal) |
| `v2.39.2_internal_d64_qL1024_auto_falls_back_to_sdpa` | v2.39.2-internal | D=64 qL=1024: below v2.39.2-internal carve-out floor (regresses ~15% vs SDPA-vjp empirically); carve-out correctly does not engage | `MFA_ENABLE_V34_BACKWARD=1` (still sdpa_fallback) | Same shape with `qL=1024` | REACHABLE (correct fallback below new floor) |

### Reproduce snippet template (per §Z)

```python
import os
os.environ["MFA_ENABLE_V34_BACKWARD"] = "1"
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
print(f"V34 backward: {statistics.median(ts):.2f} ms")
# M5 Max, fp16: ~9.78-9.91 ms (vs SDPA-vjp ~17.67-18.10 ms = 1.81× faster)
```

---

## Retracted claims (historical record — DO NOT REINSTATE)

| Claim ID | Version intro | Version retracted | Reason |
|---|---|---|---|
| `v2.37.1_d64_qL2048_v34_wins_1.44x` | v2.37.1 | v2.37.3 | Overstated.  Current canonical-methodology bench shows 1.15× kernel-level / ~1.06× end-to-end win, within measurement noise.  v2.37.2 carve-out correctly does not engage at qL=2048.  See `docs/v6-nax/v2.37.x-perf-claim-audit.md`. |

## Reclassified claims (kernel characterization, not user-facing)

These remain in research / methodology docs but were REMOVED from
user-facing release notes / README / TRAINING_QUICKSTART because the
AUTO path doesn't engage their measured kernel.

| Claim ID | Version intro | Reclassified in | Reason |
|---|---|---|---|
| `v2.37.0_d128_v34_22_24x_slower` | v2.37.0 | v2.37.3 | D=128 V34 backward 2.2-2.4× slower than SDPA-vjp at kernel level.  AUTO path correctly never engages D=128 V34 (architectural-floor research only).  Numbers reproducible via `backend="mfa"` forced path; not user-facing perf. |
| `v2.37.3_d128_qL8192_auto_falls_back_to_sdpa` | v2.37.3 | v2.50 Prompt 5b | Superseded by `v2.50.0_prompt5b_d128_qL8192_auto_engages_v34_split_at_parity`.  Sprint B v2.40.0-internal Phase C.1.b validated D=128 split kernels at parity with SDPA-vjp (RMSE ~2e-5); Prompt 5b Section D lifted the `_v34_backward_carveout` D=128 gate.  D=128 now ENGAGES at parity (not fallback). |
| `v2.37.3_d64_qL2048_auto_falls_back_to_sdpa` | v2.37.3 | v2.39.2-internal | Superseded by `v2.39.2_internal_d64_qL2048_auto_engages_v34_at_parity` (Internal claims table).  The v2.39.2-internal carve-out floor was lowered from `qL≥4096` → `qL≥2048` after BK=16 fused kernel achieved parity at qL=2048.  qL=2048 now ENGAGES at parity (not fallback).  Below-floor fallback coverage preserved by `v2.39.2_internal_d64_qL1024_auto_falls_back_to_sdpa`. |
| `v2.50.0_pattern6_v34_sparse_bwd_falsified_at_vsr` | v2.50 Prompt 5d | v2.50 Prompt 5d | EMPIRICAL FALSIFICATION record: V34 native sparse backward projected 10× at d=0.1; empirical bench at VSR shape (B=1 H=12 qL=4096 D=128 fp16) shows native is 0.09×-0.77× SDPA-vjp dense across all densities.  Apple SDPA NAX on M5+ is empirically optimal for sparse backward — Pattern #6 inversion catalogued in `docs/v50/audit-framing-inversions.md`.  Production routing reverted to Prompt 5c hybrid.  Documented per §Z institutional discipline. |
| `v2.50.0_prompt5c_topk_bisection_auto_3_85x_phase3a` | v2.50 Prompt 5c | v2.50 Prompt 5e | Top-K bisection kernel (Architecture B) AUTO production default delivers 3.85× speedup over Phase 3a `mx.topk` at audit shape (42.91 ms → 11.15 ms).  Reclassified as documentation-grade: bench is documented but not executable via the §Z PERF_CLAIMS test harness (top-K is not `mx.grad`-routed; engagement detection via differential gradient comparison doesn't apply).  Reproduce via opt-out flag: `MFA_DISABLE_TOPK_BISECT=1` vs default.  See `docs/v50/phase-3b-approach-5-decision.md` for full bench data. |

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
- `docs/v6-nax/v2.37.x-perf-claim-audit.md` (the audit that drove §Z creation)
- `docs/skills/README.md` (/mlx-mfa-perf-audit skill)
- `~/.claude/skills/mlx-mfa-perf-audit/SKILL.md` (skill definition)
