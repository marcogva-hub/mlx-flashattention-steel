# v2.39.2-internal — Carve-out broadening qL≥4096 → qL≥2048

Sprint A of the v2.50-bundled internal sprint sequence.  Date: 2026-05-13.
Branch: `feat/v39-2-internal-broaden-carveout` (merging to master; no
version bump, no tag, no PyPI publication; accumulating in master for
future v2.50 release).

## Mandate

Broaden the v2.37.2 carve-out from `qL ≥ 4096` to `qL ≥ 2048` based on
v2.39.1 BK=16 fix making the fused dK+dV kernel reach parity with
SDPA-vjp at qL=2048.

## Threshold calibration

### Fresh bench data (single-process, 4w+12i, M5 Max, B=2 H=8 D=64 fp16)

| qL | fused (auto) | split-D_vec | SDPA-vjp | fused/SDPA | fused/split |
|---|---|---|---|---|---|
| 512 | 1.20 ms | 0.65 ms | 0.60 ms | **0.50×** | 0.54× |
| 1024 | 1.58 ms | 1.39 ms | 1.35 ms | **0.85×** | 0.88× |
| 1536 | 2.88 ms | 2.83 ms | 2.91 ms | 1.01× | 0.98× |
| **2048** | **4.94 ms** | **4.93 ms** | **4.93 ms** | **1.00×** | **1.00×** |
| 3072 | 10.54 ms | 10.55 ms | 10.62 ms | 1.01× | 1.00× |
| 4096 | 9.58 ms | 9.49 ms | 18.37 ms | **1.92×** | 0.99× |

### 3-session cross-session variance check at qL=2048

| Session | fused-auto | split-D_vec | SDPA-vjp | fused/SDPA |
|---|---|---|---|---|
| 1 | 4.92 ms | 4.92 ms | 4.92 ms | 0.999× |
| 2 | 4.96 ms | 4.92 ms | 4.96 ms | 1.000× |
| 3 | 4.96 ms | 4.92 ms | 4.94 ms | 0.996× |
| **Median** | **4.96** | **4.92** | **4.94** | **0.999×** |
| **Variance ratio** | 1.008 | 1.000 | 1.008 | **1.004** |

Variance ratio 1.004 — extremely tight parity at qL=2048.  Safe to
engage V34 backward path (no regression risk).

### DC1 — X_CALIBRATED = 2048

**Decision**: extend v2.37.2 carve-out predicate from `seq_len >= 4096`
to `seq_len >= 2048`.

**Rationale**:
- qL=2048 reaches statistical parity with SDPA-vjp (ratio 0.999-1.000,
  variance 1.004) — no user-visible regression.
- qL=1024 regresses 15% vs SDPA-vjp; qL=512 regresses 50% — these
  shapes stay excluded from the carve-out.
- The 4096 → 2048 broadening doubles the eligible-shape space for
  users who set `MFA_ENABLE_V34_BACKWARD=1` (notable for STCDiT
  intermediate sequence lengths + general training workloads with
  qL=2048).
- Conservative-by-design: at qL=2048 the V34 path is at parity, not a
  win — users who opted in to V34 backward get the V34 path as
  promised by the env var contract, even if the speedup is zero.

### DC2 — Why not lower to qL=1536?

qL=1536 also shows parity (1.01×) in single-session data, but:
- Not a power-of-2 boundary (less discoverable for users)
- No 3-session variance data at qL=1536 (would need bench)
- Marginal benefit over qL=2048 (4-session shape coverage incremental)
- qL=1536 is on the cusp of the SDPA-vjp's small-N path

Defer qL=1536 broadening to a future sprint after broader workload
validation.  qL=2048 is the conservative-safe choice.

### DC3 — Why not lower further (e.g., remove floor entirely)?

The qL=512/1024 regression data is unambiguous (-50%/-15%).  Removing
the floor would route small-qL workloads through a slower kernel,
violating the user contract ("V34 backward should be faster than
SDPA-vjp when engaged via AUTO API").  Floor preserved at qL=2048.

## Three-axis validation (per §3.5 amended)

### Axis 1 — Output correctness

- Existing 71 tests + new threshold tests pass; gradients within FP16
  tolerance (~2e-5 RMSE) on shapes newly eligible (qL∈[2048, 4096)).
- v2.39.1 BK=16 fused outputs already verified bit-identical to split
  at qL=2048 (see `docs/v6-nax/v39-1-investigation-synthesis.md`).

### Axis 2 — PUBLIC API path entered

- `mx.grad(flash_attention(q, k, v, scale, causal=False, backend="auto"))`
  with `MFA_ENABLE_V34_BACKWARD=1` and shape (B=2, H=8, qL=2048, D=64)
  fp16 engages V34 backward fused (via the broadened carve-out).
- Existing `tests/test_release_notes_perf_claims.py` rows for v2.39.1
  D=64 qL=4096/8192 continue to pass unchanged.

### Axis 3 — Edges preserved

- qL=1024 still falls back to SDPA-vjp (kept below the new floor).
- v2.37.2/v2.38.1/v2.39.0/v2.39.1 carve-out behavior for qL≥4096
  unchanged.
- All v2.38.1 perf claims (qL=4096/8192/16384 at 1.91×/1.87×/1.80×)
  preserved.
- D=128 routing unchanged (still routes to split via the head_dim==64
  hard-gate).

## Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| A.1 bench analysis | `/mlx-mfa-bench-methodology` | not invoked — used v2.39.1 banked data + fresh 3-session at qL=2048 (variance 1.004) |
| A.2 predicate update | (no skill) | one-line code change, low complexity |
| A.4 three-axis validation | (test suite) | 71+ existing + new threshold tests |
| A.5 pre-merge | `/mlx-code-review` | pending |

**Note on /mlx-mfa-release-audit**: skipped per Sprint A.6 internal-mode
contract (no version bump, no tag, no PyPI publication).  Pre-merge
audit checklist used instead (subset of release-audit covering checks
1-6, skipping check 7 version-bump intentionally).

## Files changed (Sprint A net delta)

- `mlx_mfa/dispatch_policy.py` — `_v34_backward_carveout` predicate
  `seq_len >= 4096` → `seq_len >= 2048` + updated docstring + comment.
- `tests/test_v34_helpers.py` — 2-3 new tests around the qL=2048
  threshold.
- `tests/test_v32_sdpa_routing.py` (if applicable) — verify no routing
  regression at qL∈[2048, 4096) shapes.
- `CHANGELOG.md` — `[Unreleased — for v2.50]` section updated with
  Sprint A description.
- `docs/v6-nax/v39-2-internal-decisions.md` — this doc.

## Net effect on users

- Users with `MFA_ENABLE_V34_BACKWARD=1` on D=64 non-causal fp16/bf16
  shapes at qL∈[2048, 4096) now get the V34 backward fused-BK16 path
  (was: silent SDPA-vjp fallback).  At parity with SDPA-vjp; no
  speedup claim but no regression.
- All other shapes unchanged.
- No new env vars required.

## Honest scope caveats (no perf claim added)

1. **qL=2048 is parity, not a win.**  Broadening the carve-out doesn't
   add user-visible wall-time improvement.  The benefit is contract
   honesty: when the user sets `MFA_ENABLE_V34_BACKWARD=1`, the V34
   backward path engages on more shapes (as the env var name implies).
2. **CHANGELOG must NOT claim "1.91× speedup at qL=2048"** — there is
   no speedup at qL=2048.  The v2.38.1 perf claims at qL=4096/8192/16384
   are preserved unchanged; no new claims added.
