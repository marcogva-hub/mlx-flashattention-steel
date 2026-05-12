# Matched-workload-family warmup — decisions (DM1-DM9)

## DM1 — Warmup kernel family: `sparse_attention_nax` (V2)

**Decision**: replace `mx.matmul(256, 256, FP16)` with a small
`sparse_attention_nax` call using the SAME V2 kernel that is measured.

**Rationale**:
- Keeps shader pipeline cache hot for V2 instantiations — no cold
  recompile/relink penalty on the first measured dispatch.
- Threadgroup dispatch policy (SG sizing, occupancy) stays in steady
  state for the kernel family under test.
- Power state holds (small dispatch every < 100ms — `downclock-threshold-data.json`).

**Anti-pattern avoided**: `mx.matmul` competed for cluster L2 (v2.36.0
regression). A different kernel family changes both occupancy state
and cache pattern at the same time.

[VERIFIED] via diagnostic doc analysis. [HIGH] confidence.

## DM2 — Warmup shape: B=1 H=4 qL=kL=2048 D=64 BT=16 density=0.10

**Decision**: warmup workload uses different shape than ANY measured
shape in the 7-shape Sprint B set.

| Measured shapes | Warmup shape | Why different |
|---|---|---|
| D=128 (all 7 shapes) | **D=64** | Different shader instantiation → no Q_smem/K_smem cache-line aliasing |
| qL=kL ∈ {4096, 8192, 16384} | **qL=kL=2048** | 2-8× smaller, still satisfies mask-size constraint |
| BT=32 (all 7 shapes) | **BT=16** | Different tile size → distinct kernel instantiation |
| H ∈ {4, 8, 12} | H=4 | Smallest grid → minimum threadgroup contention |
| density ∈ {0.01, 0.03, 0.07, 0.12, 0.24} | density=0.10 | Mid-range, exercises sparse path |

**Initial design** picked qL=kL=512, but MLX inlines buffers < 4096 bytes
into constant address space while the JIT kernel emits device-qualified
pointers. A 2D bool mask at qL=kL=512 BT=32 is only 16×16 = 256 bytes,
which fails the kernel's `mask total bytes >= 4096` precondition.
Bumping to qL=kL=2048 BT=16 gives a 128×128 = 16 KB mask, comfortably
above the threshold while keeping working set smaller than ANY measured
shape.

**Working set estimate** (warmup): Q/K/V ≈ 3 × (1 × 4 × 2048 × 64 × 2B)
= **3 MB**. Active per dispatch is one BT=16 tile: Q_tile + K_tile +
V_tile ≈ 3 × (16 × 64 × 2B) = **6 KB** — fits in per-core L1 (well
below the M5 Max ~192 KB private L1 per core). The full 3 MB resident
set goes to cluster L2 but that is **a different region** from the
D=128 measured kernels' working set (different shader, different smem
tile shapes, different stride patterns → distinct cache lines).

**Working set estimate** (measured, mid_seq8k): Q_tile + K_tile + V_tile
at BT=32 D=128 ≈ 3 × (32 × 128 × 2B) = **24 KB** active per dispatch
— still local L1, but mask + bias arrays add ~32 KB per shape across
the entire run, and prefetch streams pull more.

**Critical isolation**: D=64 vs D=128 means the JIT-emitted Metal
shader source string differs (different unroll factors, different
register allocation, different smem layout). They occupy distinct
shader cache entries, not just different invocations of the same
kernel — so pipeline-state pollution risk is bounded.

[DEDUCED] — empirical confirmation in the bench results.
[HIGH] confidence on isolation; [MEDIUM] on perf neutrality.

## DM3 — Warmup gap: 50ms

**Decision**: same as v2.36.0 — 50ms gap between warmup dispatches
during cooldowns.

**Rationale**: empirical anchor `downclock-threshold-data.json` shows
< 100ms idle holds power state. 50ms gives 2× safety margin without
elevating duty cycle materially. The dispatch itself is ~30-100µs for
the small sparse shape, so duty cycle ≈ 0.2% (negligible thermal).

[VERIFIED] anchor data. [HIGH].

## DM4 — Smoke-gate AND inter-shape warmup share inputs

**Decision**: single allocation of warmup Q/K/V/mask/bias, reused for
all warmup dispatches (priming + initial cooldown + 21 inter-cooldowns).

**Rationale**:
- One mlx allocation → stable device buffer, no realloc churn.
- Stable warmup signal (no random reshuffling between cooldowns).
- Inputs eval'd once + retained → mx.synchronize() per dispatch is
  pure-kernel timing, no I/O dominance.

[HIGH].

## DM5 — A/B/A pattern unchanged (V2 → SDPA+bias → V2)

**Decision**: preserve A/B/A pattern from v2.36.0 protocol. 5 runs per
direction. ABA drift % computed per shape.

**Rationale**: A/B/A surfaces drift caused by progression through the
session (thermals, OS bg work, memory fragmentation). Replacing the
warmup mechanism does not invalidate this design — drift detection is
warmup-independent.

[HIGH].

## DM6 — §4 cooldown durations preserved (180s / 60s / 90s)

**Decision**: initial = 180s, inter-shape = 60s, inter-round = 90s,
same as v2.36.0 protocol.

**Rationale**: cooldown DURATION is the §4-strict protocol surface;
this sprint only swaps the cooldown FILLER. Apples-to-apples
comparison with v2.36.0 baseline requires same duration.

[HIGH].

## DM7 — 3 session subprocess-isolated runs (M1 / M2 / M3)

**Decision**: same as v2.36.0 — three subprocess-isolated sessions
across distinct subprocess instances. Cross-session range % is the
canonical variance metric.

**Rationale**:
- Single-session variance can mask per-session systematic drift
  (e.g., first session always hotter).
- Variance flags (CONFIDENT < 10%, BOUNDARY 10-20%, HIGH > 20%) are
  defined over **cross-session ratio range**, not within-session
  range. Three sessions is the minimum for a credible range.

[HIGH].

## DM8 — Three-axis self-validation on the harness itself

**Decision**: apply CLAUDE_V6_NAX.md §3.5 three-axis rule to this
sprint's own deliverables:

1. **Output sanity**: per-session smoke gate (V2 vs SDPA+bias RMSE
   < 1e-3) on a stable mid-density shape before timing begins. Fail
   → exit 2.
2. **Path entered**: warmup counter logged. Verify ≥ expected count
   per cooldown interval (e.g., 90s / 0.05s ≈ 1600 dispatches; report
   actual). Below 90% expected → flag.
3. **Edges preserved**: 4 control shapes (v2.36.0 CONFIDENT) must
   stay CONFIDENT (range < 10%) under the new protocol. Any
   regression here → REGRESSION verdict, abandon.

**Rationale**: axis-3 caught the v2.36.0 protocol regression. Same
gate must apply here to avoid recapitulating that failure mode.

[HIGH].

## DM9 — Bench logs to JSON + Markdown; runlogs to .txt

**Decision**: per-session JSON dict appended to
`docs/methodology/matched-workload-data.json` (list); analysis script
emits results.md + analysis.json; per-session stdout captured to
`matched-workload-runlog-M{1,2,3}.txt`.

**Rationale**: matches v2.36.0 layout (operator familiarity, simpler
diff against prior sprint). Runlogs preserve warmup-dispatch counts
even if JSON record fails to write.

[HIGH].

## DM10 — Branch hygiene: experiment branch from master tip

**Decision**: branch `experiment/sub1ms-matched-workload` cut from
master tip `ac36d59` (v2.36.0). Old methodology branch
(`experiment/methodology-sub1ms-protocol`) preserved separately as
archaeology.

**Rationale**: master is the canonical v2.36.0 baseline. Building on
the old methodology branch would inherit its abandoned protocol
artifacts (and the master amendments already exist on master from
Sprint U).

[HIGH].
