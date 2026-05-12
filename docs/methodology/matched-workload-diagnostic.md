# Matched-workload-family — REGRESSION diagnostic

**Verdict**: REGRESSION (0/3 HIGH resolved, 3/4 CONFIDENT regressed).
Worse than the v2.36.0 256×256 matmul protocol (which resolved 2/3 HIGH
but also regressed 3/4 CONFIDENT).

## Headline numbers

| Metric | v2.36.0 matmul protocol | Matched-workload (this sprint) |
|---|:--:|:--:|
| HIGH→CONFIDENT resolved | 2/3 | **0/3** |
| CONFIDENT shapes regressed | 3/4 | 3/4 |
| Total CONFIDENT after | 3/7 | **1/7** |
| Total HIGH after | 4/7 | **6/7** |
| Median ratio ≥ 1.2× | 7/7 | 7/7 (preserved) |

V2 still wins on raw perf in every shape (ratio 1.46×–7.93× vs SDPA+bias).
Variance characterization is the only failing axis.

## Cross-shape variance breakdown

| Shape | v2.36.0 flag | Matched-workload flag | Delta | Verdict |
|---|:--:|:--:|---:|---|
| lcsa_small_seq4k | HIGH | HIGH | +2.3% | unchanged (already noisy) |
| lcsa_small_seq4k_sparse | CONFIDENT | **HIGH** | +31.8% | **REGRESSED** |
| lcsa_mid_seq8k | CONFIDENT | **HIGH** | +31.7% | **REGRESSED** |
| lcsa_mid_seq8k_sparse | HIGH | HIGH | +4.5% | unchanged (already noisy) |
| lcsa_large_seq16k | CONFIDENT | **HIGH** | +28.2% | **REGRESSED** |
| lcsa_large_seq16k_sparse | CONFIDENT | CONFIDENT | +1.4% | preserved |
| lcsa_mid_seq8k_very_sparse | HIGH | HIGH | -17.8% | unchanged (improved within HIGH) |

**Critical observation**: the matched-workload regressed the **same
class** of CONFIDENT shapes the matmul protocol regressed (small_seq4k_sparse
+ large_seq16k), and additionally regressed `mid_seq8k`. Only
`large_seq16k_sparse` survived this protocol while it died under
matmul.

## Why the matched-workload approach failed

### Hypothesis H_MW (falsified)

H_MW asserted: a `sparse_attention_nax` warmup with different D and BT
than the measured kernel would (a) hold GPU power state above the
< 100ms downclock threshold while (b) NOT pollute the measured kernel's
L2 cache.

**(a) was achieved** — axis-2 path counters confirm 29,962-31,318
warmup dispatches per session, ~1450 per 90s cooldown interval, well
above the < 100ms threshold.

**(b) was NOT achieved**. Three CONFIDENT shapes regressed. The
"different shader instantiation" + "different tile size" isolation
arguments were insufficient.

### Probable mechanism (refined hypothesis)

The matched-workload warmup IS in the same kernel family as the
measured kernel. While the JIT shader is distinct (D=64 vs D=128, BT=16
vs BT=32), they share:

1. **Threadgroup memory layout pattern** — both allocate Q_smem/K_smem/
   V_smem/L/M scratch in the same offset structure. The L2 prefetcher
   may train on one and mispredict the other.
2. **Mask buffer access pattern** — both walk a 2D bool mask in NQ-row-
   major order. Cluster-shared L2 lines holding the warmup's 16 KB
   mask compete with the measured kernel's larger masks.
3. **bias buffer access pattern** — same logic for the float bias.
4. **Output write pattern** — both write to BHND-layout output via the
   same store path; write-allocate behavior aliases.

The matmul warmup (v2.36.0) hit a DIFFERENT set of cache lines (matmul
working set has no concept of mask/bias). The matched-workload hits
OVERLAPPING-BUT-NOT-IDENTICAL cache lines.

Crucially, the regressed shapes change between protocols:
- `large_seq16k_sparse` regressed under matmul, survived under matched-workload
- `mid_seq8k` survived under matmul, regressed under matched-workload

This **shape-specific regression flip** is the smoking gun: the warmup
mechanism interacts with the **measured kernel's working set in a
shape-specific way**. No single warmup mechanism can be "safe" for all
shapes simultaneously.

## What we now know about M5 Max sub-1ms variance

1. **GPU power state matters** (confirmed v2.36.0 §B): < 100ms idle
   causes +146% slowdown.
2. **Any warmup at < 100ms cadence holds power state** (confirmed both
   protocols).
3. **EVERY warmup that touches GPU resources perturbs the measured
   kernel's cache state** to some degree.
4. **Different warmup workloads regress DIFFERENT shapes** — no single
   warmup is universally non-polluting.
5. **The variance is real**, not a measurement artifact. The kernel
   itself has shape-dependent sensitivity to cache state, and any
   "active hold" of power state inevitably alters that state.

## Path-forward options (revised registry)

Original 4-option registry from `sub1ms-protocol-diagnostic.md`:

| Option | Status after this sprint |
|---|---|
| 1. Matched-workload family | **FALSIFIED** — different cache pollution, same failure mode |
| 2. Heartbeat-only (single threadgroup) | **PROMOTED** — only remaining cache-minimal option |
| 3. Metal API power-state lock | unknown — needs Apple API investigation |
| 4. Accept the trade-off (shape-aware default) | viable fallback |

### Option 2 design refinement (post-matched-workload)

Smallest possible warmup: a single-threadgroup dispatch that does
**no memory access beyond compute registers**. E.g., a Metal kernel
that just spins on a register accumulator for a fixed micro-time.
No buffer reads → no L2 footprint → no cache pollution. Only side
effect is power state.

Open question: does a register-only kernel actually hold power state?
The downclock threshold may be triggered by GPU clock idle, not by
buffer activity. If clock idle is the trigger, even a 1-µs register
spin every 50ms should suffice.

Required Apple-specific research: does Metal have an "idle threshold"
distinct from "no buffer access"? Initial guess: the GPU clock manager
counts cycles, not dispatches.

### Option 3 design

Investigate `MTLDevice.setLowPowerState(false)` / similar. If Metal
exposes a power-policy API, it would be the cleanest fix. As of
macOS 26, `MTLDevice` has `isLowPower` (read-only) and
`MTLCommandBufferDescriptor.errorOptions`, but no documented setter
for clock state.

### Option 4 design (most-pragmatic fallback)

Ship V2 as default ONLY for shapes where measured V2 wall-clock ≥ 2ms.
For sub-2ms shapes, fall through to V1. Implementation: shape-aware
dispatcher in `sparse_attention_forward()` that estimates V2 cost from
`qL × kL × density × D` and routes accordingly. The 2ms threshold is
empirically derived: at 2ms the < 100ms downclock window is hit
naturally by adjacent kernel dispatches, so power state holds without
explicit warmup.

## Recommendation

For the **next sprint**, prioritize **option 2 (heartbeat)** over
option 4. The reason: option 4 is a guaranteed pragmatic win but
caps the user-visible benefit. Option 2 has the only remaining
theoretical chance of producing a CLEAN sub-1ms measurement, which
benefits both the V2 default decision AND future kernel work.

If option 2 also fails, fall back to option 4 as the SHIP_BROAD
landing path for V2.

## Three-axis self-validation of this sprint

| Axis | Result |
|---|---|
| 1. Output sanity | PASS (smoke RMSE 5e-8 across all 3 sessions) |
| 2. Path entered | PASS (29-31k warmup dispatches per session, ~1450/interval) |
| 3. Edges preserved | **FAIL** — 3/4 CONFIDENT shapes regressed |

Axis 3 fired correctly. The same rule that caught v2.36.0's matmul
protocol caught this protocol. The three-axis discipline is working
as designed.

## Branch state

- Branch `experiment/sub1ms-matched-workload` preserved for archaeology.
- master remains at `ac36d59` (v2.36.0 SHIP_OPT_IN).
- Only doc deltas merge to master: this diagnostic + CLAUDE_V6_NAX.md
  §4.X amendment + SESSION_LOG entry.
- v2.35.0 SHIP_OPT_IN remains the production verdict for V2 sparse.
