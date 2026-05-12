# Canonical benchmark protocol for Apple Silicon sub-1.5ms kernels

## Why this protocol exists

mlx-mfa's §4-strict cooldown protocol (canonical for ≥1.5ms kernels)
fails to characterize sub-1.5ms wall-clock kernels under cross-session
analysis on M5 Max. Two REGRESSION sprints (mx.matmul continuous-workload
v2.36.0, matched-workload-family 2026-05-12) demonstrated empirically
that no userspace warmup approach defeats GPU power-state variance at
this scale.

Web research convergence (six sources) explains the mechanism. Apple
Silicon GPUs have aggressive dynamic frequency scaling driven by
hardware thermal/power feedback loops. The frequency manager runs
asynchronously to user dispatches and observes clock-idle windows on
its own cadence — not work intensity or buffer activity. No userspace
API exposes a P-state lock; this is an intentional design choice.

| Source | Finding |
|---|---|
| Apple Developer Forums thread 692062 | Apple engineer official response: "no way to query the GPU clock speed... Xcode developer tools allow you to select the performance state (max/medium/min). Locking the performance state gives consistent results." → No userspace API; design choice intentional. |
| Feng et al., arXiv 2501.14925 ("Profiling Apple Silicon Performance for ML Training") | Canonical methodology Apple Silicon published reference: "warm up for 10 iterations and then launch the kernel 100 times continuously and repeatedly... report the average of all these results." |
| MLX docs + WWDC25 Session 315 ("Get started with MLX for Apple silicon") | Sample code uses the `mx.eval` synchronisation call once for warmup + N iterations averaged. Pattern canonical for MLX-specific. |
| Draw Things MFA v2.5 NA release post (Nov 2025) | Production NAX precedent: "neural accelerators-enabled shaders take significantly longer time to specialize (often 10s or more for first generation)" + "ran the generation twice and took the second measurement" + binary artifacts cache. |
| MLX GitHub Discussion #1571 | Documented by MLX maintainers: "First call always much slower then fast. Curious observation, the second call always a little slower than the rest" — "second-call effect". |
| mlx-mfa internal: `docs/methodology/matched-workload-results.md` | REGRESSION verdict (2026-05-12): 3/4 v2.36.0 CONFIDENT shapes regressed under matched-workload warmup. Shape-specific regression flip vs mx.matmul protocol confirms warmup approach is intrinsically unable to be universally non-polluting. |

The Apple Silicon ecosystem canonical methodology is **warmup +
continuous back-to-back measurement**. mlx-mfa adopts this protocol
for sub-1.5ms kernels while retaining §4-strict for ≥1.5ms.

## Protocol specification

### Warmup phase

- **10 warmup iterations** per direction per shape (per Feng et al.)
- `mx.eval` synchronisation inside warmup loop
- Discarded from measurement

### Measurement phase

- **100 timed iterations** per direction per shape, **continuous**
  (no gap between iterations)
- `mx.eval` synchronisation inside the timing loop (canonical MLX
  pattern per WWDC25 Session 315 and MLX official docs)
- Record per iteration: wall-clock from `time.perf_counter` enclosing
  the `mx.eval` call
- Report stats: `p50`, `p95`, `p99`, `mean`, `min`, `max`

### Cross-session analysis

- **Ratio analysis preferred over absolute timings.** For each session,
  measure both V2 and SDPA back-to-back within the same continuous
  block, compute `ratio = V2_p50 / SDPA_p50`. The ratio is more stable
  across sessions than absolute timings because both numerator and
  denominator share the same per-session power-state baseline.
- Across 3 sessions, report `ratio range %` = `(max_ratio - min_ratio) / median_ratio × 100`.

### Verdict criteria

| Cross-session ratio range | Flag |
|---|:--:|
| < 10% | **CONFIDENT** |
| 10-20% | BOUNDARY |
| > 20% | HIGH_VARIANCE |

Same thresholds as §4-strict for consistency. Crucially, the variance
metric is **ratio range**, not absolute-timing range, because absolute
timings inherit GPU power-state variance that ratio analysis cancels.

## Protocol selection rule

Pre-bench, estimate wall-clock per shape from prior single-session data:

| Shape regime | Protocol |
|---|---|
| Wall-clock ≥ 1.5 ms | §4-strict cooldown (canonical for this regime) |
| Wall-clock < 1.5 ms | Canonical warmup + continuous (this protocol) |
| Unknown | Run §4.1 first, switch to §4.2 if wall-clock < 1.5 ms confirmed |

The 1.5 ms threshold is the empirical inflection: under §4-strict, shapes
above 1.5 ms keep the GPU busy enough that power-state variance is
inherently dampened by the kernel's own runtime; shapes below this
floor are dominated by power-state cycling between sessions.

## Production decision

mlx-mfa ships V2 sparse as default for shapes where canonical protocol
yields CONFIDENT ratio. Sub-1.5ms shapes (where ratio range exceeds 10%
even under canonical protocol) keep V1 default conservatively. Users
can opt into V2 via `MFA_LCSA_KERNEL_VERSION=v2` for these shapes
unconditionally.

This decision honors the **auto-default principle** (codified in
Sprint U / `docs/RELEASE_PHILOSOPHY.md`): optimizations activate
transparently for users where validated, opt-in elsewhere only where
validation is hardware-bounded.

## When the protocol still does not fit

If a shape produces HIGH_VARIANCE ratio under canonical methodology
across multiple sessions, the failure mode is no longer measurement
methodology but kernel-shape interaction. Such shapes should remain
SHIP_OPT_IN (V1 default + env override available) until either:

1. The kernel is re-engineered to reduce shape-dependent cache
   sensitivity (architectural fix), or
2. Apple exposes a P-state lock userspace API (unlikely per source 1
   above), or
3. The shape is explicitly documented as a known-variance case in
   `ENV_VARS.md` with the env override pattern.

## References

1. Apple Developer Forums thread 692062 — no userspace API for P-state
   lock. https://developer.apple.com/forums/thread/692062
2. Feng et al. (arXiv 2501.14925) — canonical warmup + continuous
   reference. https://arxiv.org/abs/2501.14925
3. MLX docs / WWDC25 Session 315 ("Get started with MLX for Apple
   silicon") — official `mx.eval` pattern.
4. Draw Things MFA v2.5 NA release post (Nov 2025) — production NAX
   precedent for warmup pattern + binary artifact cache.
5. MLX GitHub Discussion #1571 — documented second-call effect.
   https://github.com/ml-explore/mlx/discussions/1571
6. `docs/methodology/matched-workload-results.md` — REGRESSION verdict
   2026-05-12; falsifies userspace warmup as universally
   non-polluting.
