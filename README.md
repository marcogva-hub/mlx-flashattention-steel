# mlx-mfa

`mlx-mfa` is a Metal Flash Attention + serving-oriented runtime layer for MLX on
Apple Silicon. It provides high-performance attention kernels, runtime helpers,
and cache abstractions for dense training/inference plus modern serving flows.

Current version: **2.61.0** (release candidate — not yet on PyPI; current
PyPI release: **2.60.1**).  The complete, corrected state of
the 2026-06 optimization campaign (Phases I–III): the headline
promotions plus the Phase III-4 fresh-eyes whole-repo audit (9 passes,
run repeat-until-clean to a zero-finding fixed point; ~73 fixes), and the
III-6 conv3d small-channel kernel root-cause fix.

**Performance** (measured, per-cell, M5 NAX). ⚠ **Re-measured on macOS
26.6 (Sprint III-11):** Apple improved its primitives (SDPA-vjp, conv)
between the OS these were first measured on and 26.6, so the relative
speedups are **smaller and strongly shape-dependent on current macOS**.
mlx-mfa is **not slower** (kernels unchanged; forward stays bit-identical
to SDPA) — Apple's baselines got faster. Ratios vary with qL and thermal
state; see `.doc-archive/docs/v50/campaign-2026-06/phase3/sprint-III-11-report.md` for
the full table + methodology.

- V6NAX NAX backward **D=64 default-on**, **faster than SDPA-vjp**, qL-dependent:
  **~1.5–1.7× @qL2048, 2.2–2.8× @qL4096, 2.3–3.0× @qL8192** (causal ≳ non-causal).
  Canonical headline: **2.16–3.05× vs SDPA-vjp at qL≥4096** (M5 Max / macOS 26.6 /
  MLX 0.31.2; controlled B=1 H=4 median-30, fresh-input, engagement-proven
  `v6_split_backward` trace + fp32 oracle; ±30–40% run-to-run under thermal load).
  D=128 backward is **≈ break-even** on 26.6. Forward stays bit-identical to Apple SDPA.
- conv3d via the Apple MPP convolution2d primitive, default-on — fp16
  **median 1.64× vs `mx.conv_general`** (the public fallback the
  `MFA_DISABLE_CONV3D_MPP=1` knob selects), measured across the 6-shape
  SeedVR2 VAE production set (3-session §4-compliant, M5/26.6/MLX-0.31.2);
  bf16 **≈ parity** vs `mx.conv_general`. The separate **2.3–2.5× fp16 /
  1.4–2.7× bf16** figures are vs the *internal materialized-im2col*
  methodology baseline (NOT the public fallback — PERF_CLAIMS H-07), not a
  user-selectable denominator. Correctness, not speed, is the reason it is
  default-on (legacy im2col silent-corruption history).
- TurboQuant paged decode (opt-in KV compression via
  `TurboQuantPagedInferenceContext` — *not* auto-routed; you choose the
  trade-off) — **trades ~1.4–3× decode-step latency for a ~4–5× KV-cache
  memory reduction at cosine ~0.96, vs fp16 dense decode** (`step()`:
  `0.75 ms vs 0.33 ms` per step @ S=16K; KV cache e.g. `32 MB → ~6.5 MB`
  @ S=8K). That is the user-facing choice: spend a little decode latency to
  fit much longer context / higher concurrency in the same memory.
  *(Internal-perf history, not a user-facing choice: the gather/dequant +
  Apple-SDPA `step()` path is 6.5–23× faster than the fused TQ attend kernel
  it replaced — `0.75 ms vs 16.8 ms` @ S=16K, re-confirmed on 26.6 — but that
  prior kernel is gone, so it is context, not a baseline you can select.)*
- LCSA mask build: the "15.4×" figure is a **historical build-time**
  improvement (the prior builder is gone), **not a current runtime speedup**.
- D=256 causal M5 dispatch inversion: correctness fix (routes to the
  correct path), not a speed claim.

*All ratios above state numerator vs denominator with direction (a bare
"N×" is ambiguous). Numbers measured on
macOS 26.6 / M5 Max, median of N sessions; large-size ratios carry ±30–40%
run-to-run variance (clock-state bimodality). v2.56.0 adds the TQ-decode
eval-collapse speedups (IV-D1/D2) + a latent-overflow address-arithmetic fix
(A3-1); the attention/conv compute kernels are otherwise unchanged from v2.52.1
(the earlier 26.6 perf shifts were the OS/reference moving, not the kernels).*

> **⚠ Use the latest published release (current PyPI: v2.60.1; v2.61.0 is a
> release candidate, not yet on PyPI).** Historical note: v2.51.0 had two
> CRITICAL silent-corruption bugs (top-K Metal-grid undercount; NaN gradients through
> `return_lse=True`), fixed in v2.52.0; v2.52.0 had a small-channel conv3d
> silent-corruption bug (`C_in` not a multiple of 32), fixed in v2.52.1. Any release
> **≥ v2.52.1** is clear of those; always prefer the latest. See `CHANGELOG.md`.

New opt-in surface: hook telemetry `mlx_mfa.get_hook_stats()` (public, in `__all__`)
and the `mx.conv3d` auto-hook. The cider-style GQA-decode prototype is reachable
only as a **direct dotted-path import** (`from mlx_mfa.gqa_decode_cider import …`) —
it is **not** in the public `mlx_mfa.__all__` and not auto-routed (dormant; see CC-22).
Full campaign record: `CHANGELOG.md [2.52.0]` and
`.doc-archive/docs/v50/campaign-2026-06/` (`phase3/PHASE-III-CLOSE.md`).

### v2.50 highlights (shipped 2026-05)

- **Top-K bisection kernel** AUTO production default (Prompt 5c): 3.85×
  speedup over Phase 3a `mx.topk`-based path at audit shape.
- **V6NAX backward broadened to D ∈ {64, 128}** (Prompt 5b Section D):
  split kernels engage at parity with SDPA-vjp for D=128 + qL≥2048.
- **D=128 + causal + attn_bias mode 1/2** correctness fix (Prompt 5b
  Section C): V1 STEEL silent bias-drop bug resolved via V2 routing.
- **Sparse backward** via Prompt 5c hybrid orchestrator (NAX sparse
  forward + native sparse dV + SDPA-vjp dQ/dK).  4 native sparse
  kernels SHIPPED (Prompt 5d) but routed via opt-in
  `MFA_V6_BWD_SPARSE_NATIVE=1` per **Pattern #6** empirical finding
  (Apple SDPA NAX outpaces V6NAX NAX backward on M5+ at most shapes).
- **Sprint 4 V6NAX backward causal** production-active (D=64 + D=128)
  via Prompt 4 multi-gate dispatch fix.

Detailed v2.50 chronology + decisions: `.doc-archive/docs/v50/sprint-5d-decisions.md`.
Hardware support matrix: `docs/reference/HARDWARE_SUPPORT.md`.
Environment variables: `ENV_VARS.md`.
Perf claims registry: `docs/reference/PERF_CLAIMS.md`.
Known debt: `.doc-archive/docs/v50/known-debt-v2.50.md`.

---

### v2.39.1 era (historical)

Historical record: since v2.51.0 the V6NAX backward D=64 path (causal +
non-causal) is **default-on** — see the version header at the top of
this README; the opt-in env vars described below reflect the v2.39.x
state.

Option γ outcome **α**: H1 register pressure
root-caused + fixed.  The v2.39.0 outcome δ regression (-25% to -33% on
the fused dK+dV kernel) traced to per-SG register spilling at the
default BK=32 (TK=2 → two 8KB FP32 accumulators per SG).  Sprint v2.39.1
investigation lowered the default to BK=16 (TK=1), halving the
accumulator footprint and bringing the kernel under the M5 NAX
compiler's spill threshold.  Auto-default flipped back to fused for D=64.

**Measured speedups vs SDPA-vjp** (M5 Max, 3-session × 4w+12i median,
PUBLIC AUTO API `mx.grad(flash_attention(..., backend="auto"))` +
`MFA_ENABLE_V6_BACKWARD=1`):

| qL | v2.39.1 speedup | wall-time | Δ vs v2.38.1 |
|---|---|---|---|
| **4096** | **2.00×** | 9.31 ms | -2.9% |
| **8192** | **1.95×** | 37.73 ms | -1.4% |
| 16384 | 1.72× (3-sess) | 176.4 ms | thermal-drift footnote* |

\* qL=16384 3-session median 1.72× shows monotonic decline (1.88 →
1.72 → 1.67) attributable to thermal drift; fresh-machine spot-check
1.89×.  Session 1 representative of typical interactive workloads.

Investigation evidence: H1 register pressure CONFIRMED; H3 occupancy
FALSIFIED; H2 cache absorption partial-supporting.  Full record at
`.doc-archive/docs/v6-nax/v39-1-investigation-synthesis.md`.

Net effect on users: identical to v2.38.1 or modestly better.  No new
env vars required.  `MFA_V6_BWD_KERNEL=split` available as opt-out.

Builds on **v2.39.0** (Option γ fused kernel architectural addition,
outcome δ documented), **v2.38.1** (D_vec precompute), **v2.38.0**
(refactor + cleanup, investigation foundation).

Net effect on users: identical to v2.37.3.  The v2.37.x perf-claim
audit (`.doc-archive/docs/v6-nax/v2.37.x-perf-claim-audit.md`) and the two new
institutional rules (`CLAUDE_V6_NAX.md` §Z public API path testing
rule, §AA skill invocation checkpoints) remain in force.  v2.37.x
claim corrections (carried over unchanged):

- v2.37.1 "D=64 qL=2048: V6NAX wins 1.44×" → **retracted** (current
  canonical-methodology bench shows 1.15× kernel-level / ~1.06×
  end-to-end win, within measurement noise; v2.37.2 carve-out
  correctly does not engage at qL=2048)
- v2.37.0 "D=128 V6NAX backward 2.2-2.4× slower than SDPA-vjp" →
  **reclassified** as research characterization requiring
  `backend="mfa"` override; the public AUTO API correctly falls
  back to SDPA-vjp at parity (no user-facing impact)

**Reachable via public AUTO API** (current, since v2.51.0 — supersedes the
v2.37.2 carve-out below):
- D=64, qL ≥ 2048, causal + non-causal, f16/bf16, M5+ NAX — **default-on,
  no env var** → **2.16–3.05× faster end-to-end backward vs SDPA-vjp**
  (canonical, M5/26.6/MLX-0.31.2; opt-out `MFA_DISABLE_V6_BACKWARD=1`). The
  old "1.81-1.82× via `MFA_ENABLE_V6_BACKWARD=1`" figure is **withdrawn**
  (PERF_CLAIMS, superseded by the default-on split-V6 number).
- `MFA_ENABLE_V6_BACKWARD=1` is now the **D=128** opt-in only (D=128
  backward is ≈ break-even / research; D=64 needs no env).
- All other shapes: AUTO path defaults to SDPA-vjp — correct,
  no user action needed

See `docs/reference/TRAINING_QUICKSTART.md` for the updated user-facing perf
recommendation and `.doc-archive/docs/v6-nax/v2.37.x-perf-claim-audit.md` for
the per-claim reachability audit that drove these corrections.

D=128 V6NAX backward is 2.2-2.4× slower **only on the forced `backend="mfa"` path** (architectural floor at FP16 NAX hardware peak; Apple's SDPA-vjp uses a different algorithm) — the **public AUTO API does not engage it**: D=128 backward falls back to SDPA-vjp (≈ break-even), so AUTO users are unaffected (consistent with lines :150/:167 and `docs/reference/PERF_CLAIMS.md`). At the time, the default (env unset) preserved v2.36.1-exact behavior; since v2.51.0 the D=64 backward is default-on. All prior ship-defaults preserved: shape-aware V2 sparse default (v2.36.1), canonical Apple Silicon benchmark methodology (`.doc-archive/docs/methodology/canonical-protocol.md`), Sprint U auto-on-import hooks, Conv3D NAX.

## Minimal Usage (auto-default)

```python
# illustrative-fragment: x, weight, q, k, v are defined in surrounding prose
import mlx.core as mx
import mlx_mfa  # auto-installs optimization hooks at import

# Eligible Conv3D shapes on M5+ auto-route to the Apple MPP convolution2d
# primitive — median ~1.64× vs un-hooked mx.conv_general (SeedVR2 VAE
# production set). (The 2.3–2.5× fp16 figure quoted elsewhere is vs an
# internal im2col baseline, not vs this mx.conv_general path — H-07.)
y = mx.conv_general(x, weight, padding=(1, 1, 1))

# Sparse attention on M5+ auto-routes to NAX-aware dispatcher:
from mlx_mfa import flash_attention_sparse
out = flash_attention_sparse(q, k, v, block_mask)
```

## Three usage levels

1. **Default (auto-on-import)** — `import mlx_mfa` activates all validated
   optimizations transparently. See above.
2. **Explicit API** — `from mlx_mfa import flash_attention, sparse_attention_dispatch, ...`
   for direct calls when you need control or mlx-mfa-specific features
   (varlen, paged, TurboQuant, etc.).
3. **Expert mode** — `patch_seedvr2_vae(model)`, `patch_flashvsr_lcsa(model)`,
   `patch_mlx_lm()` for granular per-module control + verbose logging.

## Disabling auto-hooks

```bash
# Global disable via env var
export MFA_DISABLE_AUTO_HOOKS=1
python your_script.py
```

```python
# Programmatic disable / re-enable (idempotent)
import mlx_mfa
mlx_mfa.disable()  # restore vanilla MLX
# ... your benchmark ...
mlx_mfa.enable()   # restore mlx-mfa hooks
mlx_mfa.hooks_status()  # introspection dict
```

## Foreword

**MLX Metal Flash Attention - Why?**

I've been working on personal ports of Video Super Resolution and Video 
Reconstruction models for months, but always ended up frustrated by the 
slow inference in my M1 Max MacBook Pro. And to try to mitigate this without
having to buy a brand-new, very expensive new M4, then M5 Max, I decided to
at least try to port Flash Attention to Mac, hoping for better results. And 
having better results porting VSR/VR models to MLX than MPS, that's why I ended
up doing it.

At this point, despite the lower than hoped for results, I'm still pretty
satisfied with the results in my M1 Max MBP.

Since early May 2026, all development and testing has run on my M5 Max, where
the focus has been the NAX (Neural Accelerator) implementation that this
hardware makes possible. My M1 Max now serves as a secondary validation target.

v2.32.0 introduces a **strategic shift in dispatch on M5+ NAX hardware**.
Apple's MLX 0.31.2 ships an excellent NAX-based SDPA kernel
(`steel_attention_nax.h`) that matches the V6NAX NAX-direct path mlx-mfa
shipped in v2.31.0 — and Apple's kernel benefits from continuous upstream
tuning. Rather than compete on a surface where Apple has structural
advantages, mlx-mfa now **routes forward attention to MLX SDPA on M5+
when SDPA covers the shape and feature set optimally**, and keeps native
kernels for everything else:

- `head_dim ∉ {64, 128}` (D=80, D=96, D=192, D=256, D=512) → mlx-mfa
- Block-sparse / LCSA mask                                 → mlx-mfa
- Additive attention bias (modes 1, 2)                     → mlx-mfa native bias kernel
- Sliding window                                           → mlx-mfa STEEL window kernel
- Backward pass                                            → mlx-mfa (Apple's NAX backward NYI)
- All M1–M4 hardware (no NAX)                              → mlx-mfa V2/V3/V6 NAX legacy
- Specific empirical carve-outs from Sprint A sweep        → mlx-mfa

Override via `MFA_DISABLE_SDPA_ROUTE=1` (recovers v2.31.0 dispatch on M5+).
This preserves mlx-mfa as a unified attention toolkit across all Apple
Silicon generations while stopping unnecessary competition with Apple's
upstream optimizations on shapes Apple covers well.

The v2.31.0 performance numbers (V6NAX +33-40% wins on D=128) were measured
under specific environmental conditions that did not reproduce in the
v2.32.0 cross-session diagnostic. v2.32.0 ships with reproducible-conditions
methodology baked into the bench infrastructure (`bench/v32_multisession_capture.py`,
`.doc-archive/docs/v6-nax/v32-multisession-protocol.md`, `CLAUDE_V6_NAX.md` Artifact #5).
The architectural improvements that motivated v2.31.0 (V6NAX NAX-direct
forward kernel, multi-SG parallelism via per-SG row partitioning) remain
in the codebase as a regression canary and as the dispatched path when
`MFA_DISABLE_SDPA_ROUTE=1` is set.

v2.31.0 shipped the **V6NAX NAX-direct rewrite**. V6 NAX's forward hot path
uses Apple's `NAXFrag::mma` and `NAXTile<T, TQ, TD>` primitives directly
(the pattern from `steel_attention_nax.h`), bypassing MPP cooperative_tensor
constraints that previously imposed `execution_simdgroups<1>`. Multi-SG
parallelism comes from per-SG row partitioning at the kernel level
(`tm = 16 * TQ * sgid`), not via cooperative_tensor distribution — so the
V33 cross-SG opacity issue disappears entirely.

The historic D=128 long-N gap is **closed**: production VSR/DiT shapes
that were stuck at 1.5–1.7× SDPA now run at SDPA parity. **SeedVR2-small
at 0.89× SDPA actually beats SDPA**, the first time V6 NAX has dipped
below 1.0× on a production shape. Numerics also improve 4–30× over legacy
because the manual `simd_shuffle_xor` row reductions on FP32 accumulators
inside `NAXFrag::row_reduce` are bit-exact, vs MPP's `reduce_rows` which
had tile-boundary FP rounding artifacts. Dispatch is shape-aware: V6NAX is
default for D=128 and D=64 N≥2048, legacy stays for D=64 small-N
(FlashVSR-dense regresses under V6NAX — root cause TBD).

v2.30.0 extended v2.29.0's V6 NAX work along three axes: (1) **GQA
single-Otile** — the BHND rewriter now handles `Hq % Hk == 0` so GQA
shapes use the single-Otile kernel directly, gaining 7-14% over the
v2.29.0 legacy fallback; (2) **dispatch v5** (the v6 attempt was reverted
after thermal-controlled re-bench); (3) **tgmem allocation cleanup** —
single-Otile + bypass no longer allocates the unused P_buf threadgroup
memory.

v2.29.0 shipped **V6 NAX single-Otile** for M5+ hardware: an Apple-style
single-buffer kernel (`loopForwardSingleTile`) with autoresearch-tuned
default tile config (BQ=16 universal, per-D BK/SG).

v2.27.0 added native Metal `attn_bias` kernel support (additive bias on
attention logits without SDPA fallback), a dispatch audit for 11 DiT/UNet
architectures, and varlen validation for token merging workflows.
See `CHANGELOG.md` for full details per version.

Thank you for your interest, and let me know if you've been able to improve
on my work!

## Current Repository Status

- **V2 dense** is the main production path.
- Strongest dense wins on M1 Max remain **causal D=64/128** and tile-skip
  regimes (window/sparse).
- **D=256** is narrow benchmark-backed only (not broad promotion).
- **D=512** remains SDPA-default.
- **Native dense backward** was benchmarked and not promoted.
- **Sage** is a specialized decode backend (narrow, benchmark-gated use).
- **V3** remains experimental/hardware-dependent (conditionally auto-routed on M5).
  **V4/V5** STEEL forward variants were retired in v2.61.0 (removed from the build).
- **TurboQuant** KV cache compression (Phase 1–4) production-ready.
- **SVDQuantLinear** W4A16 + optional SVD low-rank correction for DiT quantization.
- **GNA native kernel** inline 3D window attention (D=128, f16/bf16, forward-only).
- **Native `attn_bias`** additive bias on logits via Metal kernel (modes 1/2: per-KV and per-head per-KV broadcast).
- Serving/runtime capability surface is now substantially expanded:
  - paged KV + packed varlen query support
  - paged continuous batching/remap
  - explicit chunked prefill
  - runtime-managed prefix reuse
  - runtime speculative draft/verify flow
  - deeper splitfuse runtime integration
  - KV cache abstraction layer
  - minimal real hybrid/offload-capable cache behavior (local offload tier)
  - TurboQuant compressed KV serving (`create_decode_runtime(turboquant=True)`)

## Limitations

- Primary development and validation hardware is **Apple M5 Max** (NAX focus);
  **Apple M1 Max** is a secondary validation target. M1-Max validation of the
  latest changes is pending hardware availability (see TESTING / CHANGELOG).
- Broad parity claims against CUDA FlashAttention ecosystems are not made.
- Some advanced paths are intentionally narrow, bridge-based, or explicit-only.
- Hybrid offload is currently a **local offload milestone**, not remote/
  distributed cache infrastructure.
- Future major hardware-specific optimization work is deferred pending newer
  Apple hardware (M5+).

### Fixed in the audit routing pass (Phase F, 2026-06-18)

The audit's Phase F routing rebuild fixed the two largest sparse-dispatch gotchas
(both verified by runtime fingerprint + three-axis):

- **D=128 sparse with a built-in mask now routes to the real NAX-sparse kernel.**
  The makers (`make_causal_block_mask`, `make_sliding_window_mask`,
  `make_strided_mask`, `make_lcsa_mask`, …) emit a **symmetric 32×32** block-mask
  (was asymmetric 32×16), so the M5+ symmetric-bt auto-route engages NAX instead of
  silently falling back to dense SDPA — **1.7–4.2× faster** at density < 0.78. A
  density gate routes near-dense masks (≥ 0.78, env `MFA_NAX_SPARSE_DENSITY_CEILING`)
  to SDPA, where it is faster. [Verified — three-axis, M5/26.6]
- **D=64 sparse now routes to V2 (matmul2d), ~9× faster** than the old V1-scalar
  path. The `qL*kL*D = 2^31` work-product threshold in `decide_auto_version` is
  retired; routing is by V2-capability (head_dim). V1-scalar was never fastest.
  [Verified — three-axis, M5/26.6]

### Known issues — verified M5/26.6 runtime dispatch (audit, 2026-06-18)

- **`flash_attention_sparse` at D=128 falls back to dense SDPA** for *asymmetric /
  custom* masks (`bt_q≠bt_k`), *small* masks (mask bytes < 4096; NAX device-pointer
  lowering excludes them), or *near-dense* masks (density ≥ 0.78). These are
  deliberate routes (SDPA is correct + faster there), not a silent bug. [Verified]
- **`mx.grad(flash_attention_sparse)` runs a *dense* SDPA-vjp backward by default** —
  the sparse forward win does not carry to the backward unless `MFA_ENABLE_V6_BACKWARD=1`
  (+ `bt≥64`). [Verified]
- **Dense forward routing (M5):** `backend="auto"` dense **D=128** routes to the **NAX
  matmul2d** forward (`v6_nax_forward`), which is **parity-to-modest-win vs Apple SDPA at
  D=128** (0.89–1.03× across N, never loses; F-2 Change 3). Works at **all scales** (the
  scale is plumbed through the binding); backward is SDPA-vjp (bit-exact). **D=64** routes
  to SDPA (the dense-NAX decision: NAX matmul2d loses 1.17–1.22× there) **on M5/NAX
  for ALL D=64 dense `auto`** (`should_use_mfa(D=64, has_nax=True)`=False → byteΔ=0 vs
  SDPA, verified) — the "causal & B·H ≥ 4 & N ≥ 4096 → MFA primitive (V3 conditional-auto)"
  pre-emption is **M3/M4-tier only** (`has_nax=False`); cross-attention (N≠S), windowed,
  and biased shapes also stay SDPA. Opt out of the dense
  NAX route with `MFA_DISABLE_V6_DENSE=1`. **`backend="mfa"` (simdgroup STEEL) remains
  legacy on M5** — Apple SDPA is 2–4× faster than the *simdgroup* kernels (a different
  family from the NAX matmul2d forward). The remaining ~5–7pp ALU gap to a larger D=128 dense
  win is a future single-`O`-accumulator source-generator rewrite. [Verified]
- **STEEL V4/V5 forward variants were removed from the build in v2.61.0** (never
  auto-routed; previously env-gated opt-in, found to hold no advantage on M5). [Verified]
- **`sage_attention` (int8) is ~4.7× slower than SDPA on M5** (cos ~0.997) — kept as
  an expert/opt-in backend, not auto-routed. [Verified]

**Path-dependent env semantics (verified):** `MFA_ENABLE_V6_BACKWARD=1` engages
**full-native** dQ/dK/dV for the *dense* D=128 backward, but only **native-dV** (dQ/dK
stay SDPA-vjp) for the *sparse* hybrid backward; full-native *sparse* backward needs
`MFA_V6_BWD_SPARSE_NATIVE=1` (declined-on-perf, opt-in). See `NAMING.md` for the env-var
rename table.

**Correctness coverage (verified):** every kernel **path** is independently
oracle-locked across its supported dtype / causal / feature regimes, and the
runtime dispatch is fingerprint-locked — see the kernel-math oracle envelope
`tests/test_oracle_envelope.py` (dense/sparse/decode/varlen/backward × {f16,bf16}
× causal × shape-regime, plus D=256, GQA fwd+bwd, softcap, sliding-window,
asymmetric `D_v`, sparse non-causal, GNA-windowed, `return_lse` value, TurboQuant
pack/unpack — each with an independent fp32/fp64 oracle + byteΔ-vs-SDPA engagement
proof) and the paged envelope `tests/test_paged_envelope.py` (homogeneous +
heterogeneous `seq_lens`, per-sequence causal, GQA, validation),
`tests/test_{sparse_family,dense_steel_family,backward_family,b4_family}_*_lock.py`,
`tests/test_dispatch_map_lock.py`, and `tests/test_fingerprint_discipline.py` (the last
makes "test passes while running the wrong kernel" structurally catchable). Maintainer
reference: `.doc-archive/docs/v50/campaign-2026-06/audit/` (dispatch map + 4 per-kernel family specs).

[See the v2.31.0 V6 NAX foreword above and the "Best M5 Max Benchmark
Highlights (v2.31.0)" table below for current numbers.]

## Best M1 Max Benchmark Highlights

Representative benchmark-backed outcomes (see `RESULTS.md` and
`docs/reference/BENCHMARKS.md` for details):

| Area | Representative result (M1 Max) | Interpretation |
|---|---|---|
| Dense causal V2 | up to ~**1.82x** vs SDPA (D=64, N=8192) | Primary production win regime |
| Dense causal V2 | up to ~**1.75x** vs SDPA (D=128, N=16384) | Strong long-sequence causal performance |
| Sliding window | **20.8× (M4 Max) / 18.4× (M1 Max)** vs full SDPA at D=128 N=8192 win=256 (RESULTS.md §2) — scales with mask sparsity | Tile-skip regime remains strongest |
| D=256 | narrow causal long-N wins (for example ~**1.16x** at N=16384 f16) | Keep narrow policy only |
| D=512 | decision pass found **no broad wins** | SDPA-default remains correct |

## Best M5 Max Benchmark Highlights (v2.31.0)

V6 NAX path on production VSR/DiT shapes (cross-session multi-run, iStat performance fan profile).
The shape-aware dispatch picks V6NAX (NAX-direct) where it wins, legacy V6 NAX otherwise.

| Shape | D | Path | V6 NAX vs SDPA |
|---|---|---|---|
| FlashVSR-dense | 64 | legacy | 1.23× SDPA |
| LTX2-cross | 64 | **V6NAX** | **1.07× SDPA** |
| SeedVR2-small | 128 | **V6NAX** | **0.89× SDPA ⭐ (beats SDPA)** |
| CogVideoX | 128 | **V6NAX** | **1.03× SDPA** (parity) |
| SeedVR2-large | 128 | **V6NAX** | **1.01× SDPA** (parity) |

GQA shapes (Sprint B single-Otile path, legacy V6 NAX):

| Shape | V6 NAX vs SDPA |
|---|---|
| GQA-Hq32-Hk8 D=128 | 1.06× ⭐ |
| GQA-Hq16-Hk4 D=64 | 1.17× |
| GQA-Hq40-Hk8 D=128 | 1.16× |
| GQA-Hq8-Hk2 D=64 | 1.18× |

Numerical: V6NAX RMSE FP32 vs SDPA reference is 9e-7 to 4e-6 across all 5 shapes —
4–30× more stable than legacy V6 NAX (1.5e-5 to 6e-6). Manual simd_shuffle_xor row
reductions on FP32 accumulators are bit-exact, vs MPP's reduce_rows which had
tile-boundary FP rounding.

## Serving/Runtime Capability Summary

| Capability | Maturity | Current status |
|---|---|---|
| Paged KV decode runtime | Fully usable | Explicit runtime/API usage; no broad auto-promotion |
| Paged + packed varlen queries | Production (fused kernel) | Single-dispatch fused kernel for all query/KV length combinations |
| Paged continuous batching remap | Fully usable | Explicit `cache_batch_idx` semantics + runtime helpers |
| Chunked prefill | Fully usable (scheduler-oriented) | Operational capability; not a throughput win on current matrix |
| Runtime prefix caching | Fully usable | Register/seed/reuse path integrated with runtime metadata |
| Runtime speculative decode | Fully usable (narrow) | `speculative_step` + verify integration; scheduler engine still future work |
| Splitfuse runtime integration | Narrow/conditional | Runtime path exists; performance remains shape-sensitive |
| Hybrid KV cache + local offload tier | Narrow/conditional milestone | Real hot/cold/offloaded behavior locally; remote offload future work |
| TurboQuant KV compression (Phase 4) | Production | 5.33× K compression, WHT fused in kernel (1.1–1.4× faster) |
| SVDQuantLinear | Production | W4A16 + rank-r FP16 correction; `quantize_model()` tree walker |
| GNA native kernel | Production | Inline 3D window attention (D=128); exact per-element masking |
| Native `attn_bias` | Production | Modes 1/2 via V2 STEEL; modes 0/3 SDPA fallback |
| External cache adapter layer | Experimental groundwork | Concrete local backend provided; external backend integrations pending |

## Repository Guide

- Feature coverage: [`docs/reference/FEATURE_COVERAGE.md`](docs/reference/FEATURE_COVERAGE.md)
- API manual: [`docs/reference/API_MANUAL.md`](docs/reference/API_MANUAL.md)
- Architecture: [`docs/reference/ARCHITECTURE.md`](docs/reference/ARCHITECTURE.md)
- Inventory map: [`docs/reference/INVENTORY.md`](docs/reference/INVENTORY.md)
- Benchmark interpretation: [`docs/reference/BENCHMARKS.md`](docs/reference/BENCHMARKS.md)
- Root benchmark summary: [`RESULTS.md`](RESULTS.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Historical development archive: `.doc-archive/devnotes/` (internal archive — git history, not shipped)
- Examples: [`examples/`](examples/)

## Production vs Narrow vs Experimental

| Status | Components |
|---|---|
| Production | V2 dense causal small-D path; window/sparse tile-skip; SDPA fallback policy; TurboQuant KV compression; SVDQuantLinear; GNA native kernel; native `attn_bias` |
| Narrow / conditional | D=256 causal long-N policy; Sage decode regimes; splitfuse/page-native runtime paths; hybrid local offload behavior |
| Experimental | V3 family (V4/V5 retired in v2.61.0); external/LMCache-like backend extensions beyond local adapter |

## Recommended Usage

1. Use `backend="auto"` for dense attention and let policy route between V2 and SDPA.
2. Use `create_decode_runtime(...)` for serving flows instead of stitching helper calls manually.
3. Treat paged/packed/chunked/prefix/speculative features as explicit runtime capabilities.
4. Use Sage as a specialized decode backend only when your workload matches the
   benchmark-backed regime.

## Installation

```bash
pip install -e .
```

**Requirements:** macOS ≥ 14.0 · Python ≥ 3.10 · **MLX ≥ 0.31.2**. mlx-mfa is published
**sdist-only** — `pip install` compiles `_ext` against *your* installed MLX. The MLX floor is
**0.31.2** because `_ext` links nanobind 2.12.0 (NB_INTERNALS v19); MLX 0.31.0/0.31.1 ship an
older nanobind and would produce an ABI-incompatible extension. The V6/NAX Neural-Accelerator
paths activate at runtime on M5 / macOS 26.2+ (gated); on M1–M4 / older macOS the standard
kernels + Apple SDPA are used.

> **Network at build time:** the build pins nanobind 2.12.0 via CMake `FetchContent` (to match
> MLX's ABI), so `pip install` fetches it from GitHub during compilation — an **offline/restricted
> build environment will fail** at the nanobind fetch. Pre-warm a network-enabled build cache, or
> vendor nanobind 2.12.0, for air-gapped installs.

### Verify acceleration is active

mlx-mfa is **correct but unaccelerated** if its native extension `mlx_mfa._ext` did not load
(falls back to Apple SDPA). Check after install:

```python
import mlx_mfa
mlx_mfa.has_nax()              # True → M5+ NAX fast path is live
mlx_mfa.has_nax(reason=True)   # (False, "<code>") explains why when off
mlx_mfa.is_mfa_available()     # True → the extension (any kernel tier) loaded
```

`has_nax(reason=True)` reason codes when `False`:

| Code | Meaning | Fallback is… |
|---|---|---|
| `ext-load-failed` | On Apple Silicon, but `_ext` didn't import (Python/MLX ABI mismatch or failed build) | **Unexpected** — the library warns loudly; fix by matching Python/MLX and rebuilding |
| `pre-m5-hardware` | `_ext` loaded, GPU is M1–M4 | Expected — STEEL kernels still accelerate; only NAX/V6 is M5+ |
| `unsupported-platform` | Not Apple-Silicon macOS | Expected — no Metal backend |

On an unexpected fallback (Apple Silicon, `_ext` failed) mlx-mfa emits a one-time `RuntimeWarning`
naming the likely cause. Suppress it with `MFA_SILENCE_NAX_WARNING=1`. To **require** acceleration
(raise instead of falling back) set `MFA_REQUIRE_NAX=1`, or call `mlx_mfa.has_nax(strict=True)`.
"Fallback" always means *correct results, no speedup* — never wrong numbers.

## Minimal Usage

```python
import mlx.core as mx
from mlx_mfa import flash_attention, flash_attention_gna, create_decode_runtime
from mlx_mfa import SVDQuantLinear, quantize_model

# Dense attention
q = mx.random.normal((1, 8, 1024, 128)).astype(mx.float16)
k = mx.random.normal((1, 8, 1024, 128)).astype(mx.float16)
v = mx.random.normal((1, 8, 1024, 128)).astype(mx.float16)
out = flash_attention(q, k, v, causal=True)

# Token merging proportional attention (native Metal, no SDPA fallback)
merge_counts = mx.ones((1, 1, 1, 1024), dtype=mx.float16)
merge_counts[..., :256] = 2.0   # first 256 tokens are merged pairs
bias = mx.log(merge_counts)     # [1, 1, 1, N_kv] — mode 1 broadcast
out_biased = flash_attention(q, k, v, attn_bias=bias)

# GNA (Generalized Neighborhood Attention) — 3D window
# Video: 8 frames of 32x32, local 3D window, sliding
q_vid = mx.random.normal((1, 8, 8192, 128)).astype(mx.float16)
k_vid = mx.random.normal((1, 8, 8192, 128)).astype(mx.float16)
v_vid = mx.random.normal((1, 8, 8192, 128)).astype(mx.float16)
out_gna = flash_attention_gna(q_vid, k_vid, v_vid,
                               seq_shape=(8, 32, 32),
                               window_size=(2, 8, 8),
                               stride=(1, 1, 1))

# SVDQuantLinear — W4A16 + SVD low-rank correction
# (quantize_model replaces nn.Linear layers in-place)
# model = quantize_model(model, group_size=64, bits=4, rank=32)

# Serving-oriented runtime
rt = create_decode_runtime(
    backend="auto",
    paged=False,
    quantized_kv=False,
    B=1,
    H_q=8,
    H_kv=8,
    D=128,
    max_seq_len=4096,
)
out_prefill = rt.prefill(q, k, v)
out_step = rt.step(
    mx.random.normal((1, 8, 1, 128)).astype(mx.float16),
    mx.random.normal((1, 8, 1, 128)).astype(mx.float16),
    mx.random.normal((1, 8, 1, 128)).astype(mx.float16),
)
```

## Conv3D NAX support (M5+ Apple Silicon)

Since v2.50.2/v2.51.0 the default Conv3D path is the Apple MPP
`convolution2d` primitive. **Denominators (PERF_CLAIMS H-07):** the
**2.3–2.5× fp16 / 1.4–2.7× bf16** figures are vs the *internal
materialized-im2col* baseline (a methodology denominator), **not** vs
`mx.conv_general`. Vs the **public** fallback that `MFA_DISABLE_CONV3D_MPP=1`
selects — `mx.conv_general` — the win is **median 1.64×** on the SeedVR2 VAE
production set (below). The legacy materialized-im2col path is non-default
and reachable via `MFA_DISABLE_CONV3D_MPP=1` only as a research baseline.

mlx-mfa includes a NAX-accelerated 3D convolution path for shapes matching
the SeedVR2 VAE production profile. Sprint C v1.x landed a SHIP-DEFAULT
verdict (median **1.64×** speedup vs `mx.conv_general` across 6 production
shapes); Sprint D migrated the dispatch from Python orchestrator to a
C++ `_ext.conv3d_nax_forward` binding.

### Quickstart

```python
import mlx.core as mx
from mlx_mfa.conv_nax import conv3d_nax_forward

# Channels-last layout: (B, T, H, W, C_in)
x = mx.random.normal((1, 5, 64, 64, 512)).astype(mx.float16)
w = mx.random.normal((512, 3, 3, 3, 512)).astype(mx.float16)  # (C_out, K_T, K_H, K_W, C_in)
y = conv3d_nax_forward(x, w, stride=(1,1,1), padding=(1,1,1), dilation=(1,1,1))
# y.shape == (1, 5, 64, 64, 512)
```

### Supported shapes

- 3D inputs in `(N, T, H, W, C_in)` channels-last layout (matches `mx.conv_general`)
- `3×3×3` and `1×1×1` kernels (other small kernels may work but are not in the validated set)
- FP16 dtype (BF16 supported in code paths but not yet on the validated bench set)
- `stride = (1, 1, 1)`, `dilation = (1, 1, 1)`
- Symmetric padding (int or 3-tuple) **or** asymmetric padding via
  3-tuple of `(left, right)` pairs **or** flat 6-tuple
  `(T_left, T_right, H_left, H_right, W_left, W_right)`. Causal video
  conv: `causal_pad_t=True` flag or `padding=((K_T-1, 0), (pH,pH), (pW,pW))`.

### Expected speedup vs `mx.conv_general` (M5 Max, FP16)

| Shape profile (SeedVR2 VAE) | M | K | Speedup |
|---|---:|---:|---:|
| mid_resnet (small M, K=13824) |     20,480 | 13824 | **2.26×** |
| up1_resnet (med M, K=13824)   |    147,456 | 13824 | **2.00×** |
| up2_resnet0_chunk_cap         |    297,000 | 13824 | **1.64×** |
| up3_resnet_chunk_cap (K=3456) |    592,896 |  3456 | 1.02× (parity) |
| up2_resnet_full               |  1,114,112 |  6912 | **1.65×** |
| up2_resnet0_peakflops         |  1,114,112 | 13824 | **1.54×** |

Median across the SeedVR2 VAE production set: **1.64×**. The full 3-session §4-compliant
methodology is in `.doc-archive/docs/conv-nax/ship-shelve-decision.md` (internal archive —
git history, not shipped).

### Caveats

- At **K ≤ 3456** (small `in_channels`), speedup approaches parity (~1.0×)
  as the workload becomes bandwidth-bound. No regression, just no gain.
- **int32 byte-offset chunking invariant.** MPP `matmul2d` uses int32 for
  internal byte addresses; single-buffer reads beyond `2^31` bytes produce
  NaN. `conv3d_nax_forward()` auto-chunks `M` to keep each chunk's
  im2col buffer below the safety limit (`2^31 × 0.875` bytes). Users
  don't need to think about this; documenting it because it's the
  Sprint C Phase 1.2 lesson learned and the institutional rule for any
  future MPP-based code in this repo.
- **C++ entry point.** Production dispatch goes through
  `mlx_mfa._ext.conv3d_nax_forward` (Sprint D migration). The Python
  orchestrator is preserved as
  `_conv3d_nax_forward_python_legacy` for diagnostics; toggle via
  `MFA_CONV_NAX_USE_PYTHON_LEGACY=1`.

### Integration with SeedVR2 VAE

For drop-in replacement in SeedVR2 VAE Python code (or any MLX model
using `mx.conv_general` for Conv3D):

```python
from mlx_mfa.integrations.seedvr2_vae import patch_seedvr2_vae
model = patch_seedvr2_vae(model)
# Walks model modules, swaps Conv3D layers matching the NAX-eligible
# profile to route through conv3d_nax_forward(). Skips ineligible layers
# (logged with reason). Restorable via patch_seedvr2_vae(model, restore=True).
```

## Sparse attention on M5+

`mlx_mfa.flash_attention_sparse(q, k, v, block_mask, ...)` is the
block-sparse attention API for FlashVSR / SparkVSR / similar LCSA
patterns. On M5+ Apple Silicon it routes through the NAX-aware
dispatcher (`lcsa_nax.sparse_attention_dispatch`) by default since
**v2.36.1**, selecting NAX sparse kernels by shape and density.

On non-NAX hardware where the dispatcher does not engage, the
historical v2.33.x fallback applies: SDPA with the block mask
expanded to a float bias (bias cached by `id(block_mask)`).

Pre-M5 hardware (M1-M4) is unchanged: routes through the native C++
STEEL V1 sparse kernel that already skips masked tiles.

## License

MIT. See [`LICENSE`](LICENSE).
