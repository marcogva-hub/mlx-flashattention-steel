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

## Active claims (as of v2.61.0)

> **M5 NA peak recalibration note (docs audit 2026-06-19):** several entries below
> use "≤ 51.8 TFLOPS (effective gate)" as the plausibility check the measured rate
> was validated against. The M5 Max fp16/bf16 **matmul** peak was recalibrated to
> **~62 TFLOPS** (fp32 ~42, steel) on 2026-06-19. The recalibration only *widens* the
> plausibility band — every rate quoted below (≤49 forward-tile, ≤48.2 effective at
> 8192-nc, 7.5–20.1 backward, 55.6 conv roofline) sits below both the old 51.8 and
> the new ~62 ceiling, so no claim's verdict changes. Read "≤51.8 gate" as
> "comfortably below the M5 matmul ceiling".

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

### NAX dense D=128 routing threshold (M5 Max, `research/nax-routing-threshold-m5`, Tier-2 #1, 2026-06-18)

The F-2 dense-D=128 auto-route (above) sent **all N** to NAX, but at small N Apple SDPA is faster — a
localized regression. Crossover measured NAX vs SDPA in **absolute ms** (3-session §AA.4 subprocess
isolation; all rates ≤ 51.8 TFLOPS gate; both paths correct attention, fp32-err 3.8e-6 across the
boundary). **The crossover is governed by N (sequence length) alone**, NOT total work N·B·H — equal
N·B·H=16384 gives opposite winners: (N=512,B=4)=SDPA 1.03× vs (N=2048,B=1)=NAX 0.997×.

- **Threshold = N ≥ 2048 → NAX; N < 2048 → SDPA** (`MFA_V6_DENSE_MIN_N`, default 2048; `=0` forces all-N
  NAX — keep-all-paths). fp16 B=1 H=8 NAX/SDPA: N=512 **1.36× nc / 1.16× causal (SDPA wins)**, N=1024
  **1.04× / 1.03×**, N=2048 **0.98× / 1.00×** (parity), N=4096 **1.00× / 0.97×**, N=8192 **0.96× / 1.00×**.
  This is the **unique** value that removes the robust small-N regression *and* makes no N slower than the
  prior all-N NAX routing: N≤1024 NAX→SDPA is faster (N=512 nc **0.36→0.24 ms**), N≥2048 stays NAX. A
  higher threshold (4096) would make N=2048-B=1 (where NAX wins, 0.98×) slower than before. **Known
  residual** (not noise): N=4096·B=4·causal SDPA wins ~**4.6 %** (cross-session, above the ±3 % band) — a
  second-order batch interaction (more batch → NAX relatively worse at large N) that a pure-N threshold
  structurally cannot capture. It is **pre-existing** (that shape routed NAX before the threshold too — the
  threshold neither introduces nor worsens it; "no N slower than before" holds). A 2D (N,B) threshold would
  address it; **not pursued** — marginal, pre-existing, and chasing it conflicts with the simple,
  regression-free N-threshold. Lock: `tests/test_nax_routing_threshold_lock.py`
  (config fingerprint: byteΔ vs forced-SDPA = 0 ⇒ SDPA, ~1e-6 ⇒ NAX). Reproduce: `flash_attention(q,k,v)`
  at D=128, vary N around 2048.
- **Conv size-gate: evaluated, NOT needed (no-op with evidence).** conv-NAX vs `mx.conv` HW crossover at
  VAE channels (fp16, T=8, 3×3×3): NAX loses only at **HW=8** (C=128 1.09× / C=256 1.16× / C=512 1.93×),
  already wins at **HW=16** (0.68–0.87×) and 2–3× from HW=32 (0.35–0.46×). Realistic VSR decodes never
  reach HW=8: a VAE decoder's minimum spatial is its latent resolution (32×32 for a 256² output at 8×
  downscale — the smallest realistic VSR target; measured-real decodes — CogVideoX/SeedVR2 — sit at
  32×32). 32 ≫ 16 > 8 → no realistic geometry hits the losing regime → **no conv size-gate added**.
  **Caveat:** this assumes **full-frame decode** (Marco's measured pipelines). The HW=8 losing regime is
  reachable only via aggressive **small-tile decode** (a latent tile <16×16); the NO-GO is conditional on
  full-frame decode and would warrant re-evaluation if the pipeline ever moves to small spatial tiling.

### bf16 dense D=128 — 3-axis confirmation (M5 Max, `research/nax-routing-threshold-m5`, Tier-2 #2, 2026-06-19)

Closes the bf16 dense-D=128 cell on all three axes (route + perf + correctness), not by assumption.
The #1 bf16 audit confirmed *routing* (bf16→NAX) and the threshold pass measured the *crossover*
dtype-independent; this confirms **perf parity vs fp16** and **correctness vs an independent fp32 oracle**.

- **Route** (re-confirmed): bf16 D=128 dispatch matches fp16 exactly across production shapes — **N<2048 →
  SDPA** (byteΔ=0), **N≥2048 → NAX** (byteΔ 3e-5–5e-4); below-threshold bf16 SDPA path runs + correct.
- **Perf parity vs fp16** (3-session §AA.4, N≥2048, full forward incl. softmax): bf16/fp16 median ratio
  **0.951–1.017, all CONFIDENT** (cross-session range <10%). Absolute (fp16→bf16 ms): N=2048 causal
  **0.491→0.467**, N=4096 nc **1.665→1.667**, N=8192 nc **5.709→5.718**, N=8192 causal **3.330→3.338**;
  worst single-shape Δ **+0.106 ms (+3.1%)** at N=4096·B=2·H=16·causal. Parity holds across the full forward
  (softmax is fp32-accumulated regardless of input dtype). 8192-nc fp16 = 48.2 effective TFLOPS (≤51.8 ✓).
- **Correctness vs independent fp32 oracle** (manual softmax, not SDPA — Lesson #11): bf16 NAX max err
  **non-causal 1.8e-5–3.9e-5, causal 9.8e-4–1.3e-3** — **~4× coarser than fp16** (the bf16/fp16 mantissa
  ratio: 8 vs 11 bits), well inside the **bf16 floor (<1e-2)**. Below-threshold bf16 SDPA: 8.2e-4 (causal).
- Lock: `tests/test_bf16_routing_all_nax_lock.py` (route byteΔ + perf-parity ceiling 1.30× generous +
  fp32-correctness floor 5e-3). Reproduce: `flash_attention(q,k,v,causal=...)` at D=128 N≥2048, bf16 vs fp16.

### Dense-NAX online-softmax accumulator — near-optimal, NO rewrite (Tier-3, 2026-06-19)

Read-first characterization of the one structural (non-sweep) surface: the running `(O, m, l)` +
rescale logic interleaved with the MPP `matmul2d` calls in the production dense-NAX forward
(`NAAttentionKernel.cpp::createV6NAXSource`, the F-2 `v6_nax_forward` binary — confirmed live by
`MFA_V6_DUMP_SOURCE`, config BQ=64 BK=32 BD=128 WM=4). **Verdict: near-optimal, NO-GO — no rewrite.**

The leading hypothesis (current = per-block FA1 rescale → rewrite to deferred FA2) is **refuted by
reading**: the accumulator is **already FA2** — per K-block it does only the mandatory max-stability
correction `cO *= exp2(m_old−m_new)` (a *no-op* when the running max doesn't grow) + `cL` running-sum;
the **1/l division is deferred to a single epilogue** `O = cO·(1/cL)`. Accumulators (`cM/cL/cO`) are
**fp32 cooperative tensors** (NA-native, no spill — why bf16==fp16, Tier-2 #2); QK/PV are **full-tile
`matmul2d`** (multiply_accumulate). The structure follows Apple's `steel_attention_nax` reference. No
training-free structural opportunity with a real prize (the only micro-opt — guarding the no-op
`cO *= 1` — is TRIVIAL/≈zero-prize and adds a divergent branch). A valid negative, like the dV/dK
confirm. No code change. Full write-up: `.doc-archive/docs/v50/tier3-nax-accumulator-characterization.md`.

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
  beats Apple SDPA-vjp (this branch's tuning-block sub-measurement showed 1.4–2.7×; the **canonical
  live range is 2.16–3.05× — split-V6 vs SDPA-vjp, M5 Max / macOS 26.6 / MLX 0.31.2**, which subsumes this) at
  (N=8192 fp16 causal 14.3 vs 43.3 ms). Reproduce:
  `mx.grad(flash_attention(q,k,v,causal=...))` at D=64, N≥2048 (default-on). Lock:
  `tests/test_nax_backward_tuned_defaults_lock.py` (dQ tile fingerprint BK=32 + fp32-oracle grad).
- **D=64 split dV/dK BK confirmed-optimal at 32 — NO change (Tier-1 #2b).** Closing the last
  unswept backward surface: the split dV (`MFA_V6BWDV_*`) and dK (`MFA_V6BWDK_*`) kernels have
  their OWN dedicated knobs (not dQ's `MFA_V6BWD_*`, re-proved live). Legal BK = {32, 64} — `BK=16`
  throws (paired-MMA TK-even guard, Rule 8; the split kernels have no odd-TK tail). Across 6 shapes
  × {fp16,bf16} × {causal,non-causal}, `BK=64` is slower in ALL 12 cells (dV +18..+32%, dK
  +25..+69% — e.g. N=8192 fp16 dV 14.8→19.4 ms, dK 14.8→25.0 ms), and the dV gradient error is
  IDENTICAL across legal BK (perf-only, no accuracy tradeoff). So `BK=32` is optimal for both — the
  full D=64 backward split tile is now `dQ 32/32/2 + dV 64/32/4 + dK 64/32/4`, all locked
  (`tests/test_nax_backward_tuned_defaults_lock.py`: dV/dK BK=32 fingerprint + BK=16-throws guard).
- **D=128 backward NOT tuned — architectural floor confirmed.** The default D=128 backward is
  Apple SDPA-vjp (the native D=128 backward is opt-in via `MFA_ENABLE_V6_BACKWARD=1` and measured
  slower — 0.46–0.58× per the v2.50 carve-out record). Measured at the default: D=128 N=2048
  **2.71 ms**, N=4096 **9.94 ms** (= SDPA-vjp, the production choice). No headroom; not a tuning target.

v2.39.1 outcome α: H1 register pressure root-caused + fixed.  Fused
kernel default `BK` lowered 32 → 16 in Sprint v2.39.1 investigation.

> **WITHDRAWN / HISTORICAL (H-03/M5 reconciliation).** The fused-BK16 ratios below (2.00× / 1.95× /
> 1.72×, and the "1.01-1.12× fused vs split" edge) were measured on a SINCE-CORRECTED config and are
> **no longer active perf claims**. `MFA_V6_BWD_KERNEL=auto` resolves to **split for every D** (fused
> is opt-in via `=fused`), and on re-measurement fused is only **parity-with-split** at D=64 (no longer
> faster). The canonical LIVE D=64-backward claim is **2.16–3.05× (split-V6 vs SDPA-vjp, M5 Max / macOS 26.6 / MLX 0.31.2)**
> — see the `ii12_*` rows and the dQ/dV/dK tuning block above. Rows retained for provenance only.

| Claim ID | Version intro | Description | Reproduction |
|---|---|---|---|
| ~~`v2.39.1_d64_qL4096_fused_bk16_engages_via_auto`~~ | v2.39.1 (WITHDRAWN) | ~~D=64 qL=4096 V6NAX backward 2.00× vs SDPA-vjp~~ — withdrawn (fused-BK16 edge corrupt/superseded; D=64 default is now split-V6 2.16–3.05×) | historical only |
| ~~`v2.39.1_d64_qL8192_fused_bk16_engages_via_auto`~~ | v2.39.1 (WITHDRAWN) | ~~D=64 qL=8192 V6NAX backward 1.95× vs SDPA-vjp~~ — withdrawn (see above) | historical only |
| ~~`v2.39.1_d64_qL16384_fused_bk16_engages_via_auto`~~ | v2.39.1 (WITHDRAWN) | ~~D=64 qL=16384 V6NAX backward 1.72×~~ — withdrawn (see above) | historical only |

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
| `ii12_d64_qL8192_default_on_v6nax` | II-12 (2026-06) | D=64 backward (causal + non-causal) default-on via the clean V6NAX **split** kernel — **canonical live range 2.16–3.05× vs SDPA-vjp (M5 / MLX 0.31.2)** (the earlier "1.7-2.7x" was a narrower II-12 sub-range, now subsumed by the re-stamped 2.16–3.05× number). fused is opt-in (`MFA_V6_BWD_KERNEL=fused`) and only parity-with-split — no longer claimed faster. | env unset | B=1 H=4 qL=8192 D=64 fp16 | REACHABLE (default) |
| `ii12_d64_qL8192_optout_sdpa` | II-12 (2026-06) | `MFA_DISABLE_V6_BACKWARD=1` restores SDPA-vjp bit-exactly | opt-out env | Same shape | REACHABLE (opt-out) |
| `ii9_conv3d_t16_64x64_c128_fp16_mpp_default` | II-9 (2026-06; row added III-1) | conv3d via the MPP convolution2d primitive, default-on: 2.3-2.5x vs the materialized-im2col path (T8/T16 64x64 C128). **H-07 denominator note: the 2.3-2.5x is vs an INTERNAL direct-binding materialized-im2col baseline (a methodology denominator), NOT the public `MFA_DISABLE_CONV3D_MPP=1` fallback — that knob routes to `mx.conv_general` (MLX's own conv), a different denominator.** | env unset (opt-out `MFA_DISABLE_CONV3D_MPP=1` → `mx.conv_general`) | `install_hooks(); mx.conv3d(x, w)` with x `(1,16,64,64,128)` w `(128,3,3,3,128)` fp16, pad (1,1,1) | REACHABLE (default; telemetry-verified) |
| `iii1_conv3d_t16_64x64_c128_bf16_mpp_default` | III-1 (2026-06, KD-7 lift) | bf16 conv3d via MPP: 1.4-2.7x vs the pre-lift public bf16 path (Apple mx.conv3d fallback) at the II-9 cells | env unset (opt-out `MFA_DISABLE_CONV3D_MPP=1`) | Same shapes in bf16 | REACHABLE (default; telemetry-verified) |
| `iii2_tq_paged_decode_step_default` | III-2 (2026-06; re-confirmed III-12b on 26.6; reframed III-12c) | **User-facing trade-off (the headline): TQ paged decode trades ~1.4-3x decode-step latency for a ~4-5x KV-cache memory reduction at cos ~0.96, vs fp16 dense decode** (`step()` `0.75 ms vs 0.33 ms` @S=16K; KV `32 MB → ~6.5 MB` @S=8K). Opt-in (`TurboQuantPagedInferenceContext`), not auto-routed — the user chooses the trade-off. _Secondary / internal-perf history (NOT the user choice — the fused kernel is gone, so it is not a selectable baseline): the gather/dequant+SDPA path is 6.5-23x faster than the fused TQ attend kernel it replaced (`0.75 ms vs 16.8 ms` @S=16K)._ Lesson #15 + III-12c: lead with the actionable denominator (fp16 dense), not the biggest-number one. | env unset (opt-out `MFA_DISABLE_TQ_DECODE_SDPA=1`) | `TurboQuantPagedInferenceContext.step(q, k, v)` N_q=1, B=1 Hq=32 Hkv=8 D=128 tq3b; reproduce: `benchmarks/methodology/iii12b_tq_claim_26.6_run{1,2}.log` (script `tq_claim.py`) | REACHABLE (default; kernel-cache-verified) |

### conv3d-NAX causal/per-axis-pad — **RE-OPENED + RESOLVED** (`feature/conv3d-nax-asym-pad-m5`, 2026-06-18)

The #3 NO-GO below was *eligibility*, not kernel quality — and the eligibility gate turned out to be
**software, not fundamental**. "Causal support" was a misnomer: causality is handled *upstream* (the VAE
concats the time-pad before the conv), so the kernel only needed to accept **per-axis pad** (`(0,1,1)`)
instead of requiring symmetric `(1,1,1)`. Cost (measured, not presumed): **MODEST** — the MPP path's only
symmetric bake-in was the temporal `tf=t+kt−1` offset (+ `T_out==T`); parameterizing it to `−pT_left`
(+ `T_out=T−2`, host-computed) and relaxing the gate routes causal 3×3×3 to NAX. The 2D `convolution2d`
H/W spatial pad (`int2(1,1)`) is unchanged → symmetric `(1,1,1)` path **bit-identical** (keep-all-paths).
- **Causal `(0,1,1)` 3×3×3 now routes NAX, correct + faster.** Output matches an independent fp32 conv
  reference within the fp16 floor (≤2e-3; ≤3.9e-3 vs fp16 `mx.conv`). Per-shape: **1.3–2.7× vs `mx.conv`**
  on VAE-channel causal convs (512×5×32×32 **2.05 vs 3.57 ms**; 256×5×64×64 **2.19 vs 3.40 ms**).
- **MEASURED end-to-end (hardening A/B, supersedes the est. 32%).** Real-module CogVideoX decoder forward
  at real geometry (latent `[1,5,32,32,16]`→`[1,17,256,256,3]`; **random weights — timing is
  value-independent; not a weight-loaded decode**), feature-on (causal→NAX) vs feature-off (causal→`mx.conv`,
  the #3 baseline) via a test-harness gate monkeypatch: **1618 ms → 808 ms = 2.02× / −50.6 % total
  decode** (median of 10, stable across sessions). The decode-level output is equivalent on-vs-off
  (max-abs 2.93e-3 ≤ fp16 floor) — 52 composed NAX convs, no corruption.
- **Eligible fraction substantiated (settles "≈all").** Per-conv trace: 166 conv3d calls → **52 NAX
  (all causal 3×3×3) = 98.9 % of conv FLOP**; 114 fallback (1×1×1 pointwise, SpatialNorm `conv_y/conv_b`
  Cin=16<32) = 1.1 %. **No `conv_transpose3d`** — the decoder upsamples via nearest(`mx.repeat`)+conv3d,
  so the heavy convs are all eligible 3×3×3. Locked by `tests/test_conv3d_nax_asym_pad_lock.py` (21 cells:
  causal-routes-NAX + fp32-correctness + symmetric bit-identity + a 9-pad × 5-config **adversarial
  anti-corruption matrix** — every NAX route is fp32-correct, every unsupported pad/config falls back/raises).
- **Caveats:** measured on the standard CogVideoX decoder (random weights, one real tile geometry);
  SeedVR2 is structurally identical (`InflatedCausalConv3d` 3×3×3 causal → same eligibility). **Localized
  regression = 0 %** at this geometry (smallest decode spatial 32×32, above the HW≤16 losing regime; a
  size-gated route is the Tier-2 routing-threshold follow-up — only matters for sub-32 conv stages).
  1×1×1 pointwise deferred (1.1 % FLOP, bandwidth-bound). Separate feature for the version **after** 2.60.0.

#### Fleet generalization (M5, `feature/conv3d-nax-asym-pad-m5`, 2026-06-18) — by VAE lineage, not model count

Deduped Marco's VSR fleet by **verified VAE lineage** (one representative per distinct 3D-causal root;
all ports EXTRACTED, none invented). Measured live A/B (feature-on vs gate-monkeypatch-off) under the
deps venv's interpreter where pure-MLX; the **feature build was runtime-fingerprinted** each run (gate
present + causal→NAX) to defeat the sys.path shadow.

| VAE lineage | rep model | conv profile | eligible (conv FLOP) | measured end-to-end Δ |
|---|---|---|---|---|
| standard CogVideoX | DOVE | causal 3×3×3 groups=1 + 1×1×1 | 98.9 % | **2.02× / −50.6 %** (hardening) |
| Turbo-VAED-Cog (distilled CogVideoX) | SparkVSR (≡ Vivid-VR) | causal 3×3×3 g=1 + 1×1×1 (default `is_dw_conv=False`; **no** depthwise) | 86.8 % | **1.63× / −38.6 %** (163.5→100.5 ms) |
| SeedVR2 custom (`InflatedCausalConv3d`) | SeedVR2 | causal 3×3×3 g=1 | **96.8 %** | **1.83× / −45.3 %** (2136→1168 ms) ✅ measured |

- **Generalization confirmed + bounded:** the win holds across the standard and distilled CogVideoX
  lineages (1.63–2.02×); the distilled SparkVSR is **materially smaller** because its reduced **C=16**
  output stages fail the MPP C≥32 gate (15 of 37 3×3×3 convs fall back → 86.8 % eligible vs 98.9 %).
- **SeedVR2 NOW MEASURED (2026-06-18, gap closed):** the earlier "diffusers/ABI block" was a wrong-file
  diagnosis — `decoder3d_flash_seedvr2.py` is the *torch reference*; the production MLX VAE is
  `ComfyUI-SeedVR2…/src/mlx_native/mflux/.../seedvr2_vae/` and is **pure-MLX** (needed only `platformdirs`,
  no diffusers/torch), so it runs under the mlx-mfa `.venv` (py3.11) where the feature `_ext` loads
  natively — **no shared-venv mutation, no py3.14 rebuild**. Live A/B (decode `[1,16,5,32,32]`→`[1,3,17,256,256]`,
  random weights): **2136→1168 ms = 1.83× / −45.3 %**, **96.8 % conv FLOP eligible** (124/152 conv3d → NAX;
  no `conv_transpose`). Confirms the structural inference, between CogVideoX (2.02 %) and SparkVSR.
- **FlashVSR reclassified to 2D:** its MLX VAE decoder (`mlx_tcdecoder_sequential.py`) is **14 Conv2d,
  0 Conv3d** → no 3D conv-NAX surface (a 2D pixelshuffle decoder).

#### Image / 2D models (`Fooocus`, DLoRAL-SD2.1, UltraVSR-SD) — NO-GO (conv-NAX is 3D-only)

**NO-GO — now evidenced at the PRIZE level (2026-06-18), superseding the prior routing-level note.** The
prize has two factors: conv-fraction AND conv-speedup. (1) **conv-fraction is high** — an SD-VAE decoder is
conv-bound (measured: UltraVSR `MLXDecoder` 64×64×4→512×512, 59 conv2d ≈ the entire 95.7 ms decode). So
the prize *looked* real. (2) **But the conv-speedup is NEGATIVE**: a standalone MPP `convolution2d`
(the 2D primitive the 3D kernel uses per tap) is **SLOWER than `mx.conv2d` on every SD-VAE shape**
(512×64×64 **2.26 vs 0.72 ms**; 256×256×256 **2.13 vs 1.32 ms**; 128×512×512 2.32 vs 2.20 ms — `mx.conv2d`
1.0–3.2× faster).

**Root cause = Winograd (evidenced, not "maturity"; 2026-06-18).** `mx.conv2d` dispatches **3×3 stride-1**
convs to a **Winograd** kernel (`winograd_conv_2D_gpu`, `mlx/backend/metal/conv.cpp`) when
`C%32==0 && O%32==0 && N·H·W≥4096 && C+O≥256` — **every SD-VAE 3×3 shape satisfies this**. Winograd computes
F(2×2,3×3) with ≈2.25× **fewer real multiplies** than the direct count, an *algebraic* reduction the
matmul/NA `convolution2d` (which does the full multiply count) **structurally cannot access**. Confirmed by
the direct-FLOP-rate signature on M5 (FLOP convention = direct im2col `2·N·Hₒ·Wₒ·Cₒ·Cᵢ·kh·kw`): at C=O=512,
64², `mx.conv2d` **3×3 runs 0.739 ms (26.1 direct-TFLOP/s) — *faster than its own 1×1* (1.011 ms) despite 9×
the direct FLOP**, and 2× the per-direct-FLOP efficiency of **5×5** (4.12 ms, 13.0; no Winograd path). The
convs are **compute-bound** (AI≈1475 ≫ M5 ridge 145), ruling out a bandwidth explanation; the 3×3-vs-5×5
anomaly + the source dispatch rule out plain GEMM-maturity. **MLX has NO 3D Winograd** (3D conv only has
`implicit_gemm_conv_3D`) — which is *why* conv-NAX wins in 3D: there the baseline `mx.conv3d` also pays the
full multiply count, so the NA's raw throughput wins. **Generalization principle:** conv-NAX (matmul/NA, full
multiply count) wins where the baseline also does full multiplies — **conv3d, and 2D kernels Winograd can't
serve** (1×1, 5×5, non-%32, small) — and loses where the baseline has Winograd (**conv2d 3×3 s1**, the
SD-VAE/SDXL bulk). This makes the 2D-3×3 NO-GO **fundamental**: a from-scratch 2D-MPP kernel can't beat
Winograd's ≈2.25× fewer mults *unless it implements Winograd itself* (a different kernel, not the matmul2d
primitive). (FLUX DiT moot regardless — conv ≈ patch-embed.) §AA.5 premise-validation killed a MODEST-cost
route for a conv-bound prize before building. Runtime routing as-shipped: a 2D conv (4D weight) → `fallback+1`
(the hook gates on 5D-weight conv3d), so 2D models are correctly untouched.

### conv3d-NAX VAE decode profile (M5 Max, `profile/conv3d-nax-vae-m5`, 2026-06-18) — NO-GO superseded by the resolution above

Re-opened the M1-era "custom conv3d not worth it" closure for M5 (which adds the MPP `matmul2d`
NA path `mfa_conv_nax` targets). MEASURE-and-decide; **no code change**. Verdict: **NO-GO — the M1
closure HOLDS on M5, but for a NEW, M5-specific reason.** The decision rests on three measured facts
(real VAE shapes: SeedVR2 VAE `block_out_channels [128,256,512,512]`, EXTRACTED from
`results/phase1/architecture_audit.json`; CogVideoX-class VAE per DOVE `vae_cogvideox.py`):

- **(a) Eligibility = 0 % addressable (the decider).** Every production VAE conv3d is **causal**
  (`InflatedCausalConv3d` / `CogVideoXCausalConv3d`): time-padding is applied manually (concat) and
  the `mx.conv_general` call passes `padding=(0,1,1)` for 3×3×3 and `(0,0,0)` for 1×1×1. The MPP
  auto-hook gates strictly on **symmetric `pad=(1,1,1)`** (`_conv3d_mpp_eligible`, line 240) — so BOTH
  the bulk 3×3×3 causal convs AND the 1×1×1 pointwise convs **fall back to `mx.conv`** (hook telemetry
  measured: `executed+0, fallback+1` for `pad=(0,1,1)` and `pad=0`; `executed+1` only for `(1,1,1)`).
  Causal time-padding is fundamentally asymmetric → these convs can *never* present `(1,1,1)`.
- **(b) NAX beats MLX where eligible — YES (but moot).** On *would-be-eligible* symmetric 3×3×3 convs
  at VAE channels, NAX is **1.3–2.7× faster** than `mx.conv` on M5 (512×8×32×32: **3.07 ms vs 8.47 ms**;
  256×8×64×64: **3.23 vs 8.29 ms**; 128×8×128×128: **2.83 vs 8.02 ms**) — the kernel is good. It is
  simply unreachable for causal VAEs.
- **(c) Those convs are compute-bound.** Measured M5 roofline: compute **55.6 TFLOPS** (4096³ fp16
  matmul; consistent with the recalibrated ~62 TFLOPS M5 fp16/bf16 matmul peak — the older "~51.8
  effective gate" wording was a lower estimate, 55.6 is below the true ~62 ceiling), bandwidth
  **358 GB/s** (ridge ≈ 145 FLOP/byte). The VAE 3×3×3
  convs have AI 892–3749 ≫ ridge → compute-bound → a faster compute kernel *could* help if reachable.

**Decision metric** (projected end-to-end VAE-decode conv speedup = Σ ms-saved on
NAX-eligible-AND-NAX-beats layers / total conv ms) = **0 %** (eligible set empty) — far below the ~5 %
NO-GO bar. **The bottleneck is the causal-padding eligibility gate, not kernel quality or roofline.**
A GO would require adding **causal (asymmetric-time) padding support to the NAX conv kernel** — a
*kernel* project, larger than the wiring sprint a GO normally scopes; deferred, not scoped here.
Verdict is shape-INDEPENDENT (causal → ineligible for any spatial dims), so it is NOT conditional on
spatial-shape confirmation. Reproduce: hook telemetry via nested `get_hook_stats()` on a `pad=(0,1,1)`
3×3×3 conv (falls back) vs `pad=(1,1,1)` (routes). Journal:
`.doc-archive/docs/v50/conv3d-nax-vae-profile-m5.md`.

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
# M5 Max, fp16: illustrative single-run timing only (varies run-to-run). The
# canonical, stamped D=64 backward claim is 2.16–3.05× vs SDPA-vjp (split-V6,
# M5 Max / macOS 26.6 / MLX 0.31.2) — see the headline above. The older inline
# "1.81×" was a pre-re-stamp single measurement; do not cite it as the headline.
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
