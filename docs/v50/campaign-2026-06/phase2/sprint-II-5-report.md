# Sprint II-5 — Deep Literature Autoresearch (2026-06-12)

**Status**: COMPLETE.
**Headline**: the Sprint II-2 int8 DECLINE is **FALSIFIED** — MPP int8
matmul2d IS implemented on macOS 26.4/M5 and sustains **2.00× fp16**
(264.9 TOPS vs 132.6 TF, in-repo microbench). The Sage-NAX int8 kernel
sprint premise is **REVIVED**. Secondary discovery: an undocumented
**MPP `convolution2d` primitive** exists in the 26.4 SDK, runs, and is
deterministic — a potential Apple-primitive replacement for the
Marco-gated fused-im2col XL candidate.

## Method

Three parallel web-research agents (families 1+2 / 3+4+5 / 6+7+discovery),
2025–2026 primary sources only, every candidate filtered against the
Pattern #6 reality (Apple SDPA NAX unbeaten on dense forward; M5
dispatch map authoritative; known-dead ledger). Survivors were
premise-probed locally on M5 Max / macOS 26.4 / MLX 0.31.2 this sprint
(§AA.5 protocol). 28 techniques assessed; 21 declined with reason;
7 survived to probe/disposition.

## Consolidated verdict table

| Family | Technique (source) | Claimed gain / baseline | Verdict |
|---|---|---|---|
| 1 | FlashAttention-4 (Tri Dao, arXiv 2603.05451) | 1.1–1.3× vs cuDNN on B200 | DIE — gains are Blackwell pipelining (TMEM/UMMA); no Metal counterpart. Rescale-skip portable bit is noise on Apple same-pipe SIMD [deduced] |
| 1 | FlashMLA / DeepSeek-V3.2 sparse (deepseek-ai) | 3000 GB/s on H800 | DIE — Hopper-specific; MLA needs MLA checkpoints (model property) |
| 1 | Native Sparse Attention (arXiv 2502.11089) | 9× fwd at 64K vs FA2 | DIE — needs NSA-pretrained models; selection substrate = repo top-K (built, declined 0.15×, II-3) |
| 1 | Sparse VideoGen 1/2 (2502.01776, 2505.18875) | 2.3× on HunyuanVideo, H100 | DIE — SVG2 needs k-means+gather (scalar-grade on M5) + diffusion-layer integration (out of scope); SVG1 static masks already expressible via repo block-sparse/GNA |
| 1 | LeanAttention (2405.10480) | 2.6–8.3× decode vs FA2-style | DIE — functionally = repo's Track-H split-KV + LSE reduce |
| 1 | MFA v2.5 NAX (Draw Things 2025-11) | 3.6–5.5× vs M4 | DIE as competitor — same class as Apple SDPA NAX (dense fwd closed); external-repo integration out of scope. Evidence base only |
| 2 | SageAttention3 (NeurIPS'25) | 2–5× via FP4 on Blackwell | DIE — Metal has no fp4/fp8 types (until macOS 27 per WWDC26 #330) |
| 2 | SageAttention2/2++ | 2.7–5.1× via INT4-QK+FP8-PV | DIE — needs int4 tensor cores + fp8; no variant avoids quantized-matmul HW |
| 2 | **Metal Quantized Attention (Draw Things 2026-03)** | **1.24–1.41× vs their fp16 NAX attention on M5** | **SURVIVOR → int8 revival (below)** — design blueprint: QK row-group scale, V row-wise affine, int8 NAX MMA, fused dequant epilogue |
| 2 | INT-FlashAttention (2409.16997) | full-int8 FA fwd, Ampere | Dataflow template for the revived sprint; not standalone |
| 2 | KVLinC / RotateKV / KITTY / RateQuant (KV-quant 2025-26) | quality at 2–4 bit, not speed | DIE — TurboQuant occupies this slot; KVLinC linear-correction noted as future quality option |
| 3 | **cider GQA decode kernel (Mininglamp-AI, MIT)** | 1.04–1.57× vs mx.fast.sdpa (M5 Pro) | **SURVIVOR — premise-benched here: CONFIRMED-NARROW** (below) |
| 3 | vLLM RPA v3 fused KV-append (2025-10) | hides scatter latency (TPU) | SURVIVOR (idea) — queued behind II-7 premise measurement (append cost share on real decode steps); kill-bench <8% step win |
| 3 | FlashInfer BSR/JIT (MLSys'25) | 29–69% latency cuts (CUDA) | DIE — repo already owns block-sparse+JIT+paged; GQA-scheduling piece captured by cider survivor |
| 3 | llama.cpp Metal FA decode | — | DIE — no evidence it beats mx.fast sdpa_vector; M5 dense-decode gap closed (II-1 map) |
| 3 | mlx-qsdpa fused dequant decode | 1.7× vs two-call quant path; 0.52–0.78× of fp16 | DIE — TurboQuant supersedes; keep as competitive bench target |
| 3 | TQ-fork sparse-V dequant skip (ggml #20969) | +22.8% @32K, lossy | SURVIVOR (opt-in) — queued behind TQ quality harness; gate `MFA_TQ_SPARSE_V`; kill: quality budget or <10% |
| 3 | DFlash/DeFT/MTP speculative layouts | DFlash 1.34–4.37× on M5 Max | DIE for kernel work — routes through standard SDPA; llama.cpp MTP = net loss on Metal (issue #23752) |
| 3 | NHD vs HND KV layout | no single-device delta found | DIE — distributed-serving motivation; layout churn ≫ benefit |
| 4 | MLX in-tree `implicit_gemm_conv_3D_gpu` | prior art | INFO — de-risks fused-im2col design (steel/conv loader); pad-to-16 interim inapplicable (repo channels already %16==0, and repo conv3d NAX already beats MLX's implicit path 1.6–2.3×) [deduced from II-4 data] |
| 4 | **MPP `convolution2d` (26.4 SDK, undocumented)** | unknown | **NEW SURVIVOR — probed here (below)** |
| 4 | NOVA Winograd points (2512.18453) | accuracy recovery fp16 F(8,3) | DIE for now — removes the *numerics* half of the Phase-I Winograd decline; utilization-loss half stands. Citation recorded if ever revisited |
| 4 | Turbo-VAED depthwise-sep distill | 84.5× via retraining | DIE — model-side (retraining), out of scope |
| 5 | FlashDecoding++ unified-max softmax (MLSys'24) | 1.14× decode (CUDA) | SURVIVOR (small) — applies only to repo-owned split-K kernels; queued with kill <5% / NaN-adversarial test; fold into the decode sprint if cider port proceeds |
| 5 | Kahan in attention accumulators | no speed-at-accuracy win exists | DIE — SageAttention measured fp16-accum lossless; Kahan ≈4× add cost |
| 5 | Stochastic rounding on Apple GPU | no HW support | DIE — software-PRNG only, training-oriented |
| 5 | Batch-invariant deterministic kernels (Thinking Machines 2025-09) | feature, ~single-digit % cost | SURVIVOR (feature) — `MFA_DETERMINISTIC=1` fixed-split decode; **routed to Sprint II-6** (numerics) where it belongs |
| 6 | MTL4 ML command encoders (WWDC25) | model-graph execution | DIE — app-developer surface; MLX manages own encoders |
| 6 | WWDC26 #330 TensorOps additions | coop-tensors-as-inputs, reduce_rows, int4/int8 types | CONFIRMED PRESENT in 26.4 SDK (`reduce_rows` MPPTensorOpsMatMul2d.h:588, `map_iterator` :613) — feeds the int8 kernel design |
| 7 | Fused QK-norm / RoPE+attn / attn+proj fusion | various CUDA wins | DIE — must beat two NAX-grade calls; same falsified-premise shape as the fused-RoPE decline (mx.fast.rope+SDPA won 4×) |
| 7 | Mirage MPK megakernels (2512.22219) | 1.2–6.7× vs multi-launch CUDA | DIE — no Metal persistent-kernel/grid-sync model; MLX command-buffer batching amortizes launches |
| D | SpargeAttn (ICML'25) dynamic block prediction | 4–7× vs dense FA (CUDA) | SURVIVOR (conditional) — queued behind II-7: needs real VSR attention maps to know achievable sparsity; on M5 break-even sparsity is far higher than CUDA baselines; predictor feeds existing `flash_attention_sparse` (no new kernel for prototype) |
| D | Attention sinks (gpt-oss family) | feature | SURVIVOR (feature) — removes `patch_mlx_lm` sinks fallback; Marco priority call |
| D | StreamingLLM eviction | memory | DIE as perf; optional HybridKVCache feature |

## Probe 1 — int8 MPP matmul2d revival (II-2 FALSIFIED)

**Trigger**: two independent web-verified production implementations
contradict II-2's "unimplemented" verdict — Draw Things Metal Quantized
Attention (1.24–1.41× over their fp16 NAX attention) and
Mininglamp-AI/cider (W8A8 via `matmul2d(16,32,16)`, MIT, MLX custom
primitives, same nanobind/MLX stack as this repo).

**Root cause of the II-2 false negative** [VERIFIED]: the MPP header
static_asserts full-cooperative int8 to fragment dims **M,N,K ∈ {16,32}**
(`MPPTensorOpsMatMul2dImpl.h:4249-4252`). II-2 probed only 64×64×128
tiles; the resulting "Unsupported type" diagnostic pointed at dtype
rather than dims. Working form:

```metal
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto a = op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
// element-wise register fill; no .load() from device tensor<>
```

**Evidence chain** (all on M5 Max, macOS 26.4, runtime-compiled MSL4):
1. Standalone probe: compiles, pipeline OK, **bit-exact 16×32×16 int8
   GEMM vs CPU reference**.
2. Dims isolation: same form at 64×64×128 static_asserts (the actual
   II-2 failure); device-tensor operands at (16,32,16) also compile.
3. In-repo `mpp_int8_microbench()` (commit `c480c51`), apples-to-apples
   at the cider form: **fp16 132.6 TF, int8/i32 264.9 TOPS = 2.00×**.
   Legacy 64×64×128 fp16 path unchanged at 134.7 TF (small-tile form
   sacrifices nothing).

**Kill gate (II-2, ≥1.3×): PASS at 2.00×.**

**Disposition**: Sage-NAX int8 attention kernel sprint REVIVED. Marco
originally greenlit this build (Phase II orchestration); the decline
was gate-driven and the gate is reversed. Recommended as the next
kernel sprint (after II-6/II-7 or interleaved — Marco's sequencing
call). Design inputs now available: DT blueprint (QK row-group-wise
scale, V row-wise affine, fused dequant epilogue), INT-FlashAttention
dataflow, cider's working MSL (MIT), WWDC26 reduce_rows/map_iterator
present in 26.4 SDK. Quality risk real: DT published no quantitative
quality metrics; three-axis gate must include cos-sim + E2E checks.
Note: the ledger's "re-run probe after macOS updates" trigger was
Marco-gated; this re-run was triggered instead by new external evidence
under II-5's explicit prototype authorization.

## Probe 2 — MPP `convolution2d` (NEW, undocumented primitive)

`MPPTensorOpsConvolution2d.h` ships in the 26.4 SDK (NHWC activation,
HWIO weights, groups=1, strides/dilations, multiply_accumulate mode,
cooperative destination). Not referenced in any campaign doc until now.

**Probe results** (M5 Max, runtime MSL4):
- Compiles, pipeline OK, executes, **deterministic** (0/576 diff across
  runs).
- Impulse-response test identifies the convention [VERIFIED]:
  **centered cross-correlation** — `D[y,x,o] = Σ A[y+a−1, x+b−1, c] ·
  W[a,b,c,o]` with zero-pad at bounds (pad=(K−1)/2 origin), extents
  fastest-first, scope `metal::execution_simdgroups<N>` (full TG).
- **Unresolved**: multi-threadgroup tiling semantics. Descriptor-as-tile
  + `set_offsets` hypothesis produced wrong results; needs WWDC26 #330
  sample code or impl-header reading (est. ½–1 day).

**Why it matters**: `multiply_accumulate` mode means conv3d decomposes
as kT accumulated conv2d calls — an Apple-primitive path that could
supersede the Marco-gated fused-im2col XL candidate (2.6× ceiling on
K=3456 shapes) at far lower effort, eliminating the materialized im2col
(62% of small-K time, II-4). Classic §AA.5: prefer the Apple primitive.
**Disposition**: follow-up probe queued (tiling resolution + throughput
vs repo conv3d NAX + Pattern-#6 bench vs MLX conv3d). Recommended
before any fused-im2col work.

## Probe 3 — cider GQA decode kernel (premise: CONFIRMED-NARROW)

Built cider from source (isolated /tmp venv, M5 Max, MLX 0.31.2) and
ran its own correctness+perf bench vs `mx.fast.sdpa`:
- Correctness: PASS everywhere (max diff ≤1.2e-4, fp16 grade).
- Perf: **70 wins / 29 ties / 3 losses over 102 decode configs**, but
  magnitudes 1.00–1.24× — best 1.22–1.24× at GQA-factor 8–16,
  N=16K–32K; ≈1.0× at MHA, low GQA, or short S. The README's 1.57×
  does NOT reproduce here: MLX 0.31.2's improved vector fused-GQA SDPA
  moved the baseline (their numbers were M5 Pro, likely older MLX).

**Disposition**: technique is real but the window is narrow
(high-GQA long-context LLM decode; ≤1.24× ceiling on current MLX).
Repo integration would be an M-effort port into the Track-H
flash-decode kernel (grid `(NQ·splits, H_kv, B)` + per-thread Q-head
tiling), also portable into paged/TQ decode kernels where Apple has no
kernel (compounding). **Presented for Marco's call** — Pattern #6 says
the numbers are the argument, and 1.0–1.1× across most of the grid is
thin; the paged/TQ transplant is the stronger half of the case.

## Dispositions summary (NO-DEFERRAL accounting)

| Item | Disposition |
|---|---|
| int8 revival | PROBED + benched + committed (`c480c51`); kernel sprint revived, sequencing = Marco |
| MPP convolution2d | PROBED (3 iterations); tiling follow-up queued with est. + kill-bench |
| cider GQA decode | PREMISE-BENCHED on-device; port = Marco call (numbers above) |
| Fused KV-append | Queued behind II-7 measurement (append share of step time) — sequencing, not deferral |
| Sparse-V skip | Queued behind TQ quality harness; env-gated design recorded |
| Unified-max softmax | Folded into decode-sprint scope if port proceeds; kill <5% |
| Deterministic mode | Routed to II-6 (numerics sprint — natural owner) |
| SpargeAttn | Queued behind II-7 real-workload attention maps (premise data) |
| Attention sinks | Feature ledger (not perf); removes a documented fallback |
| All 21 DIEs | Declined with reason + citation in table |

## Skill invocations (§AA.2)

| Skill | When | Result |
|---|---|---|
| `/mlx-mfa-apple-primitives-coverage` (protocol) | int8 + conv2d + decode probes | §AA.5 executed literally: primitive inventory → decomposition → on-device candidate bench → verdicts (FALSIFICATION, NEW-PRIMITIVE, CONFIRMED-NARROW) |
| `/mlx-mfa-bench-methodology` (protocol) | microbenches | GPU-time medians, warmup discarded, multi-iteration; sub-ms cells use rep-amortized dispatch |
| `/mlx-mfa-perf-audit` | deferred | No public perf claim ships this sprint (all numbers are internal sprint evidence); mandatory before any release-note claim from the revived int8 sprint |

## Sources

Primary sources cited inline in the table; key links:
[DT Metal Quantized Attention](https://releases.drawthings.ai/p/metal-quantized-attention-pulling) ·
[DT MFA v2.5 NAX](https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural) ·
[Mininglamp-AI/cider](https://github.com/Mininglamp-AI/cider) ·
[WWDC26 #330](https://developer.apple.com/videos/play/wwdc2026/330/) ·
[FA4 blog](https://tridao.me/blog/2026/flash4/) ·
[SpargeAttn](https://arxiv.org/abs/2502.18137) ·
[Thinking Machines determinism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) ·
[FlashDecoding++](https://arxiv.org/abs/2311.01282) ·
[NOVA Winograd](https://arxiv.org/abs/2512.18453) ·
[MLX conv.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/conv.cpp) ·
[Apple ML Research MLX-on-M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
