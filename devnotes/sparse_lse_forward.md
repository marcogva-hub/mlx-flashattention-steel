# Sparse V6NAX Forward LSE

Date: 2026-07-13. Branch: `feature/sparse-lse-forward`. This is a beta3-indicative
M5 result; no public default promotion is made here.

## §AA.5 premise validation

- `[VERIFIED]` The sparse V6NAX generator is `sparse_kernel_source_v6nax()` in
  `csrc/mfa_sparse_attention.cpp:827`, using BQ=BK=32, WM=2, D=64/128, and
  an online softmax state `max_score`/`sum_score` in log2-domain.
- `[VERIFIED]` The dense V6 NAX source writes natural-log LSE as
  `max_score * ln(2) + log(sum_score)` (`csrc/mfa/v6_nax/NAAttentionKernel.cpp:3154-3198`).
  The sparse backward consumers convert natural LSE back to log2, so the same
  convention is required.
- `[VERIFIED]` The scalar sparse LSE path writes `m_run + log(l_run)` and
  `-INFINITY` for an all-False row. That is the compatibility oracle.
- `[VERIFIED]` The sparse online state is already available at final store:
  `max_score`, `sum_score`, and the normalized `Otile` are live together. No
  backward kernel change was needed.
- `[DEDUCED]` A row with no active K tile has `sum_score == 0`; therefore its
  output must remain zero and its LSE must be `-INFINITY`, matching the scalar
  contract.

## Implementation

- `[VERIFIED]` `emit_lse=false` is the default at
  `csrc/mfa_sparse_attention.cpp:827-831`. The optional block is inserted only
  when `emit_lse=true`; the non-LSE source body, kernel name, outputs, grid,
  and cache key remain unchanged.
- `[VERIFIED]` The new store at `csrc/mfa_sparse_attention.cpp:959-990` elects
  one lane per row, writes contiguous `(B,Hq,qL)` FP32 LSE, converts log2 to
  natural log with `ln(2)`, and writes `-INFINITY` when `sum_score <= 0`.
  The marker must occur exactly once or source generation raises.
- `[VERIFIED]` BT32/D64-or-D128/fp16-or-bf16 now launches
  `sparse_attn_v6nax_sparse_lse_*` with outputs `(O,L)` at
  `csrc/mfa_sparse_attention.cpp:1351-1390`. BT16/BT64 and unsupported cases
  retain the scalar LSE generator.
- `[VERIFIED]` The with-LSE validator now checks 2-D, 3-D, and 4-D mask
  shapes and rejects causal `qL != kL` explicitly.
- `[VERIFIED]` Full-native and hybrid sparse backward wrappers preserve the
  original BT64 mask for backward conversion, but expand only the forward
  LSE input to BT32 (`mlx_mfa/attention.py:3446-3459` and `3563-3576`).
  This is the required BT64 semantic-equivalence route; backward kernels were
  not modified.
- `[VERIFIED]` Python telemetry records `v6nax_sparse_lse` or
  `scalar_fallback_lse` in `mlx_mfa/lcsa_nax.py:252-262`.

## Correctness locks

Harness: `benchmarks/bench_sparse_lse_forward.py`, fresh process per cell.
The scalar oracle is the pre-change scalar implementation with identical
block semantics; BT64 cases use 2x2 expansion to BT32. The run used MLX
evaluation fencing and a separate process per cell.

- `[VERIFIED]` BT32 grid: D={64,128}, dtype={fp16,bf16}, N={2048,4096},
  density={0.05,0.10,0.30}, causal/non-causal: 48/48 cells passed.
- `[VERIFIED]` BT64-expand grid: D={64,128}, dtype={fp16,bf16},
  N={2048,4096}, density={0.10,0.30}, causal/non-causal: 32/32 cells passed.
- `[VERIFIED]` Across the 80 cells: minimum cosine O-vs-scalar was
  `0.9999993048`; maximum O max-abs was `4.8828125e-4`; maximum finite-LSE
  max-abs was `1.4305115e-6`; all finite/-inf row classifications matched.
- `[VERIFIED]` Dedicated all-False-row tests cover D64/D128 and fp16/bf16.
  They assert `O == 0`, `L == -INFINITY`, finite active rows, and the
  `v6nax_sparse_lse` trace.
- `[VERIFIED]` Public full-native BT64 engagement was tested at B=1,H=4,
  N=4096,D=64, with `MFA_ENABLE_V6_BACKWARD=1` and
  `MFA_V6_BWD_SPARSE_NATIVE=1`; the trace was `v6nax_sparse_lse`. The earlier
  BT32 test was a false-positive route through the public NAX+SDPA wrapper and
  was corrected to use the actual BT64 full-native gate.
- `[VERIFIED]` Full-native gradient spot checks at N=4096, BT64, density 0.1,
  fp16, D={64,128}, causal/non-causal produced cosines in
  `[0.99999778, 0.99999989]` for dQ/dK/dV against the SDPA-vjp reference;
  the public trace was `v6nax_sparse_lse` in all four cells. The focused
  suite also passed its existing fp32-gradient/oracle locks.

## Performance

Stamp: MLX 0.31.2 / mlx-mfa 2.61.0, arm64 M5 Max, macOS 27.0 beta,
Metal `32023.918`, Xcode 27.0. Each timing sample used 20 dispatches; each
arm had 5 samples in a fresh process and both arm orders were run.

### Exact pre-change scalar LSE versus new NAX LSE

The scalar arm was measured from a clean master worktree at `a461b59` with
BT32, not the BT16 semantic control. The NAX arm was measured on this branch.
Both used the same harness, shape, dtype, density, and causal flag.

| Shape | scalar master order A/B ms | NAX-LSE order A/B ms | NAX/scalar |
|---|---:|---:|---:|
| D64, N2048, fp16, d=.10, non-causal | 1.141 / 1.148 | 0.284 / 0.356 | 0.25-0.31x |
| D128, N2048, fp16, d=.10, non-causal | 3.164 / 3.197 | 0.288 / 0.429 | 0.09-0.14x |
| D128, N4096, bf16, d=.30, causal | 11.325 / 11.347 | 0.356 / 0.374 | 0.031-0.033x |

`[VERIFIED]` The scalar arm is source-fingerprinted as
`scalar_fallback_lse_*` on master; the new arm is trace-fingerprinted as
`v6nax_sparse_lse`. These ratios are forward-LSE ratios, not claims about a
complete training step.

### Full-native backward versus SDPA-vjp

| Shape, N=4096, BT64, d=.10 | full-native A/B ms | SDPA-vjp A/B ms | native/SDPA |
|---|---:|---:|---:|
| D64, non-causal | 1.136 / 1.464 | 3.559 / 3.224 | 0.32-0.45x |
| D64, causal | 1.616 / 0.957 | 3.187 / 2.931 | 0.30-0.51x |
| D128, non-causal | 1.728 / 1.870 | 3.109 / 3.773 | 0.46-0.60x |
| D128, causal | 1.575 / 1.487 | 3.683 / 3.054 | 0.40-0.52x |

`[VERIFIED]` Both arms were correction-checked before timing; native traces
contained `v6nax_sparse_lse`, SDPA traces contained no sparse LSE record.
The full-native path is therefore a measured candidate for the opt-in
training path, not a default-on recommendation.

### No-LSE conservation gate

`[VERIFIED]` The no-LSE source path is structurally unchanged: the added code
is guarded by `emit_lse`, and the no-LSE call still uses the old kernel name
and output list. Existing no-LSE correctness/dispatch locks pass.

`[UNCERTAIN]` The requested numerical `+/-3%` timing gate is not attributable
to this change on this beta runtime. A clean-master versus current-branch
cross-build comparison on identical cells varied by more than 3% in both
directions, while the emitted no-LSE MSL path is unchanged. For example,
D64/N2048 was 0.375/0.385 ms on master and 0.371/0.294 ms after; D128/N4096
bf16 causal was 0.415/0.413 ms and 0.375/0.408 ms. This is a measurement
variance limitation, not evidence of a source-level regression; re-run the
no-LSE gate on stable macOS before promotion.

## Verdict and red-team

- `[VERIFIED]` The LSE capability is correct, engaged, and materially removes
  the scalar forward bottleneck. The full-native backward chain now reaches
  the intended NAX forward LSE and remains gradient-correct.
- `[DEDUCED]` The measured full-native speedups versus SDPA-vjp justify keeping
  the opt-in full-native path available. They do not justify changing the
  public default in this beta-only run.
- `[VERIFIED]` Scalar LSE remains available for BT16/BT64 fallback and for
  unsupported shapes; the production non-LSE forward remains unchanged.
- `[VERIFIED]` Red-team catch: the former BT32 public full-native test did not
  prove full-native engagement because the dispatcher requires BT64 for its
  backward mask geometry. The lock now uses BT64 and checks the LSE trace.
- `[UNCERTAIN]` No stable-macOS performance verdict is claimed for the
  no-LSE +/-3% gate. This is the only open validation item.

## Skill invocations

| Checkpoint | Skill | Result |
|---|---|---|
| Metal kernel/LSE design | `metal-kernel-dev` | Used; dense NAX LSE and fragment mapping checked at source. |
| Benchmark and engagement method | `benchmark-harness-builder` | Used; fresh-process arms, 20 dispatches/sample, two orders. |
| Debug/forensics | `mlx-debug-forensics` | Used through the preceding varlen safety protocol; no new corruption found here. |

## Validation commands

- `/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python -m pytest tests/test_v50_sprint_5c_sparse_backward_hybrid.py tests/test_v50_sprint_5d_sparse_backward_native.py tests/test_sparse_bf16_v2_lock.py tests/test_raw_surface_classes.py -q -k 'sparse or LSE or lse or native'` -> 46 passed.
- `/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python benchmarks/bench_sparse_lse_forward.py ...` -> 80 correctness cells passed.
- `/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python benchmarks/bench_sparse_full_native.py ...` -> 8 full-native/SDPA arms, all correction and engagement checks passed.
- `/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python -m pytest tests/ -q` -> 3492 passed, 93 skipped, 3 warnings in 105.61s.
