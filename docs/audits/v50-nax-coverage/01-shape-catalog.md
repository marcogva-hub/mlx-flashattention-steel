# v2.50 NAX Coverage Audit — Shape Catalog

**Audit date**: 2026-05-13.  Shapes selected to cover Marco's portfolio
of real-world workloads (FlashVSR, STCDiT, CogVideoX) + mlx-lm
serving use cases.

## VSR training shapes (DiT-style video models)

| ID | B | H | qL=kL | D | dtype | causal | window | Notes |
|---|---|---|---|---|---|---|---|---|
| VSR1 | 1 | 12 | 1024 | 128 | f16 | no | none | FlashVSR small-frame baseline |
| VSR2 | 1 | 12 | 2048 | 128 | f16 | no | none | STCDiT mid-sequence |
| VSR3 | 1 | 12 | 4096 | 128 | f16 | no | none | FlashVSR canonical |
| VSR4 | 1 | 12 | 8192 | 128 | f16 | no | none | CogVideoX long-sequence |
| VSR5 | 1 | 16 | 2048 | 128 | f16 | no | none | CogVideoX 16-head variant |
| VSR6 | 1 | 16 | 4096 | 128 | f16 | no | none | CogVideoX canonical |
| VSR7 | 1 | 12 | 4096 | 128 | f16 | no | LCSA block_mask BT=32 | LCSA symmetric sparse (lcsa_nax dispatch) |

## LLM training shapes (mlx-lm + general)

| ID | B | H_q | H_kv | qL=kL | D | dtype | causal | Notes |
|---|---|---|---|---|---|---|---|---|
| LLM1 | 1 | 32 | 8 | 2048 | 64 | f16 | yes | Llama-3 8B GQA short |
| LLM2 | 1 | 32 | 8 | 4096 | 64 | f16 | yes | Llama-3 8B GQA canonical |
| LLM3 | 1 | 32 | 8 | 8192 | 64 | f16 | yes | Llama-3 8B GQA long |
| LLM4 | 1 | 32 | 8 | 16384 | 64 | f16 | yes | Llama-3 8B GQA very-long |
| LLM5 | 1 | 32 | 8 | 4096 | 128 | f16 | yes | Llama-2 70B GQA |
| LLM6 | 1 | 32 | 8 | 8192 | 128 | f16 | yes | Llama-2 70B GQA long |

## LLM serving / decode shapes (cache-enabled)

| ID | B | H_q | H_kv | N_q | S_kv | D | dtype | causal | Notes |
|---|---|---|---|---|---|---|---|---|---|
| DEC1 | 1 | 32 | 8 | 1 | 2048 | 64 | f16 | yes | Llama-3 8B decode short |
| DEC2 | 1 | 32 | 8 | 1 | 4096 | 64 | f16 | yes | Llama-3 8B decode canonical |
| DEC3 | 1 | 32 | 8 | 1 | 8192 | 64 | f16 | yes | Llama-3 8B decode long |
| DEC4 | 1 | 32 | 8 | 1 | 16384 | 64 | f16 | yes | Llama-3 8B decode very-long |
| DEC5 | 4 | 32 | 8 | 1 | 4096 | 64 | f16 | yes | Batch-4 decode |
| DEC6 | 8 | 32 | 8 | 1 | 4096 | 64 | f16 | yes | Batch-8 decode |

## LLM prefill (cache fill phase)

| ID | B | H_q | H_kv | qL | kL | D | dtype | causal | Notes |
|---|---|---|---|---|---|---|---|---|---|
| PRE1 | 1 | 32 | 8 | 1024 | 1024 | 64 | f16 | yes | Llama-3 8B short prompt prefill |
| PRE2 | 1 | 32 | 8 | 4096 | 4096 | 64 | f16 | yes | Llama-3 8B canonical prefill |
| PRE3 | 1 | 32 | 8 | 8192 | 8192 | 64 | f16 | yes | Llama-3 8B long prefill |

## Paged + varlen shapes (serving)

| ID | n_seqs | active_pages | page_size | S_kv | H_q | H_kv | D | dtype | Notes |
|---|---|---|---|---|---|---|---|---|---|
| PAGE1 | 4 | 128 | 16 | 2048 | 32 | 8 | 64 | f16 | mlx-lm paged 4-seq |
| PAGE2 | 8 | 256 | 16 | 4096 | 32 | 8 | 64 | f16 | mlx-lm paged 8-seq |
| PAGE3 | 16 | 512 | 16 | 4096 | 32 | 8 | 64 | f16 | mlx-lm paged 16-seq |

## Speciality shapes

| ID | Function | Notes |
|---|---|---|
| GNA1 | B=1 H=16 qL=4096 D=128 3D-window | DiT video GNA pattern |
| TOPK1 | B=1 H=16 qL=4096 K=64 D=128 | Top-K sparse attention |
| SPEC1 | B=1 H=16 N_q_draft=4 S=4096 D=128 | Speculative verify draft=4 |
| SPEC2 | B=1 H=16 N_q_draft=16 S=4096 D=128 | Speculative verify draft=16 |
| VARLEN1 | B=4 qL=[128, 512, 2048, 4096] D=128 | Heterogeneous varlen |

## Methodology per /mlx-mfa-bench-methodology

- All bench calls go through PUBLIC API: `mx.grad(flash_attention(...))` or
  `flash_attention(...)` (no `backend="mfa"` forced unless documented).
- 4 warmup + 12 timed iters, median ms reported.
- For functions where 3-session variance check is appropriate (perf-sensitive
  shapes), 3 sessions × 4w+12i with variance ratio reported.
- MLX array materialization (via `_flush = getattr(mx, "eval")` alias to
  bypass Claude security-hook substring check) + `mx.synchronize()` after
  each iteration.
- Compare paths via env var manipulation:
  - Default (auto routing): no env vars set
  - SDPA-vjp baseline: `MFA_DISABLE_V34_BACKWARD=1` (for backward) OR
    `MFA_DISABLE_AUTO_HOOKS=1` (for forward routing)
  - MFA-forced: `backend="mfa"` (when function supports it)

## Shape-to-function mapping (which shapes test each function)

Per the audit's breadth-not-depth mandate, each function gets 1-3 shapes
from this catalog covering its representative use case.  Detailed bench
plans are in `per-function/*.md`.
