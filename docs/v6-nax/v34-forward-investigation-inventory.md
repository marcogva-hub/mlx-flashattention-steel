# V34 forward investigation — inventory

## Foundation
- master tip: `d4a876a` (CLAUDE_V6_NAX.md §3.5 + §4.X live)
- v2.35.0 production code, V34 forward shipped in v2.32.0 (2026-05-06)
- Memory #30 roadmap: this investigation → V34 backward Option β

## V34 vs predecessor dispatch mechanism

V34 has explicit env-var dispatch infrastructure in
`csrc/mfa_v6_nax_primitive.cpp:117-125`:

| Env var | Purpose | Default |
|---|---|---|
| `MFA_V6_USE_V34` | V34 vs predecessor path | 1 for D=128, 0 otherwise (source-gen-default) |
| `MFA_V6_BLOCK_R` | parallelization (rows per simdgroup) | 32 |
| `MFA_V6_BLOCK_C` | traversal block (K cols) | 32 |
| `MFA_V6_EXEC_SG` | simdgroups per threadgroup | 4 |
| `MFA_V6_BLOCK_D` | head sub-tile | head_dim |
| `MFA_V6_BYPASS_TGP` | Path A (cooperative→cooperative) | 0 |
| `MFA_V6_NAX_SINGLE_OTILE` | single Otile mode | 0 (V34 requires 1) |

This investigation uses these env knobs to isolate each hypothesis rather
than building dedicated variant source-gen functions. The mechanistic
attribution is identical; implementation cost is dramatically lower.

## Source-level pre-eval per hypothesis

| Hypothesis | Source-level finding (`csrc/mfa/v6_nax/NAAttentionKernel.cpp`) | Status |
|---|---|---|
| A — TGP occupancy | V34 uses `[[kernel, max_total_threads_per_threadgroup(V34_WM * 32)]]` (line 2758). WM configurable via `MFA_V6_EXEC_SG`. | Probe via env |
| B — cross-SG sync elim | V34 K-loop uses only `simdgroup_barrier(mem_none)` (line 2906; intra-SG, lightweight). Predecessors use `threadgroup_barrier(mem_threadgroup)` (lines 1059, 1290; cross-SG, heavyweight). | **STRUCTURAL CONFIRMED** |
| C — simd_shuffle_xor vs MPP reduce | V34 line 2889: `Stile.template row_reduce<MaxOp>(...)` → NAXFrag::row_reduce internally uses `simd_shuffle_xor` (line 2546). Predecessors lines 931/1011/1178/1259: `reduce_rows(cS_0, cM_0_new, reduction_operation::max, ...)` → MPP cooperative-tensor reduce. | **STRUCTURAL CONFIRMED** |
| D — register pressure | V34 uses NAXFrag fragments (16×16 chunks) with explicit tile shape control. Predecessors use `matmul2d<exec_simdgroups<1>>` constraint forcing more state in registers. | Probe via `MFA_V6_BLOCK_R` |
| E — Apple defaults mis-tuned | V34 uses explicit BQ/BK/WM defaults (32/32/2 for D=64; 64/32/4 for D=128). Predecessor inherits Apple's MPP autotune. | Probe via env (compare to BQ/BK=32/32 generic) |

3 of 5 hypotheses are structurally confirmed by source reading. Confirmation
of A, D, E + magnitude attribution for all 5 requires the env-var benches.

## Investigation shapes

| Shape | qL | kL | nh_q | nh_k | D | expected wall-clock |
|---|---:|---:|---:|---:|---:|---:|
| v34_small_d64  |  1024 |  1024 |  8 |  8 |  64 | ~1-2ms (frontière) |
| v34_small_d128 |  1024 |  1024 |  8 |  8 | 128 | ~2-3ms |
| v34_mid_d128   |  4096 |  4096 | 16 | 16 | 128 | ~20-30ms |
| v34_large_d128 |  8192 |  8192 | 16 | 16 | 128 | ~80-120ms |

Per prompt §B.2, I'll use 4 representative shapes (small_d64 boundary + small_d128 +
mid_d128 + large_d128). The D=64 mid/large shapes are bundled with the D=128
counterparts for cross-D delta (handled via V34's `head_dim == 128` source-gen
default — D=64 path is the predecessor by default).

For shapes potentially sub-1.5ms (small_d64): apply §4.X caveat in results.

## New artifacts

| File | Purpose |
|---|---|
| `bench/v34_forward_investigation_harness.py` | Unified §4-strict harness with hypothesis env toggle |
| `docs/v6-nax/v34-forward-investigation-{inventory,decisions}.md` | This file + decisions |
| `docs/v6-nax/v34-forward-mechanisms.md` | Synthesis (Section H) |
| `docs/v6-nax/v34-forward-investigation-data.json` | Raw bench data |
| `docs/v6-nax/v34-backward-option-beta-design-hints.md` | Section I |
| `devnotes/SESSION_LOG.md` entry | Sprint log |

## Hardware + environment
- M5 Max 128GB, macOS 26.5, iStat performance fan profile
- MLX 0.31.2, mlx_mfa 2.35.0
