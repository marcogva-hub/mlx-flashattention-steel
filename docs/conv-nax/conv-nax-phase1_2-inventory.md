# Phase 1.2 — File + Binding Inventory

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_2` (branched from `experiment/conv-nax-phase1_1` tip)
**Scope:** up1_resnet (multi-chunk) + asymmetric/causal pad_T + K_T=1 routing

## Files modified

| Path | Δ lines | Purpose |
|------|--------:|---------|
| `mlx_mfa/conv_nax.py`     | +168 / -100 | M-chunking + asymmetric padding + causal_pad_t flag |
| `tests/test_conv_nax.py`  | +180 | 7 new Phase 1.2 tests (up1 + causal pad_T + K_T=1) |

## Files added

| Path | Lines | Purpose |
|------|------:|---------|
| `docs/conv-nax/conv-nax-phase1_2-inventory.md`  | this | This file |
| `docs/conv-nax/conv-nax-phase1_2-decisions.md`  | TBD | D-numbered Phase 1.2 decisions |
| `docs/conv-nax/conv-nax-phase1_2-results.md`    | TBD | Correctness + NaN root-cause |
| `docs/conv-nax/conv-nax-phase1_2-data.json`     | TBD | Bisection + chunking data |

## Files unchanged

- All `csrc/` (Sprint A V6 NAX) untouched.
- `bench/conv_nax_matmul2d_microbench.py` (Phase 1.1 microbench) untouched.
- `docs/conv-nax/conv-nax-phase1_0-*.md` and `conv-nax-phase1_1-*.md` frozen.
- Phase 1.1 HANDOFF (`conv-nax-phase1_1-handoff-for-1_2-1_5.md`) frozen as
  historical record; Phase 1.2 corresponding HANDOFF updates are in this
  phase's docs.

## Tests inventory

**11 total** (4 Phase 1.1 + 7 Phase 1.2):

Phase 1.1:
- test_mid_resnet_finite_shape_dtype
- test_mid_resnet_vs_torch_cpu_fp32
- test_mid_resnet_vs_mlx_conv_general
- test_mid_resnet_sentinel_coverage

Phase 1.2:
- test_up1_resnet_finite_shape_dtype
- test_up1_resnet_vs_torch_cpu_fp32
- test_up1_resnet_vs_mlx_conv_general
- test_up1_resnet_sentinel_coverage (probes chunk boundaries)
- test_mid_resnet_causal_pad_t
- test_mid_resnet_causal_pad_t_flag
- test_kt1_routing

All 11 PASS. 3-session bit-exact reproduction verified for both
mid_resnet and up1_resnet (10-decimal identity).

## Commits on branch (chronological)

1. `8a099dd` — feat(conv-nax): M-chunking + asymmetric padding (Phase 1.2 core)
2. `46f7645` — test(conv-nax): Phase 1.2 -- up1_resnet + causal pad_T + K_T=1
3. (next) — docs: Phase 1.2 deliverables

## API summary

```python
from mlx_mfa.conv_nax import conv3d_nax_forward, get_chunk_plan

# Symmetric padding (Phase 1.1 compatible)
y = conv3d_nax_forward(x, w, stride=(1,1,1), padding=(1,1,1), dilation=(1,1,1))

# Asymmetric padding (Phase 1.2 new)
y = conv3d_nax_forward(x, w, padding=((2, 0), (1, 1), (1, 1)))

# Causal pad_T convenience flag (Phase 1.2 new)
y = conv3d_nax_forward(x, w, padding=(0, 1, 1), causal_pad_t=True)

# Inspect chunking plan (Phase 1.2 new, for tests + Phase 1.3 instrumentation)
chunks = get_chunk_plan(M=147456, K=13824, dtype_bytes=2)
# → [(0, 49152), (49152, 49152), (98304, 49152)]
```

## Validation status

- Phase 1.1 mid_resnet correctness: **unchanged, all 4 tests PASS**
- Phase 1.2 up1_resnet correctness: **4 tests PASS, rel_err = 3.23e-5**
- Phase 1.2 causal pad_T: **2 tests PASS, bit-exact flag vs explicit**
- Phase 1.2 K_T=1 routing: **PASS, rel_err < 1e-4**
- 3-session bit-exact reproduction: **PASS** (both shapes)
- Regression scan: **931 pass + 11 conv_nax pass = 942 total; 6 pre-existing failures unchanged**
