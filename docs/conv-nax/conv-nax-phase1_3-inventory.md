# Phase 1.3 — File + Test Inventory

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_3` (branched from `experiment/conv-nax-phase1_2` tip)
**Scope:** Multi-chunk working-set instrumentation + 16 GB hard gate + per-chunk synchronization

## Files modified

| Path | Δ lines | Purpose |
|------|--------:|---------|
| `mlx_mfa/conv_nax.py`     | +96  | `estimate_working_set()`, per-chunk eval, tighter sanity assert |
| `tests/test_conv_nax.py`  | +106 | 4 new Phase 1.3 tests (working set + 5-chunk correctness) |

## Files added (deliverables)

| Path | Purpose |
|------|---------|
| `docs/conv-nax/conv-nax-phase1_3-inventory.md` | This file |
| `docs/conv-nax/conv-nax-phase1_3-decisions.md` | D23-D26 |
| `docs/conv-nax/conv-nax-phase1_3-results.md`   | Working-set table + perf preview |
| `docs/conv-nax/conv-nax-phase1_3-data.json`    | Per-shape peak memory data |

## Public API additions

```python
from mlx_mfa.conv_nax import (
    conv3d_nax_forward,    # existing, now per-chunk eval enabled internally
    get_chunk_plan,         # existing
    estimate_working_set,   # NEW — Phase 1.3
)

ws = estimate_working_set(M_total, K, N, dtype_bytes=2)
# {
#   "chunks": [(m_offset, m_chunk), ...],
#   "n_chunks": int,
#   "per_chunk_im2col_bytes": int,
#   "per_chunk_matmul_out_bytes": int,
#   "per_chunk_peak_bytes": int,
#   "concat_out_bytes": int,
#   "total_peak_bytes": int,
#   "within_hard_gate": bool,
#   "hard_gate_bytes": 16 * 1024**3,
# }
```

## Tests inventory

**15 total** (4 Phase 1.1 + 7 Phase 1.2 + 4 Phase 1.3, all PASS):

Phase 1.3 additions:
- `test_working_set_all_production_shapes_within_gate` — all 6 design §3.1 shapes fit < 16 GB peak
- `test_working_set_chunk_plan_correctness` — chunks sum to M_total, M_TILE-aligned, monotonic
- `test_working_set_oversize_rejected_by_sanity` — 16+ GB shapes fire ValueError at API entry
- `test_multi_chunk_correctness_5chunks` — 5-chunk shape (M=297000, K=13824) rel_err < 1e-4

## Commits on branch (chronological)

1. `ca4b529` — feat+test(conv-nax): Phase 1.3 working-set + per-chunk eval + 4 new tests
2. (next) — docs(conv-nax): Phase 1.3 deliverables

## Validation status

- Phase 1.1 + 1.2 tests: 11/11 PASS unchanged
- Phase 1.3 tests: 4/4 PASS new
- Real-memory probe (large shape, M=1.1M, 17 chunks): peak 3.53 GB observed (estimate 2.38 GB), rel_err 3.4e-5
- Per-chunk eval: 9× peak memory reduction vs lazy accumulation (32.29 GB → 3.53 GB at 17-chunk shape)
- Hard gate enforced at sanity-assert time (Category 7)
