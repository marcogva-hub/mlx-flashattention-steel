# Paged Correctness + Validation Envelope (Volet H)

Host M5 Max / macOS 26.6 / MLX 0.31.2 · generated from tests/test_paged_envelope.py

## Correctness cells (valid → independent per-sequence fp64 oracle)

| cell | H/Hk | D | seqs | Nq | causal | dtype | relerr vs fp64 |
|---|---|---|---|---|---|---|---|
| homo_D64_f16 | 4/4 | 64 | [48, 48] | 17 | True | f16 | 2.35e-04 |
| hetero_NqLTNk_D64_f16 | 4/4 | 64 | [31, 50] | 17 | True | f16 | 3.30e-04 |
| hetero_NqGTNk_D64_f16 | 4/4 | 64 | [13, 40] | 24 | True | f16 | 3.18e-04 |
| hetero_nc_D64_f16 | 4/4 | 64 | [31, 50] | 17 | False | f16 | 3.44e-04 |
| gqa_hetero_D64_f16 | 8/2 | 64 | [31, 50] | 17 | True | f16 | 2.88e-04 |
| hetero3_D64_f16 | 4/4 | 64 | [20, 48, 70] | 9 | True | f16 | 3.99e-04 |
| homo_D128_f16 | 4/4 | 128 | [48, 48] | 17 | True | f16 | 2.32e-04 |
| hetero_NqLTNk_D128_f16 | 4/4 | 128 | [31, 50] | 17 | True | f16 | 3.28e-04 |
| hetero_NqGTNk_D128_f16 | 4/4 | 128 | [13, 40] | 24 | True | f16 | 3.40e-04 |
| hetero_nc_D128_f16 | 4/4 | 128 | [31, 50] | 17 | False | f16 | 3.15e-04 |
| gqa_hetero_D128_f16 | 8/2 | 128 | [31, 50] | 17 | True | f16 | 2.50e-04 |
| hetero3_D128_f16 | 4/4 | 128 | [20, 48, 70] | 9 | True | f16 | 2.92e-04 |
| homo_D64_bf16 | 4/4 | 64 | [48, 48] | 17 | True | bf16 | 2.99e-03 |
| hetero_NqLTNk_D64_bf16 | 4/4 | 64 | [31, 50] | 17 | True | bf16 | 2.65e-03 |
| hetero_NqGTNk_D64_bf16 | 4/4 | 64 | [13, 40] | 24 | True | bf16 | 1.89e-03 |
| hetero_nc_D64_bf16 | 4/4 | 64 | [31, 50] | 17 | False | bf16 | 2.99e-03 |
| gqa_hetero_D64_bf16 | 8/2 | 64 | [31, 50] | 17 | True | bf16 | 3.63e-03 |
| hetero3_D64_bf16 | 4/4 | 64 | [20, 48, 70] | 9 | True | bf16 | 2.92e-03 |
| homo_D128_bf16 | 4/4 | 128 | [48, 48] | 17 | True | bf16 | 1.98e-03 |
| hetero_NqLTNk_D128_bf16 | 4/4 | 128 | [31, 50] | 17 | True | bf16 | 2.53e-03 |
| hetero_NqGTNk_D128_bf16 | 4/4 | 128 | [13, 40] | 24 | True | bf16 | 2.57e-03 |
| hetero_nc_D128_bf16 | 4/4 | 128 | [31, 50] | 17 | False | bf16 | 1.84e-03 |
| gqa_hetero_D128_bf16 | 8/2 | 128 | [31, 50] | 17 | True | bf16 | 1.99e-03 |
| hetero3_D128_bf16 | 4/4 | 128 | [20, 48, 70] | 9 | True | bf16 | 2.34e-03 |

## Validation cells (malformed → must raise ValueError)

| cell | result |
|---|---|
| cx02_int64_cu | raises ✓ |
| cx02_float_cu | raises ✓ |
| cx03_v_blocks | raises ✓ |
| cx03_v_heads | raises ✓ |
| cx03_v_headdim | raises ✓ |
| cx03_v_blocksize | raises ✓ |
| cx03_raw_steel_v | raises ✓ |
| seq_short | raises ✓ |
| float_meta | raises ✓ |
| oob_page | raises ✓ |

CX-01 (per-sequence causal offset), CX-02 (cu_seqlens int32), CX-03 (K/V pool shape) are the completeness oracles. byteΔ-identity on homogeneous seq_lens proven (fixed vs batch-global before = 0.0; heterogeneous = 2.35).

