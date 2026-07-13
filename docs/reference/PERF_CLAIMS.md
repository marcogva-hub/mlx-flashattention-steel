# Performance Claim Registry

## Active claims

The identifiers below are executable reachability locks retained for compatibility with the test registry. Their versioned names are historical identifiers; this table asserts terminal engagement, not the old ratio embedded in a name or release note.

| Claim ID | Current assertion |
|---|---|
| `v2.39.1_d64_qL4096_fused_bk16_engages_via_auto` | registered public-path engagement case |
| `v2.39.1_d64_qL8192_fused_bk16_engages_via_auto` | registered public-path engagement case |
| `v2.38.1_d64_qL4096_v6nax_dvec_engages_via_auto` | registered public-path engagement case |
| `v2.38.1_d64_qL8192_v6nax_dvec_engages_via_auto` | registered public-path engagement case |
| `v2.39.1_d64_qL16384_fused_bk16_engages_via_auto` | registered public-path engagement case |
| `v2.38.1_d64_qL16384_v6nax_dvec_engages_via_auto` | registered public-path engagement case |
| `v2.37.2_d64_qL4096_v6nax_engages_via_auto` | registered public-path engagement case |
| `v2.37.2_d64_qL8192_v6nax_engages_via_auto` | registered public-path engagement case |
| `v2.50.0_prompt5b_d128_qL8192_auto_engages_v6nax_split_at_parity` | registered public-path engagement case |
| `v2.39.2_internal_d64_qL2048_auto_engages_v6nax_at_parity` | registered public-path engagement case |
| `v2.39.2_internal_d64_qL1024_auto_falls_back_to_sdpa` | registered fallback case |
| `ii12_d64_qL8192_default_on_v6nax` | registered default-on backward case |
| `ii12_d64_qL8192_optout_sdpa` | registered backward opt-out case |
| `ii9_conv3d_t16_64x64_c128_fp16_mpp_default` | registered fp16 Conv3D hook case |
| `iii1_conv3d_t16_64x64_c128_bf16_mpp_default` | registered bf16 Conv3D hook case |
| `iii2_tq_paged_decode_step_default` | registered TurboQuant decode case |

## Current measured claims

Only [RESULTS.md](../../RESULTS.md) carries active performance numbers. Those numbers come from hardened same-dtype harnesses with both terminals fingerprinted and are stamped M5 Max, MLX 0.31.2, macOS 27 beta, July 2026.

## Retracted claims

Ratios produced by pre-hardening harnesses are excluded from current documentation. Their executable cases remain above where they still protect route reachability. A claim can return to the current table only after same-dtype, two-terminal, oracle-correct remeasurement.
