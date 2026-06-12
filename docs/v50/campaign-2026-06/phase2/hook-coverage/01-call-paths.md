# II-13 — User-facing call-path enumeration (2026-06-12)

Verified by source introspection of mlx.nn (MLX 0.31.2) + the
_auto_hooks patch list. Classification: COVERED / CORRECTLY-UNHOOKED /
GAP.

| Entry point | Underlying op | install_hooks patches? | Classification |
|---|---|---|---|
| `nn.Conv3d` | `mx.conv3d` | YES (II-7) | **COVERED** |
| `mx.conv3d` direct | itself | YES (II-7) | **COVERED** |
| `mx.conv_general` (5D weights) | itself | YES (v2.36+) | **COVERED** |
| `nn.Conv2d` / `mx.conv2d` | `mx.conv2d` | no | CORRECTLY-UNHOOKED — mlx-mfa has no 2D-conv acceleration |
| `nn.Conv1d` / `mx.conv1d` | `mx.conv1d` | no | CORRECTLY-UNHOOKED — same |
| `nn.ConvTranspose3d` | `mx.conv_transpose3d` | no | CORRECTLY-UNHOOKED — input-dilation class, NAX-ineligible by envelope |
| `nn.MultiHeadAttention` | `mx.fast.scaled_dot_product_attention` | no | CORRECTLY-UNHOOKED **on M5 by Pattern #6** (Apple SDPA NAX owns dense fwd; hooking would route to slower paths). FLAGGED-GATED: on M1/M2 MFA fwd historically won 1.6-2.2x — a candidate M1-gated SDPA hook requires an M1 bench (no M1 here); recorded for the ledger, NOT enabled. |
| `mx.fast.scaled_dot_product_attention` direct | itself | no | same as above |
| `nn.RoPE` / `mx.fast.rope` | `mx.fast.rope` | no | CORRECTLY-UNHOOKED — mx.fast.rope IS the optimum (the repo's own STEEL rope was declined 4x against it) |
| `nn.QuantizedLinear` | `mx.quantized_matmul` | no | CORRECTLY-UNHOOKED — SVDQuant is an explicit model-surgery API (Auto-default tier 3), not a transparent hook |
| VSR-portfolio idiom (SeedVR2 class) | `mx.conv_general` + direct `mlx_mfa.*` calls | YES | COVERED (Phase-96 telemetry: executed=12408) |

**Coverage verdict**: zero remaining GAP entries on M5. The only
flagged item is the M1-gated SDPA hook candidate (cannot be benched on
this hardware; Pattern #6 forbids enabling unbenched).
