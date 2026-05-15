# Integration smoke tests — v2.50.1 Prompt 5g Phase D

Multi-model smoke tests that exercise the auto-hooked Conv3D NAX path
using **synthetic inputs matching the architectural signatures** of
user's VSR model portfolio.  Real model weights are too large for the
repo; these tests verify that:

1. The hook engages (not falls back) for each model's typical input
   pattern.
2. Outputs are finite + shape-correct.
3. Hook telemetry confirms NAX engagement across the workload.

Models simulated (architecture-pattern level):

- **SeedVR2** (VAE Conv3D with fp32 input + fp16 weights — the Pattern
  #8 root-cause shape signature)
- **FlashVSR** (LCSA sparse, D=128 patches; Conv3D used in VAE encode/
  decode paths)
- **STCDiT** (Wan2.1 backbone, video DiT with 3D conv preconditioning)
- **SparkVSR** (CogVideoX1.5-5B-I2V backbone, Conv3D in VAE)

These are intentionally lightweight — full inference would require
GB-scale model weights.  They verify the **NAX dispatch path engages**
for the canonical input patterns each model uses.
