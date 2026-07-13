# Hardware and Platform Support

## Installation floor

mlx-mfa builds only on Apple Silicon macOS. The package requires Python 3.10 or newer and MLX 0.31.2 or newer. CMake defaults the deployment target to macOS 14.0; callers may override that target at build time.

The distribution model is source-first: installation compiles the extension against the MLX headers present in the active environment. Unsupported systems fail during build rather than installing a nonfunctional binary.

## Runtime tiers

| Runtime capability | Main implementation families |
|---|---|
| Apple GPU without NAX | STEEL attention, SDPA fallbacks, portable Python composition |
| M5-class NAX available | V6 NAX dense, sparse, GNA, selected backward, and expert kernels |
| Unsupported shape or dtype | MLX SDPA or another documented fallback |

NAX availability is detected at runtime. It is not inferred solely from a package version or a user-provided environment variable.

## Metal language use

Metal shader source is generated and compiled at runtime. Cooperative-tensor kernels are selected only when the device reports the required capability. Host APIs introduced after the deployment floor are guarded by availability checks.

## Dtype envelope

The accelerated attention families primarily accept `float16` and `bfloat16`. Public APIs preserve correctness for `float32` by delegating where a native kernel does not support it. Mixed Q/K/V dtypes are rejected.

Quantized utilities have narrower contracts documented by their API. Development-only int8 probes are not public routes.

## Head dimensions

Dense M5 auto-routing uses D=128. Several expert and fallback families support D=64 or D=256. D=512 public attention is deliberately delegated to SDPA; there is no mlx-mfa D=512 attention kernel.

## Portability status

Performance values in this documentation were collected on an M5 Max with MLX 0.31.2 and a macOS 27 beta runtime in July 2026. They are β3-indicative and must be revalidated on the stable macOS release before thresholds are treated as durable across systems.
