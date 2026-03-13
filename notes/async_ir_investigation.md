# Async Copy IR Investigation Report

Date: 2026-03-13
Hardware: Apple Silicon (local machine)
macOS: darwin25.4.0 target toolchain (macOS 26 generation)
Xcode/Metal toolchain: Apple metal version 32023.864 (metalfe-32023.864)

## Investigation log

### Task 1 — Shipped metallib inspection

Artifact:
- `mlx_mfa/precompiled/async_v2.metallib`

Tooling used:
- `xcrun metal-objdump --section-headers`
- `xcrun metal-objdump --disassemble`
- `xcrun metal-objdump --air-version`

Key findings:
- Metallib parses as `file format metallib`.
- Header shows `Platform: MACOS`, `PlatformMajor: 15`, `File version: 1.2.7`.
- AIR version entries:
  - `mlx_mfa_v2_async_attention: 2.7`
  - `mlx_mfa_v2_async_attention_d128: 2.7`
- Disassembled AIR IR **contains async intrinsics and event wait calls**:
  - `@air.simdgroup_async_copy_2d.p3i8.p1i8`
  - `@air.wait_simdgroup_events`
  - `struct.metal::simdgroup_event`

Representative IR evidence from disassembly:
- `call ... @air.simdgroup_async_copy_2d.p3i8.p1i8(...)`
- `call void @air.wait_simdgroup_events(...)`
- `declare ... @air.simdgroup_async_copy_2d.p3i8.p1i8(...)`
- `declare void @air.wait_simdgroup_events(...)`

Interim conclusion:
- Stage 2 (AIR -> metallib) did **not** strip async-copy symbols in the shipped library.
- The shipped metallib still encodes async-copy-related AIR intrinsics.

## Pending tasks
- Task 2: compile current source with current toolchain and compare behavior/artifacts.
- Task 3: runtime async-vs-sync benchmark in separate processes + pipeline-path logging.
- Task 4: GPU capture (if available).
- Task 5: final stage-by-stage diagnosis and recommendation.
