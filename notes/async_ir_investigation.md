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

### Task 2 — Current toolchain compile and AIR comparison

#### 2A. Compile original async source on current toolchain

Command used:
- `xcrun -sdk macosx metal -target air64-apple-macos15.0 -c csrc/async_v2_kernel.metal -o /tmp/async_v2_current.air`

Result:
- **Compilation fails** on current toolchain.

Observed errors (first-order failures):
- `error: illegal string literal in 'asm'` for:
  - `air.simdgroup_async_copy_1d.p3i8.p1i8`
  - `air.simdgroup_async_copy_1d.p1i8.p3i8`
  - `air.simdgroup_async_copy_2d.p3i8.p1i8`
  - `air.simdgroup_async_copy_2d.p1i8.p3i8`
  - `air.wait_simdgroup_events`
- Follow-on unresolved symbol errors for `__metal_simdgroup_async_copy_2d` and `__metal_wait_simdgroup_events`.

Interpretation:
- Stage 1 (`MSL -> AIR`) for this source is blocked on this toolchain due to rejection of inline `__asm` AIR intrinsic mapping.

#### 2B. Build stripped no-asm variant

Created:
- `csrc/async_v2_noasm.metal`

Change strategy:
- Replaced only the `simdgroup_event` intrinsic bridge block with a synchronous software implementation of `simdgroup_event::async_copy(...)` and no-op `wait(...)`.
- Kept kernel entrypoints and main flow intact for comparison purposes.

Command used:
- `xcrun -sdk macosx metal -target air64-apple-macos26.0 -c csrc/async_v2_noasm.metal -o /tmp/async_v2_noasm.air`

Result:
- Compiles successfully (warnings only).

#### 2C. Compare shipped IR vs no-asm IR

Commands used:
- `xcrun metal-objdump --disassemble mlx_mfa/precompiled/async_v2.metallib > /tmp/async_v2_metallib_disasm.ll`
- `xcrun metal-objdump --disassemble /tmp/async_v2_noasm.air > /tmp/async_v2_noasm.ll`
- Symbol grep + diff on async/event terms.

Findings:
- Shipped metallib IR contains:
  - `@air.simdgroup_async_copy_2d.p3i8.p1i8`
  - `@air.wait_simdgroup_events`
  - `struct.metal::simdgroup_event`
- No-asm AIR contains **none** of those symbols/calls.

Interim conclusion:
- Shipped artifact and no-asm artifact are materially different at AIR level.
- Therefore, any runtime serialization hypothesis now points to Stage 3/4 (runtime JIT/lowering/execution), not Stage 1/2 for the shipped binary.
