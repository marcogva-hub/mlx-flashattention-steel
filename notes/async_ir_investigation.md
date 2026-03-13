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

### Task 3 — Runtime behavior: async vs sync (separate processes)

#### 3A. Separate-process benchmark scripts

Added:
- `scripts/bench_async_path.py` (ensures `MFA_DISABLE_ASYNC` unset)
- `scripts/bench_sync_path.py` (sets `MFA_DISABLE_ASYNC=1`)

Both run shape:
- `B=1, H=8, N=8192, D=128, causal=True`
- warmup=3, timed iterations=10

#### 3B. Pipeline-path logging

Added temporary gated logging in `csrc/shader_cache.mm` under env:
- `MFA_IR_INVESTIGATE=1`

Logs include:
- metallib load path
- async pipeline SUCCESS/FAILED reason
- disable-by-env status

#### 3C. Results

Initial async run (before placing metallib in site-packages path):
- stderr: `[MFA-IR-INVESTIGATE] Async pipeline: FAILED (metallib missing)`
- async median: **24.707 ms**

Initial sync run:
- stderr: `[MFA-IR-INVESTIGATE] Async pipeline: disabled by MFA_DISABLE_ASYNC`
- sync median: **25.560 ms**

Observation:
- The first async-vs-sync comparison was invalid for Stage-3 analysis because async metallib was not loaded.

To force true async-metallib loading for this editable install setup, copied:
- `mlx_mfa/precompiled/async_v2.metallib`
  -> `.venv/lib/python3.11/site-packages/mlx_mfa/precompiled/async_v2.metallib`

Async run after copy:
- stderr includes:
  - `[MFA-IR-INVESTIGATE] Async pipeline: loading .../site-packages/mlx_mfa/precompiled/async_v2.metallib`
  - `[MFA-IR-INVESTIGATE] Async pipeline: SUCCESS (mlx_mfa_v2_async_attention_d128)`
- async median: **27.776 ms**

Sync run after copy:
- stderr: `[MFA-IR-INVESTIGATE] Async pipeline: disabled by MFA_DISABLE_ASYNC`
- sync median: **24.577 ms**

Measured ratio (same post-copy environment):
- `sync / async = 24.577 / 27.776 = 0.885x`
- equivalently async path is ~**13.0% slower** than sync fallback on this machine/OS for this shape.

Interim conclusion:
- Shipped async metallib **does load** successfully (Stage 3 load succeeds).
- But observed runtime performance indicates no practical overlap benefit on this setup; behavior is consistent with serialization/emulation/ineffective async execution at runtime.

### Task 4 — GPU capture / trace attempt

Environment availability:
- `xcodebuild -version`: Xcode 26.2 (17C52)
- `xcrun xctrace version`: 26.0 (17C52)

Command-line captures executed:
- `xcrun xctrace record --template 'Metal System Trace' --time-limit 8s --output /tmp/mfa_async.trace --launch -- .venv/bin/python scripts/bench_async_path.py`
- `xcrun xctrace record --template 'Metal System Trace' --time-limit 8s --output /tmp/mfa_sync.trace --launch -- .venv/bin/python scripts/bench_sync_path.py`

Exported TOC metadata:
- Async run duration (trace summary): ~1.51s
- Sync run duration (trace summary): ~2.57s
- Template setting in both traces reports:
  - `Shader Timeline: Disabled`

Implication:
- These captures confirm trace collection is working, but they do **not** provide direct shader-timeline overlap visualization for DMA-vs-ALU overlap in this run.
- To answer overlap conclusively, an Xcode GUI capture with Shader Timeline enabled is still required.

Interim interpretation for Stage 4:
- No direct overlap evidence was extractable from the recorded traces in this configuration.
- Combined with benchmark results, behavior remains consistent with ineffective async overlap on this setup.
