# Dense D=128 Tile Defaults: Resolution-Gated Recheck

Date: 2026-07-13
Branch: `explore/dense-tile-defaults`
Source HEAD: `a461b59`
Runtime: MLX `0.31.2`, mlx-mfa `2.61.0`, Apple M5 Max (40 GPU cores), macOS `27.0`, Metal `32023.918`
Perf status: **beta-3 indicative; no default change**

## Skill Invocations

| Checkpoint | Result |
|---|---|
| Benchmark harness | used `marco-mlx-suite-v3:benchmark-harness-builder` |
| Validation | used `marco-mlx-suite-v3:mlx-validation-runner` |
| Metal/NAX source review | used `marco-mlx-suite-v3:metal-kernel-dev` |

## AA.5: source and binary ground truth

The live public D=128 path is `_ext.v6_nax_forward` through
`csrc/mfa_v6_nax_primitive.cpp` and `NAAttentionKernel::createV6NAXSource`, not a retired
STEEL/probe generator. Source inspection shows the production defaults are
`BQ=64, BK=32, WM=4`; the candidate is the environment override
`BQ=64, BK=32, WM=2`. The old `MFA_V6_BLOCK_*` knobs do not control this live NAX path.
[HIGH][VERIFIED]

The smoke fingerprint compiled the exact requested MSL in separate fresh processes:

| Arm | Requested config | Source dump |
|---|---|---|
| default | no NAX tile override | `BQ=64 BK=32 BD=128 WM=4` |
| candidate | `MFA_V6_NAX_BQ=64`, `...BK=32`, `...WM=2` | `BQ=64 BK=32 BD=128 WM=2` |

The benchmark binary was kept fixed for every arm. Its path was
`/Users/marcomarcelino/code/mlx-mfa-v2/.venv/lib/python3.11/site-packages/mlx_mfa/_ext.cpython-311-darwin.so`,
SHA-256 `8f573060d80ac4ce6820a7252303ea1a88b63cfa2e1ecdd3c2b1a964f2c35c43`, Mach-O arm64.
An attempted editable rebuild was rejected by CMake because that subprocess reported `x86_64`,
while the already installed extension and Python runtime are arm64. No replacement binary was
used and no rebuild occurred between arms. [HIGH][VERIFIED]

## Phase 0: null resolution floor

The decision rule was fixed before any candidate timing:

> A candidate gain is real only if it exceeds `2x` the A-vs-A null dispersion in **both** arm orders.

The reference cell is `B=1, Hq=Hkv=8` (`B*H=8`), `N=4096`, `D=128`, fp16, non-causal. Each
fresh process ran 2 warmup groups, then 5 timing sessions with 20 dispatches per session. The
intended path and correction were checked before accepting its timing: every session traced
`nax_dense`, source dumped `BQ=64 BK=32 BD=128 WM=4`, and every output was finite with cosine
`0.9999999404` against the fp32 SDPA oracle. [HIGH][VERIFIED]

| Null set | Median range (ms) | Relative dispersion |
|---|---:|---:|
| Order A, 5 fresh default processes | `1.716194 .. 2.139825` | `23.8%` |
| Order B, 5 fresh default processes | `1.670713 .. 1.706298` | `2.1%` |
| All 10 fresh default processes | `1.670713 .. 2.139825` | **`27.4%`** |

The global population standard deviation was `8.1%` of the mean. The largest order-specific
floor is `23.8%`; the conservative all-session floor is `27.4%`. Therefore a candidate would
need to exceed `47.6%` in order A and `4.2%` in order B, or `54.8%` under the conservative global
floor. The historical `+3.3%` candidate signal is below this requirement. [HIGH][VERIFIED]

The order-A processes were internally stable within each process, but ran at distinct process
power/performance levels (for example `2.1175..2.4211 ms` in the second process); this is exactly
the cross-process beta variance the null calibration is intended to expose. No thermal-invalid
run was silently discarded: the run was retained as a conservative instrument floor. [HIGH][VERIFIED]

## Phase 1: candidate and generalization gate

The candidate timing grid was **not run**. The prompt makes it conditional on the Phase-0 floor;
running a `~3.3%` candidate against a `23.8%`/`27.4%` null would create an invalid ratio, not
evidence. The prior report's `BQ64_BK32_WM2` result was read as historical context only; it was
not reissued as a current performance claim. The candidate MSL smoke fingerprint above proves
configuration reachability, not performance or correction. [HIGH][VERIFIED]

Consequently there is no accepted cell-level candidate gain, slope, or regression result from this
run. The requested grid (`N={2048,4096,8192}`, fp16/bf16, causal/non-causal,
`B*H={8,32,64}`) is explicitly deferred to a stable macOS/runtime where the null floor can resolve
a few-percent change. [HIGH][DEDUCED]

## Verdict

**Issue 3: indécidable sur beta; close without changing the default.** [HIGH][VERIFIED]

- Production default remains `BQ=64, BK=32, WM=4`.
- No candidate tile is promoted, routed, or added to a decision cache.
- No production source was changed.
- Re-measure on stable macOS with the same-build protocol; repeat Phase 0 first and only run the
  candidate/generalization grid if the null floor falls below the required resolution.

## Red-team

- A skeptic could attribute the high order-A floor to a transient power state. That is not a reason
  to discard it silently: the user-set rule requires two-order evidence, and one order is currently
  unresolved. [HIGH][VERIFIED]
- The smoke source dump is a real tile fingerprint, but it does not prove the candidate is faster;
  no candidate ratio is reported. [HIGH][VERIFIED]
- The existing extension predates this branch switch and could not be rebuilt because CMake saw
  x86_64. The binary hash is recorded so the measurement is reproducible; a future stable rerun
  should rebuild once in a native arm64 build environment before recalibrating the floor. [HIGH][VERIFIED]
- Correctness of the candidate across the full grid was not claimed because Phase 1 was gated off.
  The default null arm alone passed the required oracle check. [HIGH][VERIFIED]

## Validation

- Ran: `py_compile benchmarks/bench_dense_tile_defaults.py`; 10 fresh default processes, 5 per
  order, 20 GPU-fenced dispatches per timing session; source-dump fingerprint and fp32 correction.
- Ran: `/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python -m pip install --no-build-isolation -e .`
  (rejected by CMake architecture guard; no replacement binary installed).
- Ran: `/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python -m pytest tests/ -q` — `3488 passed,
  93 skipped, 3 warnings`.
- Validated: null floor computed before candidate timing; all null outputs finite, cosine
  `0.9999999404`, terminal trace `nax_dense`; no production routing/default changed. [HIGH][VERIFIED]

## Git

Harness and report are on `explore/dense-tile-defaults`; no merge, push, tag,
or release was made. Raw benchmark JSON/stderr files under `benchmarks/results/dense_tile_item8/`
remain local untracked artifacts and were intentionally not included in the commit.
