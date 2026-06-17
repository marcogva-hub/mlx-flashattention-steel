# Metal kernel debugging — institutional techniques

Codified from v2.50 Prompt 4 dV residual investigation.  Use these
techniques *before* deeper gradient-bisection or kernel-disassembly
work — they isolate dispatch-routing bugs in minutes rather than
hours.

---

## §1 — When to use which technique

| Symptom | First technique | Why |
|---|---|---|
| Output numerically wrong but kernel "ran" | §2 Sentinel writes | Confirms which dispatch path is active |
| Output partially correct (some rows ok, others zero) | §3 Tile-boundary inspection | Isolates per-tile vs per-row bug |
| Gradient differs from autodiff reference by small amount | §4 Forward-vs-backward consistency | LSE convention / scale factor mismatch |
| Gradient differs by large amount with zeroed regions | §5 Output region map | Active-region vs masked-region |

These are layered: §2 ALWAYS runs first.  Don't proceed to §4 if §2
shows the wrong code path is active.

---

## §2 — Sentinel writes (the dispatch-path probe)

The fastest way to confirm "is this code actually executing?"

### Technique

1. Pick a uniquely-valued constant: a value that cannot be produced by
   the legitimate computation.  Examples:
   - `123456.0f` (FP32) — too large to occur naturally in attention
   - `-999.0f` for outputs normally in [-3, 3]
   - Encoding the call site: `bit_cast<float>(0xDEADBEEFu)` for the
     mask-prep kernel, `bit_cast<float>(0xCAFEBABEu)` for the score-
     reduction kernel.

2. Replace the kernel's first non-trivial output write with the
   sentinel, conditionally on a runtime input that *should* be true:

   ```metal
   // After the value is computed but before it's written:
   if (causal && tid_x == 0 && tid_y == 0) {
     out[0] = 123456.0f;  // sentinel — only on the eligible path
   } else {
     out[output_index] = computed_value;
   }
   ```

3. Run the test with the sentinel in place; read `out[0]` from Python
   after invoking `mx.eval(out)`:

   ```python
   out = flash_attention(q, k, v, causal=True)
   mx.eval(out)
   print(float(out[0, 0, 0, 0]))  # 123456.0 ⇒ this kernel is active
   ```

4. Three outcomes:
   - **Sentinel present** (`out[0] == 123456.0`): the gated path is
     active.  Numerical bug must be downstream of the sentinel.
   - **Sentinel absent + computed value present**: the gate condition
     is false; the kernel runs but the eligibility branch is wrong.
   - **Neither sentinel nor computed value**: a different kernel
     entirely handled this dispatch.  This is the dispatch-routing
     bug.

### Empirical case (v2.50 Prompt 4)

The V6NAX backward dV residual investigation initially suspected the dV
kernel's softmax stabilisation.  A sentinel write in the V6NAX dV kernel
showed it was NOT active for causal forward — `causal=True` routed the
forward to STEEL (different LSE convention), and the V6NAX backward
consumed the wrong-domain LSE.  Time-to-diagnosis: ~20 minutes with
sentinels vs the ~6 hours of gradient bisection that had preceded.

### Caveats

- Sentinels must be reverted before any benchmark.  Use a build-time
  `#ifdef MFA_DEBUG_SENTINEL` guard.
- Don't rely on `printf` from Metal kernels — printf output is
  asynchronous and sometimes silently dropped on Apple Silicon under
  load.  Sentinel writes are synchronous via the output buffer.
- The sentinel's data type must match the output buffer's element
  type.  FP16 sentinels need a value representable in FP16 (e.g.,
  `mx.float16(60000)`).

---

## §3 — Tile-boundary inspection

When some rows of the output are correct and others are wrong (e.g.,
"first 1024 rows match, rows 1024+ are zero"), the bug is in a
tile-boundary or `kb_lim`/`qb_lim` computation.

### Technique

1. Compute the diff between `out` and reference:
   ```python
   diff = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))
   diff_per_row = diff.mean(axis=-1)  # average over D
   diff_per_row_np = np.array(diff_per_row)
   ```

2. Plot or print the per-row diff.  Look for:
   - Sharp transitions at row indices divisible by BQ (32, 64) → tile
     loop bug
   - Step changes at row indices matching `qL_off` boundaries →
     paged/varlen offset bug
   - Diff increasing monotonically → accumulator overflow

3. Cross-reference the transition row indices with the kernel's tile
   parameters (BQ, BK, WM, qL_off).

### Empirical case (v2.50 Prompt 5a B.5)

`TestNativeBackwardRouting[128-2048]` showed `out[0, 0, 1024:, :] == 0`
exactly.  1024 = 16 × BQ (BQ=64) — the tile loop terminated 16 tiles
into a 32-tile problem.  Real kernel bug isolated to D=128 backward
(D=64 path correct).  Deferred per documented "D=128 backward is
research-only" carve-out.

---

## §4 — Forward-vs-backward LSE consistency

In FlashAttention, the backward kernel reads `lse` written by the
forward.  Mismatched conventions are silent — both kernels run, but
the backward decodes `exp(score - lse)` with the wrong base.

### Convention map (current mlx-mfa)

| Kernel family | LSE convention |
|---|---|
| V6NAX forward / backward | Natural log: `lse = log(sum(exp(score)))` |
| STEEL legacy forward | Log2: `lse = log2(sum(exp2(score)))` |
| SDPA reference | Natural log (numpy convention) |

### Technique

1. After a forward run, materialise and read a single lse value
   (Python evaluation of the MLX array):
   ```python
   _, lse = flash_attention(q, k, v, return_lse=True)
   mx.eval(lse)
   print(f"lse[0,0,0] = {float(lse[0,0,0]):.4f}")
   ```

2. Compute expected lse manually (single row, manageable size):
   ```python
   scores = (q[0,0,0] @ k[0,0].T) * scale  # [S]
   expected_nat = math.log(float(mx.exp(scores).sum()))
   expected_log2 = math.log2(float(mx.exp(scores).sum()))
   print(f"expected nat={expected_nat:.4f}, log2={expected_log2:.4f}")
   ```

3. Compare to the kernel's lse value — which convention matches?

4. If the forward writes log2 but the backward reads as natural-log
   (or vice versa), `dV = P @ dO` will be off by a factor of `1/ln(2)`
   in some entries (Prompt 4 case).

### Empirical case (v2.50 Prompt 4)

`lse[0,0,0]` from V6NAX forward causal was natural-log; STEEL legacy
forward returned log2 for the same inputs.  The dispatch in
`MFAV6Forward::eval_gpu()` routed causal to STEEL legacy, but the V6NAX
backward expected natural-log.  Fix: lift the causal-routing gate so
all V6NAX-eligible inputs use V6NAX forward (consistent LSE convention).

---

## §5 — Output region map

When mask-dependent code paths are suspected (causal, sliding-window,
sparse), produce a visual map of "where in output does the diff live?"

### Technique

```python
import numpy as np
diff = np.array(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))
# diff shape: [B, H, N, D]
per_q_max = diff.max(axis=(0, 1, 3))  # max diff per query row
# Discretise: 0 = pass, 1 = fail
fail_rows = (per_q_max > tol).astype(int)
# Print as ASCII map (one char per row, 64 per line)
for i in range(0, len(fail_rows), 64):
    print("".join("X" if x else "." for x in fail_rows[i:i+64]))
```

The pattern tells you what:
- Solid `X` block at the end → tail-row write bug
- Sawtooth pattern → per-tile correctness varies → tile-level bug
- Diagonal pattern → causal mask alignment error
- Sparse `X`s scattered → FP noise around the tolerance threshold

---

## §6 — Anti-patterns

DO NOT:

1. **Trust "the kernel ran without errors"** as proof of correctness.
   Metal kernels can silently produce wrong output (out-of-bounds
   writes, type-pun bugs, missed barriers).  Always validate with a
   reference.

2. **Rely on `mx.metal.start_capture`** as a first-line debug tool.
   GPU capture is essential for register-pressure / occupancy
   analysis but slow to set up and overkill for dispatch-routing
   bugs.  Try sentinels first.

3. **Comment-out kernel sections to bisect**.  Removing sections
   often changes register allocation and SIMD scheduling — the bug
   you're hunting may disappear (or new bugs appear) for reasons
   unrelated to the change.  Use sentinels to verify *which* path
   is active without modifying the path.

4. **Re-run the suspected-buggy bench more times hoping for a clean
   number**.  If max_diff is 0.4 and the spec demands < 0.05, it's
   not noise.  Stop running and start instrumenting.

---

## Cross-references

- `docs/v50/audit-framing-inversions.md` §6 (Pattern #5 incomplete-fix
  dispatch-chain)
- `CLAUDE_V6_NAX.md` §AA.5.x (multi-gate audit requirement)
- `CLAUDE_V6_NAX.md` §AA.4 (canonical bench methodology — required
  before/after kernel changes)
- `~/.claude/skills/metal-kernel-dev/SKILL.md` (pre-impl register
  budget review)
