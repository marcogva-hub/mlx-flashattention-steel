# Speculative Decode Runtime Design (Draft/Verify Integration)

Date: 2026-03-13
Branch: `codex/speculative-decode-runtime`

## 1) Current capability snapshot

Current repo state already includes a low-level verify primitive:

- `flash_attention_speculative_verify(q_target, k_cache, v_cache, draft_ids, ...)`
  - returns `(out, lse, target_logprobs)`
  - computes target-side attention for draft queries and a proxy target log-prob per draft token
  - does **not** decide accept/reject prefix length
  - does **not** mutate runtime/cache state

Runtime currently exposes this as:

- `DecodeRuntime.speculative_verify(...)`
  - dense runtime can use internal cache if explicit `k_cache/v_cache` are omitted
  - non-dense backends must pass explicit cache tensors

## 2) Manual gap

Today callers still have to manually orchestrate:

1. call verify helper,
2. compare draft vs target likelihoods,
3. determine accepted prefix length,
4. build accepted/rejected token slices,
5. manage cache/state consequences externally.

This keeps speculative decode at helper-level rather than runtime-native serving flow.

## 3) Target semantics for this branch

Add an explicit runtime-level speculative step API that:

- keeps low-level verify API unchanged,
- computes an **inspectable** accept/reject outcome from verify outputs,
- returns structured runtime metadata for serving orchestration,
- remains narrow and safe (no full scheduler implementation).

Proposed runtime method:

- `DecodeRuntime.speculative_step(...)`

Core behavior:

- takes `q_target` and `draft_ids`
- runs `speculative_verify(...)`
- derives per-token acceptance against a threshold
- computes accepted prefix length (contiguous from token 0)
- returns accepted/rejected token partitions and diagnostic metadata

## 4) Backend/cache integration in this pass

Supported now:

- Dense runtime using internal cache (same requirement as `speculative_verify`)
- Any backend with explicit dense `k_cache/v_cache` passed by caller

Out of scope for this pass:

- fully automatic paged-cache native speculative state mutation
- packed-query-only speculative fast paths
- full scheduler queue/state management

## 5) Interaction with other runtime features

- Prefix reuse: compatible when dense cache is already seeded
- Chunked prefill: compatible as a producer of runtime cache before speculative step
- Paged runtime: supported only through explicit cache tensors in this pass
- Splitfuse/shared-prefix helpers remain unchanged

## 6) Correctness/test expectations

New coverage should validate:

- full accept and partial accept behavior
- accepted-prefix length semantics
- deterministic threshold behavior
- clear failures for unsupported cache/backend combinations
- runtime metadata reflects speculative-step usage and last outcome

## 7) Benchmark goal

Add focused benchmark matrix for acceptance-rate-sensitive scenarios to measure:

- normal decode path vs speculative runtime path
- low/medium/high acceptance regimes
- whether benefit is throughput, or mostly runtime capability/integration
