"""Benchmark varlen vs padded dense for token-merged/pruned sequences.

Validates that flash_attention_varlen works correctly and performantly
for sequences whose lengths change between diffusion steps (token merging).

Usage:
    python benchmarks/bench_varlen_pruning.py
"""
import mlx.core as mx
import time
import numpy as np
from mlx_mfa import flash_attention
from mlx_mfa.attention import flash_attention_varlen


# ─── Benchmarking helpers ────────────────────────────────────────────

def bench_padded_dense(lengths, D, H, n_warmup=3, n_iters=10):
    """Pad all sequences to max length, run dense attention."""
    max_len = max(lengths)
    B = len(lengths)
    q = mx.random.normal((B, H, max_len, D), dtype=mx.float16)
    k = mx.random.normal((B, H, max_len, D), dtype=mx.float16)
    v = mx.random.normal((B, H, max_len, D), dtype=mx.float16)
    mx.eval(q, k, v)

    for _ in range(n_warmup):
        o = flash_attention(q, k, v)
        mx.eval(o)
    times = []
    for _ in range(n_iters):
        mx.synchronize()
        t0 = time.perf_counter()
        o = flash_attention(q, k, v)
        mx.eval(o)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000


def bench_varlen_packed(lengths, D, H, n_warmup=3, n_iters=10):
    """Pack all sequences contiguously, use varlen attention."""
    total_tokens = sum(lengths)
    q_packed = mx.random.normal((1, H, total_tokens, D), dtype=mx.float16)
    k_packed = mx.random.normal((1, H, total_tokens, D), dtype=mx.float16)
    v_packed = mx.random.normal((1, H, total_tokens, D), dtype=mx.float16)
    cu_seqlens = mx.array([0] + [int(x) for x in np.cumsum(lengths)], dtype=mx.int32)
    max_seqlen = max(lengths)
    mx.eval(q_packed, k_packed, v_packed)

    for _ in range(n_warmup):
        o = flash_attention_varlen(
            q_packed, k_packed, v_packed,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        )
        mx.eval(o)
    times = []
    for _ in range(n_iters):
        mx.synchronize()
        t0 = time.perf_counter()
        o = flash_attention_varlen(
            q_packed, k_packed, v_packed,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        )
        mx.eval(o)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2] * 1000


# ─── cu_seqlens rebuild cost ─────────────────────────────────────────

def bench_cu_seqlens_rebuild(n_sequences=4, n_iters=1000):
    """Measure cu_seqlens construction time."""
    lengths = [int(np.random.randint(1000, 4096)) for _ in range(n_sequences)]
    t0 = time.perf_counter()
    for _ in range(n_iters):
        cu = mx.array([0] + [int(x) for x in np.cumsum(lengths)], dtype=mx.int32)
        mx.eval(cu)
    total = (time.perf_counter() - t0) / n_iters * 1000
    return total


# ─── Correctness check ──────────────────────────────────────────────

def test_varlen_correctness(D, H):
    """Verify varlen matches per-sequence dense calls."""
    mx.random.seed(42)
    lengths = [512, 1024, 768]
    errors = []

    # Generate per-sequence QKV and compute reference outputs
    per_seq_q, per_seq_k, per_seq_v = [], [], []
    per_seq_out = []
    for n in lengths:
        q = mx.random.normal((1, H, n, D), dtype=mx.float16)
        k = mx.random.normal((1, H, n, D), dtype=mx.float16)
        v = mx.random.normal((1, H, n, D), dtype=mx.float16)
        mx.eval(q, k, v)
        out = flash_attention(q, k, v)
        mx.eval(out)
        per_seq_q.append(q)
        per_seq_k.append(k)
        per_seq_v.append(v)
        per_seq_out.append(out)

    # Pack into varlen format: concatenate along seq dim (axis=2)
    q_packed = mx.concatenate(per_seq_q, axis=2)
    k_packed = mx.concatenate(per_seq_k, axis=2)
    v_packed = mx.concatenate(per_seq_v, axis=2)
    cu_seqlens = mx.array([0] + [int(x) for x in np.cumsum(lengths)], dtype=mx.int32)
    mx.eval(q_packed, k_packed, v_packed)

    varlen_out = flash_attention_varlen(
        q_packed, k_packed, v_packed,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
    )
    mx.eval(varlen_out)

    # Compare each sequence's output
    offset = 0
    for i, n in enumerate(lengths):
        seq_out = varlen_out[:, :, offset : offset + n, :]
        ref_out = per_seq_out[i]
        max_err = float(
            mx.max(mx.abs(seq_out.astype(mx.float32) - ref_out.astype(mx.float32)))
        )
        errors.append((i, n, max_err))
        offset += n
    return errors


# ─── Scenarios ───────────────────────────────────────────────────────

SCENARIOS = [
    {
        "label": "Uniform merge 50% (DiT, 4 seqs)",
        "lengths": [2048, 2048, 2048, 2048],
    },
    {
        "label": "Variable merge 30-50% (DiT, 4 seqs)",
        "lengths": [2867, 2048, 3277, 2458],
    },
    {
        "label": "Heavy prune 70% (aggressive, 4 seqs)",
        "lengths": [1228, 1228, 1228, 1228],
    },
    {
        "label": "Single sequence post-merge (CogVideoX)",
        "lengths": [49140],
    },
    {
        "label": "Two sequences unequal (SeedVR2 batching)",
        "lengths": [20000, 6730],
    },
]


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    for D in (64, 128):
        H = 8
        print("=" * 70)
        print(f"Varlen vs Padded Dense — Token Merging Scenarios (D={D}, H={H})")
        print("=" * 70)

        results = []
        for scenario in SCENARIOS:
            label = scenario["label"]
            lengths = scenario["lengths"]
            total = sum(lengths)
            max_len = max(lengths)

            t_padded = bench_padded_dense(lengths, D=D, H=H)
            t_varlen = bench_varlen_packed(lengths, D=D, H=H)
            speedup = t_padded / t_varlen if t_varlen > 0 else float("inf")
            results.append((label, lengths, total, max_len, t_padded, t_varlen, speedup))

            print(f"\n{label}")
            print(f"  Lengths: {lengths} (total={total}, max={max_len})")
            print(f"  Padded dense: {t_padded:.2f}ms")
            print(f"  Varlen packed: {t_varlen:.2f}ms")
            print(f"  Speedup: {speedup:.2f}x")

        # cu_seqlens rebuild cost
        print("\n" + "-" * 70)
        cu_time = bench_cu_seqlens_rebuild(n_sequences=4)
        print(f"cu_seqlens rebuild (4 seqs): {cu_time:.4f}ms")
        cu_time_8 = bench_cu_seqlens_rebuild(n_sequences=8)
        print(f"cu_seqlens rebuild (8 seqs): {cu_time_8:.4f}ms")

        # Correctness
        print("\n" + "-" * 70)
        print(f"Correctness check (D={D}):")
        errs = test_varlen_correctness(D=D, H=H)
        all_ok = True
        for i, n, e in errs:
            status = "PASS" if e < 0.01 else "FAIL"
            if e >= 0.01:
                all_ok = False
            print(f"  Seq {i} (N={n}): max_err={e:.6f} {status}")
        print(f"  Overall: {'PASS' if all_ok else 'FAIL'}")
        print()
