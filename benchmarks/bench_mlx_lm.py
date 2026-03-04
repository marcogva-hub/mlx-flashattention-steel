"""Benchmark mlx-mfa integration with mlx-lm models.

Measures tokens/second for prompt processing (prefill) and generation (decode)
with and without STEEL attention.

Requirements:
    pip install mlx-lm

Usage::

    # Baseline (stock mlx-lm SDPA):
    python benchmarks/bench_mlx_lm.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit

    # With STEEL attention:
    python benchmarks/bench_mlx_lm.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit --steel

    # Compare both side-by-side:
    python benchmarks/bench_mlx_lm.py \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit --compare
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx


def _make_prompt(approx_tokens: int) -> str:
    """Generate a repetitive prompt of approximately ``approx_tokens`` tokens."""
    # Each "the history of " ≈ 4 tokens; pad to target length.
    filler = "the history of " * (approx_tokens // 4 + 1)
    return "Tell me about " + filler


def bench_model(
    model_name: str,
    use_steel: bool = False,
    prompt_lengths: list[int] = (64, 256, 1024),
    gen_tokens: int = 100,
    n_runs: int = 3,
) -> list[dict]:
    """Run generation benchmark for one mode (STEEL or baseline).

    Args:
        model_name:     HuggingFace / mlx-community model ID.
        use_steel:      If True, patch mlx-lm with STEEL before loading model.
        prompt_lengths: Approximate prompt token counts to test.
        gen_tokens:     Number of tokens to generate per run.
        n_runs:         Number of timed runs (median reported).

    Returns:
        List of result dicts: {prompt_len, gen_tokens, median_time, tokens_per_sec, steel}.
    """
    if use_steel:
        from mlx_mfa.integrations.mlx_lm import patch_mlx_lm
        patch_mlx_lm()

    try:
        from mlx_lm import generate, load
    except ImportError as exc:
        raise ImportError(
            "mlx-lm is not installed. Install with: pip install mlx-lm"
        ) from exc

    model, tokenizer = load(model_name)
    label = "STEEL" if use_steel else " SDPA"

    results = []
    for prompt_len in prompt_lengths:
        prompt_text = _make_prompt(prompt_len)

        times = []
        for _ in range(n_runs):
            mx.synchronize()
            t0 = time.perf_counter()
            _ = generate(
                model,
                tokenizer,
                prompt=prompt_text,
                max_tokens=gen_tokens,
                verbose=False,
            )
            mx.synchronize()
            times.append(time.perf_counter() - t0)

        median_t = sorted(times)[len(times) // 2]
        tps = gen_tokens / median_t
        results.append(
            {
                "prompt_len": prompt_len,
                "gen_tokens": gen_tokens,
                "median_time": median_t,
                "tokens_per_sec": tps,
                "steel": use_steel,
            }
        )
        print(
            f"  [{label}] prompt≈{prompt_len:>5} gen={gen_tokens} "
            f"time={median_t:.2f}s  tps={tps:.1f}"
        )

    if use_steel:
        from mlx_mfa.integrations.mlx_lm import unpatch_mlx_lm
        unpatch_mlx_lm()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="mlx-mfa mlx-lm benchmark")
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="mlx-lm compatible model ID (default: Llama-3.2-3B-Instruct-4bit)",
    )
    parser.add_argument(
        "--steel",
        action="store_true",
        help="Use STEEL attention (mlx-mfa patch)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare baseline SDPA vs STEEL side-by-side",
    )
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=[64, 256, 1024],
        metavar="N",
        help="Approximate prompt token counts to test (default: 64 256 1024)",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=100,
        help="Tokens to generate per run (default: 100)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Timed runs per config (default: 3, median reported)",
    )
    args = parser.parse_args()

    print(f"Model:          {args.model}")
    print(f"Prompt lengths: {args.prompt_lengths}")
    print(f"Gen tokens:     {args.gen_tokens}")
    print(f"Runs per cfg:   {args.n_runs}")
    print()

    if args.compare:
        print("=== Baseline (MLX SDPA) ===")
        baseline = bench_model(
            args.model,
            use_steel=False,
            prompt_lengths=args.prompt_lengths,
            gen_tokens=args.gen_tokens,
            n_runs=args.n_runs,
        )
        print()
        print("=== STEEL (mlx-mfa) ===")
        steel = bench_model(
            args.model,
            use_steel=True,
            prompt_lengths=args.prompt_lengths,
            gen_tokens=args.gen_tokens,
            n_runs=args.n_runs,
        )
        print()
        print(f"{'Prompt≈':>8}  {'SDPA tps':>10}  {'STEEL tps':>10}  {'Speedup':>8}")
        print("-" * 44)
        for b, s in zip(baseline, steel):
            speedup = s["tokens_per_sec"] / b["tokens_per_sec"]
            print(
                f"{b['prompt_len']:>8}  {b['tokens_per_sec']:>10.1f}  "
                f"{s['tokens_per_sec']:>10.1f}  {speedup:>7.2f}×"
            )
    elif args.steel:
        bench_model(
            args.model,
            use_steel=True,
            prompt_lengths=args.prompt_lengths,
            gen_tokens=args.gen_tokens,
            n_runs=args.n_runs,
        )
    else:
        bench_model(
            args.model,
            use_steel=False,
            prompt_lengths=args.prompt_lengths,
            gen_tokens=args.gen_tokens,
            n_runs=args.n_runs,
        )


if __name__ == "__main__":
    main()
