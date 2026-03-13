from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_mfa import (
    InferenceContext,
    PagedInferenceContext,
    SageInferenceContext,
    create_decode_runtime,
    adapt_kv_cache,
    HybridKVCache,
    KVCacheOperationUnsupported,
)


class TestKVCacheAdapters:
    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(42)

    def test_dense_cache_adapter_attention_view(self):
        ctx = InferenceContext(B=1, H_kv=4, D=64, max_seq_len=64)
        q = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        _ = ctx.prefill(q, k, v)

        ad = ctx.cache_adapter
        k_attn = ad.attention_k(0)
        v_attn = ad.attention_v(0)
        mx.eval(k_attn, v_attn)

        assert ad.kind == "dense"
        assert ad.capabilities.attention_view is True
        assert ad.capabilities.paged_pool is False
        assert ad.seq_length(0) == 8
        assert k_attn.shape == (1, 4, 8, 64)
        assert v_attn.shape == (1, 4, 8, 64)

    def test_paged_cache_adapter_pool_and_tables(self):
        ctx = PagedInferenceContext(
            num_blocks=64,
            block_size=16,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        _ = ctx.prefill(q, k, v, seq_id=11)

        ad = ctx.cache_adapter
        k_pool, v_pool, block = ad.paged_pool()
        table, lens = ad.paged_tables([11])
        mx.eval(k_pool, v_pool, table, lens)

        assert ad.kind == "paged"
        assert ad.capabilities.paged_pool is True
        assert ad.capabilities.multi_seq is True
        assert block == 16
        assert ad.seq_length(11) == 6
        assert tuple(ad.active_seq_ids()) == (11,)
        assert table.shape[0] == 1
        assert lens.shape == (1,)

    def test_quantized_cache_adapter_quantized_view(self):
        ctx = SageInferenceContext(B=1, H_kv=4, D=64, max_seq_len=64)
        q = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        _ = ctx.prefill(q, k, v)

        ad = ctx.cache_adapter
        k_int8, k_scale, v_hist = ad.quantized_view()
        mx.eval(k_int8, k_scale, v_hist)

        assert ad.kind == "quantized"
        assert ad.capabilities.quantized_view is True
        assert ad.capabilities.attention_view is False
        assert ad.seq_length(0) == 8
        assert k_int8.shape[2] == 8
        assert v_hist.shape == (1, 4, 8, 64)

    def test_unsupported_ops_fail_clearly(self):
        dense_ctx = InferenceContext(B=1, H_kv=4, D=64, max_seq_len=32)
        dense_ad = dense_ctx.cache_adapter
        with pytest.raises(KVCacheOperationUnsupported, match="paged pools"):
            dense_ad.paged_pool()

        sage_ctx = SageInferenceContext(B=1, H_kv=4, D=64, max_seq_len=32)
        q_ad = sage_ctx.cache_adapter
        with pytest.raises(KVCacheOperationUnsupported, match="attention K view"):
            q_ad.attention_k(0)

    def test_adapt_kv_cache_hybrid_skeleton(self):
        dense_ctx = InferenceContext(B=1, H_kv=4, D=64, max_seq_len=32)
        hybrid = HybridKVCache(dense_ctx._cache, policy="manual")
        ad = adapt_kv_cache(hybrid)
        assert ad.kind == "hybrid"
        assert ad.capabilities.attention_view is True
        assert hybrid.ready_for_production is True
        assert hybrid.state["hot_seq_capacity"] == 1
        assert hybrid.state["residency_map"] == {}
        with pytest.raises(NotImplementedError, match="future work"):
            hybrid.offload_seq(0)


class TestCacheAbstractionRuntimeFlows:
    @pytest.fixture(autouse=True)
    def _seed(self):
        mx.random.seed(123)

    def test_runtime_metadata_exposes_cache_kind_and_capabilities(self):
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        md = rt.metadata
        assert md["cache_kind"] == "dense"
        assert md["cache_capabilities"]["attention_view"] is True

    def test_prefix_flow_still_works_with_cache_adapter_path(self):
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        k_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        v_pre = mx.random.normal((1, 4, 8, 64)).astype(mx.float16)
        rt.register_prefix("p0", q_pre, k_pre, v_pre)
        rt.seed_prefix(prefix_id="p0", reset=True)

        q_suf = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        k_suf = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        v_suf = mx.random.normal((1, 4, 4, 64)).astype(mx.float16)
        out = rt.chunked_prefill(
            q_suf,
            k_suf,
            v_suf,
            chunk_size=2,
            causal=True,
            reset=False,
        )
        mx.eval(out)
        assert out.shape == (1, 4, 4, 64)
        assert rt.seq_length() == 12

    def test_speculative_step_still_works_with_adapter_fallback(self):
        rt = create_decode_runtime(
            backend="dense",
            quantized_kv=False,
            B=1,
            H_kv=4,
            D=64,
        )
        q = mx.random.normal((1, 4, 12, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 12, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 12, 64)).astype(mx.float16)
        rt.prefill(q, k, v)

        q_target = mx.random.normal((1, 4, 3, 64)).astype(mx.float16)
        draft_ids = mx.array([[0, 1, 2]], dtype=mx.int32)
        out = rt.speculative_step(q_target, draft_ids, accept_logprob_delta=-1e9)
        mx.eval(out["accepted_prefix_lens"], out["accepted_ids"], out["rejected_ids"])
        assert tuple(out["accepted_prefix_lens"].tolist()) == (3,)

    def test_paged_chunked_prefill_still_works_with_adapter_tables(self):
        rt = create_decode_runtime(
            backend="paged",
            paged=True,
            query_layout="batched",
            quantized_kv=False,
            B=1,
            H_q=4,
            H_kv=4,
            D=64,
            num_blocks=64,
            block_size=16,
        )
        q = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        k = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        v = mx.random.normal((1, 4, 6, 64)).astype(mx.float16)
        out = rt.chunked_prefill(
            q,
            k,
            v,
            chunk_size=2,
            seq_ids=[5],
            causal=True,
            reset=True,
        )
        mx.eval(out)
        assert out.shape == (1, 4, 6, 64)
        assert rt.seq_length(5) == 6
