"""Validate architecture/weight mapping against HF on CPU.

Only for these tests, replace CUDA FlashAttention with PyTorch SDPA and disable
compilation. Real kernels/CUDA graphs are covered by validate_gpu.py on Colab.
"""
import importlib
import importlib.machinery
import sys
import types

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def cpu_modules(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda f=None, **kw: f if f else lambda x: x)
    triton = types.ModuleType("triton")
    triton.__spec__ = importlib.machinery.ModuleSpec("triton", loader=None)
    triton.jit = lambda f: f
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    flash = types.ModuleType("flash_attn")

    def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, **kwargs):
        pieces = []
        for i in range(len(cu_seqlens_q)-1):
            a, b = int(cu_seqlens_q[i]), int(cu_seqlens_q[i+1])
            c, d = int(cu_seqlens_k[i]), int(cu_seqlens_k[i+1])
            out = torch.nn.functional.scaled_dot_product_attention(
                q[a:b].transpose(0, 1).unsqueeze(0),
                k[c:d].transpose(0, 1).unsqueeze(0),
                v[c:d].transpose(0, 1).unsqueeze(0),
                is_causal=True, enable_gqa=True, scale=softmax_scale)
            pieces.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(pieces)

    flash.flash_attn_varlen_func = varlen
    flash.flash_attn_with_kvcache = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("CPU test should not use KV kernel"))
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setitem(sys.modules, "flash_attn", flash)
    from nanovllm.utils.model_loader import load_model
    from nanovllm.utils.context import set_context, reset_context
    from nanovllm.layers.rope import get_rope
    yield load_model, set_context
    reset_context()
    get_rope.cache_clear()


@pytest.mark.parametrize("tied,shard_size", [(True, "10GB"), (False, "10KB")])
def test_qwen3_loader_logits_and_causal_padding(tmp_path, cpu_modules, tied, shard_size):
    load_model, set_context = cpu_modules
    torch.manual_seed(42)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16,
                     max_position_embeddings=64, tie_word_embeddings=tied)
    reference = Qwen3ForCausalLM(cfg).eval()
    reference.save_pretrained(tmp_path, max_shard_size=shard_size)
    model = load_model(str(tmp_path), device="cpu", dtype=torch.float32)
    assert model.layers[0].self_attn.head_dim == 16  # hidden_size / heads is only 8!
    assert model.layers[0].self_attn.qkv_proj.linear.bias is None
    ids = torch.randint(0, cfg.vocab_size, (1, 11))
    def forward(tokens, sample):
        n = tokens.numel()
        cu = torch.tensor([0, n], dtype=torch.int32)
        set_context(is_prefill=True, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=n, max_seqlen_k=n)
        return model(tokens.flatten(), torch.arange(n), torch.tensor([sample]))
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits[:, -1]
        actual = forward(ids, 10)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        padded = torch.cat([ids, torch.randint(0, 128, (1, 9))], dim=1)
        torch.testing.assert_close(forward(padded, 10), actual, atol=1e-5, rtol=1e-4)


def test_public_api_metrics_exclude_ttft_from_itl(cpu_modules, monkeypatch):
    from types import SimpleNamespace
    import nanovllm.engine.engine as module
    from nanovllm.sampling_params import SamplingParams

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.config = SimpleNamespace(use_kv_cache=False, enable_prefix_caching=False,
                max_num_batch_tokens=20, max_num_seqs=1, max_model_len=20,
                num_kvcache_blocks=0, kv_cache_block_size=256, eos=99,
                hf_config=SimpleNamespace(vocab_size=100))
            self.graph_replays = 0

        def run(self, sequences, prefill):
            self.graph_replays += int(not prefill)
            return [7] * len(sequences)

    monkeypatch.setattr(module, "ModelRunner", FakeRunner)
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda path: SimpleNamespace(eos_token_id=99))
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved", "memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 100)
    times = iter([0., 1., 3., 6.])
    monkeypatch.setattr(module, "perf_counter", lambda: next(times))
    engine = module.EngineCore("unused", benchmark=True)
    ttft, itl = engine.generate([[1, 2]], SamplingParams(max_tokens=3, ignore_eos=True))
    assert ttft == 1.
    assert itl == 2.5
    assert engine.last_metrics["output_tokens_per_s"] == .5
    assert engine.last_metrics["graph_replays"] == 2
    assert engine.output_token_ids == [[7, 7, 7]]
    assert [e["itl_s"] for e in engine.last_metrics["token_events"]] == [None, 2., 3.]
