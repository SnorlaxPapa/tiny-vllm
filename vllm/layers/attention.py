import triton
import triton.language as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from vllm.utils.context import get_context

import inspect
from flash_attn import flash_attn_varlen_func

print(inspect.signature(flash_attn_varlen_func))

@triton.jit
def store_kv_inner(
    k,
    v, 
    H: tl.constexpr,
    D: tl.constexpr,
    k_cache,
    v_cache,
    slot_mapping,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping + idx)

    if slot == -1: #for fake tokens for cuda graph
        return

    rows = tl.arange(0, H)
    cols = tl.arange(0, D)
    matrix_offset = rows[:, None] * D + cols[None, :]

    k_idx = k + H * D * idx
    k_idx = k_idx + matrix_offset

    v_idx = v + H * D * idx
    v_idx = v_idx + matrix_offset

    k = tl.load(k_idx)
    v = tl.load(v_idx)

    k_cache_idx = (
        k_cache + 
        slot * H * D +
        matrix_offset
    )

    v_cache_idx = (
        v_cache + 
        slot * H * D +
        matrix_offset
    )

    tl.store(k_cache_idx, k)
    tl.store(v_cache_idx, v)




def store_kv_cache(
    key: torch.Tensor, #(T, H, D)
    value: torch.Tensor, #(T, H, D)
    k_cache: torch.Tensor, #(num_blocks, block_size, H, D)
    v_cache: torch.Tensor, #(num_blocks, block_size, H, D)
    slot_mapping: torch.Tensor, #(T, )
):
    T, H, D = key.shape
    store_kv_inner[(T, 1, 1)](
        k=key,
        v=value, 
        H=H,
        D=D,
        k_cache=k_cache,
        v_cache=v_cache,
        slot_mapping=slot_mapping,
    )


    
class attentionInterface(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """given q k v and context, if prefill use fa, decode use paged attention"""
        context = get_context()

        if self.k_cache.numel() and self.v_cache.numel(): #kv cache blocks have been allocated for this layer
            store_kv_cache(
                key=k,
                value=v,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                slot_mapping=context.slot_mapping,
            )

        if context.is_prefill:
            #prefill, use variable length fa with packed input
            if context.block_tables is not None:    # cache exists, utilize it
                k, v = self.k_cache, self.v_cache
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, causal=True, block_table=context.block_tables
            )
        else:
            #decode, one token per step, use fa with kv cache
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), self.k_cache, self.v_cache,
                cache_seqlens=context.context_lens, block_table=context.block_tables, 
                softmax_scale=self.scale, causal=True
            ).squeeze(1)

        return o