import triton
import triton.language as tl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.layers.rope import get_rope



@triton.jit
def store_kv_inner(
    k,
    v, 
    k_token_stride,
    v_token_stride,
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

    k_idx = k + k_token_stride * idx
    k_idx = k_idx + matrix_offset

    v_idx = v + v_token_stride * idx
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
    k_token_stride, v_token_stride = key.stride(0), value.stride(0)

    store_kv_inner[(T, 1, 1)](
        k=key,
        v=value, 
        k_token_stride=k_token_stride,
        v_token_stride=v_token_stride,
        H=H,
        D=D,
        k_cache=k_cache,
        v_cache=v_cache,
        slot_mapping=slot_mapping,
    )


class AttentionInterface(nn.Module):

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


class QKVProjection(nn.Module):
    def __init__(
        self,
        hidden_shape,
        num_attention_heads,
        num_kv_heads,
        head_dim,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.linear = nn.Linear(hidden_shape, (num_attention_heads + 2 * num_kv_heads) * head_dim, bias=True)


    def forward(self, hidden):
        qkv_proj = self.linear(hidden)
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        q, k ,v = qkv_proj.split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(-1, self.num_attention_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        return q, k, v



class Qwen2Attention(nn.Module):
    
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        self.qkv_proj = QKVProjection(self.hidden_size, config.num_attention_heads, config.num_key_value_heads, self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.attn = AttentionInterface(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        self.rotate = get_rope(self.head_dim, self.head_dim, self.max_position_embeddings, self.rope_theta)


    @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ):
        #get qkv projections
        q, k, v = self.qkv_proj(hidden_states)

        #rotate qk
        q, k = self.rotate(positions, q, k)

        #apply attention and return output
        o = self.attn(q, k, v)
        o = o.flatten(1, 2)
        o = self.o_proj(o)

        return o

        

