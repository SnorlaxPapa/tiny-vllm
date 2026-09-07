import torch
import torch.nn as nn
from nanovllm.layers.mlp_layer import GatedLinear
from nanovllm.layers.norm import RMSNorm
from nanovllm.layers.attention import Qwen2Attention

class Qwen2DecoderBlock(nn.Module):
    """ block structure rms norm -> attention -> rms norm (with residual) -> gated linear unit"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mlp = GatedLinear(config)
        self.input_layernorm = RMSNorm(config)
        self.post_attention_layernorm = RMSNorm(config)
        self.self_attn = Qwen2Attention(config)


    @torch.compile
    def forward(self, x, positions):
        #attention 
        residual = x
        hidden_states = self.input_layernorm(x, None)
        output = self.self_attn(hidden_states, positions)

        #mlp
        x, residual = self.post_attention_layernorm(output, residual) #allows fusion of rms norm and res (both row ops)
        x = self.mlp(x)
        x = x + residual #hard to fuse here cuz glu is matmul and res is addition by row

        return x
        



        



