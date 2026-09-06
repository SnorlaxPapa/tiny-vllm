import torch
import torch.nn as nn
from vllm.layers.mlp_layer import GatedLinear
from vllm.layers.norm import RMSNorm
from vllm.layers.attention import Qwen2Attention

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gated_linear = GatedLinear(config)
        self.rmsnorm = RMSNorm(config)
        self.attention = Qwen2Attention(config)


    def forward(self, x):
        pass



