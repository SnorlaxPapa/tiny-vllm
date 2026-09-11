import torch
import torch.nn as nn
import torch.nn.functional as F
from nanovllm.model.block import Qwen2DecoderBlock
from nanovllm.layers.norm import RMSNorm

class Qwen2Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderBlock(config) for _ in range(config.num_hidden_layers)] 
        )

        self.norm = RMSNorm(config)

    def forward(self, input_ids, positions, sample_indices):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, positions)
            
        x = x[sample_indices]
        x = self.norm(x, None)

        logits = F.linear(
            x,
            self.embed_tokens.weight,
            bias=None
        )

        return logits

        