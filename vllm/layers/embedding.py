import torch
import torch.nn as nn

class EmbeddingLayer(nn.module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.tensor:
        return self.embedding(input_ids)