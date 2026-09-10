import os
from transformers import AutoConfig


class Config:
    def __init__(
        self,
        model: str,
        max_num_batch_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = False,
        kv_cache_block_size: int = 256,
        num_kvcache_blocks: int = -1,
        eos: int = -1,
    ):
        self.model = model
        self.max_num_batch_tokens = max_num_batch_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.kv_cache_block_size = kv_cache_block_size
        self.num_kvcache_blocks = num_kvcache_blocks
        self.eos = eos

        assert os.path.isdir(self.model)
        assert self.kv_cache_block_size % 256 == 0

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings,
        )