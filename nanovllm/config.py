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
        use_kv_cache: bool = True,
        enable_prefix_caching: bool = True,
        no_cache_graph_capacity: int | None = None,
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
        self.use_kv_cache = use_kv_cache
        self.enable_prefix_caching = enable_prefix_caching and use_kv_cache
        self.no_cache_graph_capacity = no_cache_graph_capacity or max_model_len
        if min(max_num_batch_tokens, max_num_seqs, max_model_len) < 1:
            raise ValueError("Token budgets, sequence count and context length must be positive")
        if not use_kv_cache and not enforce_eager and max_num_seqs != 1:
            raise ValueError("No-cache CUDA graphs currently support max_num_seqs=1")

        assert os.path.isdir(self.model)
        assert self.kv_cache_block_size % 256 == 0

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings,
        )
        if not self.use_kv_cache:
            if self.max_num_batch_tokens < self.max_model_len:
                raise ValueError("No-cache recomputation requires a budget >= max_model_len")
            if not self.enforce_eager and not (self.max_model_len <= self.no_cache_graph_capacity <= self.hf_config.max_position_embeddings):
                raise ValueError("No-cache graph capacity must cover max_model_len and fit model context")
