import torch
import torch.nn
from transformers import AutoConfig

from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, reset_context, get_context
from nanovllm.utils.model_loader import load_model
from nanovllm.config import Config


class ModelRunner:

  def __init__(self, model_dir: str):
    """initiate model, warm up, allocate kv cache for each attention layer"""
    self.hf_config = AutoConfig.from_pretrained(model_dir)
    self.config = Config(model_dir)
    self.model = load_model(model_dir)
    self.sampler = Sampler()
    self.warmup_model()
    self.allocate_kv_cache()
    if not self.config.enforce_eager:
        self.capture_cuda_graph()


  def warmup_model(self):
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats() #reset and free up memory

      token_budget = self.config.max_num_batch_tokens
      seq_len = min(token_budget, self.config.max_model_len)

      num_sequences = min(token_budget // seq_len, self.config.max_num_seqs)

      scheduled_sequences = [Sequence([0] * seq_len) for _ in range(num_sequences)]

      for sequence in scheduled_sequences:
        sequence.num_scheduled_tokens = seq_len

      self.run(scheduled_sequences, True)

      torch.cuda.synchronize()
      torch.cuda.empty_cache()



  def allocate_kv_cache(self):
    """allocate kv cache"""
    free, total = torch.cuda.mem_get_info() #free and total vram
    used = total - free #total vram currently used
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] #peak memory allocated for torch tensors
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"] #current memory allocated for torch tensors

    head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
    dtype = next(self.model.parameters()).dtype

    block_bytes = (
        2 *
        self.config.kv_cache_block_size *
        self.hf_config.num_key_value_heads *
        self.hf_config.num_hidden_layers *
        head_dim *
        dtype.itemsize
    )

    self.config.num_kvcache_blocks = int(total * self.config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert self.config.num_kvcache_blocks > 0, "Unable to allocate kv blocks, consider increasing gpu_memory_utilization"

    kv_cache_shape = (self.config.num_kvcache_blocks, self.config.kv_cache_block_size, self.hf_config.num_key_value_heads, head_dim)
    for layer in self.model.layers:
      layer.self_attn.attn.k_cache = torch.empty(kv_cache_shape, dtype=dtype, device="cuda")
      layer.self_attn.attn.v_cache = torch.empty(kv_cache_shape, dtype=dtype, device="cuda")


  def prepare_block(self, scheduled_sequences: list[Sequence]):
    """prepares context block table"""
    block_tables = []
    max_block_len = max(len(sequence.block_list) for sequence in scheduled_sequences)

    for sequence in scheduled_sequences:
      diff = max_block_len - len(sequence.block_list)
      block_tables.append(sequence.block_list.copy() + diff * [-1])

    return block_tables


  def prepare_prefill(self, scheduled_sequences: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
    """intiialize context given current scheduled sequences"""
    packed_tokens = []
    cu_seqlen_q = [0]
    cu_seqlen_k = [0]
    positions = []
    slot_mappings = []
    block_tables = None
    max_seqlen_q = 0
    max_seqlen_k = 0

    for sequence in scheduled_sequences:
      #get our scheduled tokens
      start = sequence.num_computed_tokens
      end = start + sequence.num_scheduled_tokens
      packed_tokens.extend(sequence.token_ids[start: end])

      #configure max_seqlen_q k
      max_seqlen_q = max(max_seqlen_q, sequence.num_scheduled_tokens)
      max_seqlen_k = max(max_seqlen_k, end)

      #get cu_seqlen_qk
      query_length = sequence.num_scheduled_tokens

      cu_seqlen_q.append(cu_seqlen_q[-1] + query_length)
      cu_seqlen_k.append(cu_seqlen_k[-1] + end)

      #get positions of each token
      sequence_positions = range(start, end)
      positions.extend(sequence_positions)

      #get slot mapping
      if not sequence.block_list: continue #warm up

      for position in sequence_positions:
        block_idx = position // self.config.kv_cache_block_size
        offset = position % self.config.kv_cache_block_size

        physical_block = sequence.block_list[block_idx]
        slot = physical_block * self.config.kv_cache_block_size + offset
        slot_mappings.append(slot)

    #get physical block tables per sequence, must be uniform dim (seq, max_block_list)
    if cu_seqlen_k[-1] > cu_seqlen_q[-1]:
      block_tables = self.prepare_block(scheduled_sequences)
      block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    #move to gpu
    packed_tokens = torch.tensor(packed_tokens, dtype=torch.long, device="cuda") #long for embedding
    positions = torch.tensor(positions, dtype=torch.long, device="cuda") #long for indexing

    cu_seqlen_q = torch.tensor(cu_seqlen_q, dtype=torch.int32, device="cuda")
    cu_seqlen_k = torch.tensor(cu_seqlen_k, dtype=torch.int32, device="cuda")
    slot_mappings = torch.tensor(slot_mappings, dtype=torch.int32, device="cuda")

    set_context(
      is_prefill = True,
      cu_seqlens_q=cu_seqlen_q,
      cu_seqlens_k=cu_seqlen_k,
      max_seqlen_q=max_seqlen_q,
      max_seqlen_k=max_seqlen_k,
      slot_mapping=slot_mappings,
      block_tables=block_tables,
    )

    return packed_tokens, positions


  def prepare_decode(self, scheduled_sequences: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
    context_lens = []
    packed_tokens = []
    positions = []
    slot_mapping = []

    for sequence in scheduled_sequences:
      packed_tokens.append(sequence.last_token)
      position = len(sequence) -1

      #slot mapping
      block_idx = position // self.config.kv_cache_block_size
      offset = position % self.config.kv_cache_block_size
      physical_block = sequence.block_list[block_idx]
      physical_idx = physical_block * self.config.kv_cache_block_size + offset
      slot_mapping.append(physical_idx)


      positions.append(len(sequence) - 1)
      context_lens.append(len(sequence))

    packed_tokens = torch.tensor(packed_tokens, dtype=torch.long, device="cuda")
    positions = torch.tensor(positions, dtype=torch.long, device="cuda")

    context_lens = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    block_tables = self.prepare_block(scheduled_sequences)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device="cuda")

    set_context(
        is_prefill=False,
        context_lens=context_lens,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
    )

    return packed_tokens, positions


  def prepare_sample(self, scheduled_sequences: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = []
    sample_indices = []
    token_count = 0
    for sequence in scheduled_sequences:
      temperature.append(sequence.temperature)

      token_count += sequence.num_scheduled_tokens
      sample_indices.append(token_count - 1)

    temperature = torch.tensor(temperature, dtype=torch.float32, device="cuda")
    sample_indices = torch.tensor(sample_indices, dtype=torch.long, device="cuda")

    return sample_indices, temperature


  @torch.inference_mode()
  def run_prefill_eager_decode(
      self, 
      packed_tokens: torch.Tensor, 
      positions: torch.Tensor, 
      sample_indices: torch.Tensor, 
  ) -> torch.Tensor:

    return self.model(packed_tokens, positions, sample_indices)


  @torch.inference_mode()
  def run_cuda(
      self, 
      packed_tokens: torch.Tensor, 
      positions: torch.Tensor, 
  ) -> torch.Tensor:
    context = get_context()
    bs = packed_tokens.shape[0]
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
    graph_vars = self.graph_vars
    graph_vars["input_ids"][:bs] = packed_tokens
    graph_vars["positions"][:bs] = positions
    graph_vars["slot_mapping"].fill_(-1)
    graph_vars["slot_mapping"][:bs] = context.slot_mapping
    graph_vars["context_lens"].zero_()
    graph_vars["context_lens"][:bs] = context.context_lens
    graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
    graph.replay()

    return graph_vars["outputs"][:bs]


  def run(self, scheduled_sequences: list[Sequence], is_prefill: bool) -> list[int]:
    """split based on prefill/eager decode and cuda graph decode"""
    try:
      #prefill or eager decode 
      if is_prefill or self.config.enforce_eager:
        if is_prefill:
          packed_tokens, positions = self.prepare_prefill(scheduled_sequences)
        else:
          packed_tokens, positions = self.prepare_decode(scheduled_sequences)
        
        sample_indices, temperature = self.prepare_sample(scheduled_sequences)
        logits = self.run_prefill_eager_decode(packed_tokens, positions, sample_indices)

      else:
        packed_tokens, positions = self.prepare_decode(scheduled_sequences)
        sample_indices, temperature = self.prepare_sample(scheduled_sequences)
        logits = self.run_cuda(packed_tokens, positions)

      token_idxs = self.sampler(logits, temperature).tolist()

    finally:
      reset_context()
    
    return token_idxs



  @torch.inference_mode()
  def capture_cuda_graph(self):
    """capture decode cuda graphs for batches 1 -> max seq, pad for batches < 16"""
    #get max block size and batch size for kernel launches
    max_bs = min(self.config.max_num_seqs, 512)

    max_blocks_per_seq = (self.config.max_model_len + self.config.kv_cache_block_size - 1) // self.config.kv_cache_block_size

    #initialize our context tensors
    context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
    slot_mapping = torch.full(
      (max_bs,), -1, dtype=torch.int32, device="cuda"
    )
    block_tables = torch.zeros(max_bs, max_blocks_per_seq, dtype=torch.int32, device="cuda")

    #initialize tensors for model forward pass
    positions = torch.zeros(max_bs, dtype=torch.long, device="cuda")
    input_ids = torch.zeros(max_bs, dtype=torch.long, device="cuda")
    output = torch.zeros(max_bs, self.hf_config.vocab_size, dtype=next(self.model.parameters()).dtype, device="cuda")
    sample_indices = torch.arange(0, max_bs, dtype=torch.long, device="cuda")
    
    self.graph_bs = sorted({
        bs
        for bs in [1, 2, 4, 8, *range(16, max_bs + 1, 16), max_bs]
        if bs <= max_bs
    })

    self.graphs = {}
    self.graph_pool = None

    #set up side-stream for cuda graph warmup and capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(s):
      for bs in reversed(self.graph_bs):
        g = torch.cuda.CUDAGraph()
        
        set_context(
          is_prefill=False,
          context_lens=context_lens[:bs],
          slot_mapping=slot_mapping[:bs],
          block_tables=block_tables[:bs],
        )

        #warmup
        output[:bs] = self.model(input_ids[:bs], positions[:bs], sample_indices[:bs])

        #capture graph
        with torch.cuda.graph(g, self.graph_pool, stream=s):
          output[:bs] = self.model(input_ids[:bs], positions[:bs], sample_indices[:bs])
        if self.graph_pool is None:
          self.graph_pool = g.pool()
        self.graphs[bs] = g

        torch.cuda.synchronize()
        reset_context()
    
    torch.cuda.current_stream().wait_stream(s)

    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=output,
        sample_indices=sample_indices,
    )


      
