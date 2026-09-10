import torch
import torch.nn
from transformers import AutoConfig

from nanovllm.engine.scheduler import Scheduler
from nanovllm.model.qwen2 import Qwen2Model
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context
from nanovllm.utils.model_loader import load_model


class ModelRunner:

  def __init__(self, model_dir: str, num_blocks: int, block_size: int):
    """initiate model, warm up, allocate kv cache for each attention layer"""
    self.config = AutoConfig.from_pretrained(model_dir)
    self.block_size = block_size
    self.num_blocks = num_blocks
    self.model = load_model(model_dir)
    self.sampler = Sampler()
    self.warmup_model()
    self.allocate_kv_cache()


  def warmup_model(self):
      pass


  def allocate_kv_cache(self):
    """allocate kv cache"""
    head_dim = self.config.hidden_size // self.config.num_attention_heads
    kv_cache_shape = (self.num_blocks, self.block_size, self.config.num_key_value_heads, head_dim)
    dtype = next(self.model.parameters()).dtype

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
        block_idx = position // self.block_size
        offset = position % self.block_size

        physical_block = sequence.block_list[block_idx] 
        slot = physical_block * self.block_size + offset
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
      block_idx = position // self.block_size
      offset = position % self.block_size
      physical_block = sequence.block_list[block_idx]
      physical_idx = physical_block * self.block_size + offset
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








