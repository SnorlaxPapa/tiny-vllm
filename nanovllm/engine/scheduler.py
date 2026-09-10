from .block_manager import BlockManager
from .sequence import Sequence, SequenceStatus
from nanovllm.config import Config
from collections import deque

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_batch_tokens = config.max_num_batch_tokens
        self.max_num_seq = config.max_num_seqs
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kv_cache_block_size)
        self.block_size = config.kv_cache_block_size
        self.waiting = deque([])
        self.running = deque([])
        Sequence.block_size = self.block_size


    def preempt(self, seq: Sequence):
        """for simplicity, i used full eviction with (hopefully partial) recomputation"""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)


    def add(self, seq: Sequence):
        self.waiting.append(seq)

        
    def schedule(self):
        """for simplicity, i utilize a homogenous scheduler """
        scheduled_sequences = []
        batched_tokens = 0

        while len(scheduled_sequences) < self.max_num_seq and self.waiting:
            curr = self.waiting[0]
            remaining = self.max_num_batch_tokens - batched_tokens
            if remaining <= 0: break

            #determine if curr already has allocated blocks and if it does not, we check if we can allocate
            if not curr.block_list:
                cached_blocks, _ = self.block_manager.can_allocate(curr)
                if cached_blocks == -1: break
                num_tokens = len(curr) - (len(cached_blocks) * self.block_size)
            #if it does, means was part of a chunked
            else:
                num_tokens = len(curr) - curr.num_computed_tokens
            if not curr.block_list:
                self.block_manager.allocate(curr)

            #chunk and update
            num_tokens = min(num_tokens, remaining)
            curr.num_scheduled_tokens = num_tokens
            batched_tokens += num_tokens

            if curr.num_computed_tokens + curr.num_scheduled_tokens == len(curr):
                curr.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(curr)
            scheduled_sequences.append(curr)

        if scheduled_sequences:
            return scheduled_sequences, True

        #no prefill, we run for decode
        while self.running and len(scheduled_sequences) < self.max_num_seq:
            curr = self.running.popleft()
            while self.running and not self.block_manager.can_append(curr):
                self.preempt(self.running.pop())

            #if we cannot fit new token in, break out, else add to scheduled sequence
            if not self.block_manager.can_append(curr):
                self.preempt(curr)
                break

            curr.num_scheduled_tokens = 1
            self.block_manager.may_append(curr)
            scheduled_sequences.append(curr)

        self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False

    def postprocess(self, scheduled_sequences: list[Sequence], output_tokens: list[int], is_prefill: bool):
        """remove any sequences that have completed their runs"""
        for seq, token in zip(scheduled_sequences, output_tokens):
            seq.num_computed_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            self.block_manager.hash_blocks(seq)

            if is_prefill and seq.num_computed_tokens != len(seq): continue #skip appending decoded token for chunks
            seq.append_token(token)

            reached_completed = not seq.ignore_eos and seq.last_token == self.eos
            reached_limit = seq.num_completion_tokens == seq.max_tokens
            if reached_completed or reached_limit:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
