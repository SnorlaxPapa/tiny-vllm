from collections import OrderedDict
from .sequence import Sequence
import xxhash
import numpy as np

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = None
        self.token_ids = []

    def reset(self):
        """clear old cached content"""
        self.hash = None
        self.token_ids = []

    def update(self, block_hash, token_ids):
        self.hash = block_hash
        self.token_ids = token_ids


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = OrderedDict(
            {i: None for i in range(num_blocks)}
        )
        self.hash_to_block_id = {}

    @staticmethod
    def get_hash(token_ids: list[int], previous_hash: int = -1):
        h = xxhash.xxh64()
        if previous_hash != -1:
            h.update(previous_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())

        return h.intdigest()
    

    def _get_free_block(self):
        """Take first free block for new content"""
        block_id, _ = self.free_block_ids.popitem(last=False)

        #reset
        block = self.blocks[block_id] 
        if self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()

        block.ref_count = 1 

        return block_id


    def _deallocate_block(self, block_id: int):
        """ add block to free blocks """
        assert self.blocks[block_id].ref_count == 0
        assert block_id not in self.free_block_ids

        self.free_block_ids[block_id] = None
        if self.blocks[block_id].hash is None:
            self.free_block_ids.move_to_end(block_id, last=False)


    def _find_longest_prefix(self, seq: Sequence):
        """Given a sequence, finds the longest cached prefix"""
        token_ids = seq.token_ids
        prefix = -1
        blocks = []

        steps = (len(token_ids) - 1) // self.block_size
        left = 0 
        right = self.block_size 
        for i in range(1, steps+1):
            current_tokens = token_ids[left: right]
            prefix = self.get_hash(current_tokens, prefix)

            #address hash collision
            block_id = self.hash_to_block_id.get(prefix)
            if block_id is None: return blocks
            if self.blocks[block_id].token_ids != current_tokens:
                return blocks

            left += self.block_size  
            right += self.block_size 
            blocks.append(block_id)

        return blocks


    def allocate(self, seq: Sequence):
        """Given longest number of cached blocks, allocate new blocks for the rest"""
        assert not seq.block_list

        #allocate cached prefix
        cached_blocks = self._find_longest_prefix(seq)
        cached_tokens = len(cached_blocks) * self.block_size
        uncached_tokens = len(seq) - cached_tokens

        inactive_cached_blocks = sum(
            block_id in self.free_block_ids
            for block_id in cached_blocks
        )   

        blocks_needed = (uncached_tokens + self.block_size - 1) // self.block_size

        if blocks_needed > len(self.free_block_ids) - inactive_cached_blocks: return -1

        for block_id in cached_blocks:
            seq.block_list.append(block_id)

            if block_id in self.free_block_ids:
                self.free_block_ids.pop(block_id)

            block = self.blocks[block_id]
            block.ref_count += 1

        seq.num_computed_tokens = cached_tokens
        seq.num_hashed_tokens = cached_tokens

        #allocate for uncached tokens
        for _ in range(blocks_needed):
            free_block = self._get_free_block()
            seq.block_list.append(free_block)        

        return 0
    

    def hash_blocks(self, seq: Sequence):
        """Hash and register fully filled blocks"""
        start = seq.num_hashed_tokens // self.block_size
        end = seq.num_computed_tokens // self.block_size

        if start == end: return 
        prefix = self.blocks[seq.block_list[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_list[i]]
            token_ids = seq.block(i)
            block_hash = self.get_hash(token_ids, prefix)

            #update hash
            block.update(block_hash, token_ids.copy())
            self.hash_to_block_id[block_hash] = block.block_id
            prefix = block_hash

        seq.num_hashed_tokens = end * self.block_size


    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_list): #reverse to allow earlier cached blocks to live longer
            block = self.blocks[block_id]
            assert block.ref_count > 0

            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.reset()



