from enum import Enum, auto
from copy import copy
from itertools import count


class SequenceStatus(Enum):
        WAITING = auto()
        RUNNING = auto()
        FINISHED = auto()


class Sequence:

    block_size=4
    counter = count()

    def __init__(self, token_ids):
        self.seq_id = next(Sequence.counter)
        self.block_list = []
        self.token_ids = copy(token_ids)
        self.num_hashed_tokens = 0
        self.status = SequenceStatus.WAITING
        self.num_scheduled_tokens = 0
        self.num_computed_tokens = 0
        self.num_tokens = len(token_ids)

    def __len__(self):
        return len(self.token_ids)

    def reset(self):
        self.block_list.clear()
        self.num_hashed_tokens = 0
        self.num_computed_tokens = 0

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size