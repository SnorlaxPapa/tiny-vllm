from enum import Enum, auto
from copy import copy
from itertools import count
from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
        WAITING = auto()
        RUNNING = auto()
        FINISHED = auto()


class Sequence:

    block_size=256
    counter = count()

    def __init__(self, token_ids, sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.block_list = []
        self.token_ids = copy(token_ids)
        self.num_hashed_tokens = 0
        self.status = SequenceStatus.WAITING
        self.num_scheduled_tokens = 0
        self.num_prompt_tokens = len(token_ids)
        self.num_computed_tokens = 0
        self.last_token = token_ids[-1]
        self.max_tokens = sampling_params.max_tokens
        self.temperature = sampling_params.temperature
        self.ignore_eos = sampling_params.ignore_eos
        

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
        return (len(self) + self.block_size - 1) // self.block_size

    @property
    def num_completion_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens

    def append_token(self, token):
        self.token_ids.append(token)
        self.last_token = token