from copy import copy
from enum import Enum, auto
from itertools import count

from src.config.sampling_params import SamplingParams

class SequenceStatus(Enum):
    PREFILL = auto()
    DECODE = auto()
    FINISHED = auto()

class Sequence:
    _id_counter = count()
    block_size = 256

    def __init__(
        self,
        token_ids: list[int],
        sampling_params = SamplingParams()
    ):
        self.token_ids = copy(token_ids)
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.seq_id = next(self._id_counter)
        self.status = SequenceStatus.PREFILL
        self.last_token_id = self.token_ids[-1]

        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)

        self.num_scheduled_tokens = 0

        self.num_cached_tokens = 0
        self.block_table = []
    
    def __len__(self):
        return self.num_tokens
    
    def __getitem__(self, key):
        return self.token_ids[key]
    
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_generated_token_ids(self):
        return self.num_tokens - self.num_prompt_tokens
    @property
    def generated_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    def append(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token_id = token_id
        self.num_tokens += 1

    def block(self, block_idx):
        assert block_idx < self.num_blocks, f"block_idx {block_idx} out of range for sequence with {self.num_blocks} blocks"
        start_idx = block_idx * self.block_size
        end_idx = min(start_idx + self.block_size, len(self.token_ids))
        return self.token_ids[start_idx:end_idx]
