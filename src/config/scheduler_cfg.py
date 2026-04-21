from dataclasses import dataclass

@dataclass(slots=True)
class SchedulerConfig:
    max_num_seqs: int
    max_num_batched_tokens: int
    eos: int
    num_blocks: int
    block_size: int