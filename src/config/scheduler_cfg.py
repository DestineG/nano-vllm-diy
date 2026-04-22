from dataclasses import dataclass

@dataclass(slots=True)
class SchedulerConfig:
    max_num_seqs: int
    max_num_batched_tokens: int
    max_seq_len: int
    eos: int
    num_kvcache_blocks: int
    kvcache_block_size: int
    chunked_prefill_size: int