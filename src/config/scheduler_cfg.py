from dataclasses import dataclass

@dataclass(slots=True)
class SchedulerConfig:
    max_num_seqs: int = None
    max_num_batched_tokens: int = None
    max_seq_len: int = None
    eos: int = None
    num_kvcache_blocks: int = None
    kvcache_block_size: int = None
    chunked_prefill_size: int = None

    def __post_init__(self):
        assert self.max_num_seqs, "max_num_seqs must be specified in SchedulerConfig"
        assert self.max_num_batched_tokens, "max_num_batched_tokens must be specified in SchedulerConfig"
        assert self.max_seq_len, "max_seq_len must be specified in SchedulerConfig"
        assert self.eos is not None, "eos token id must be specified in SchedulerConfig"
        assert self.num_kvcache_blocks, "num_kvcache_blocks must be specified in SchedulerConfig"
        assert self.kvcache_block_size, "kvcache_block_size must be specified in SchedulerConfig"
        assert self.chunked_prefill_size, "chunked_prefill_size must be specified in SchedulerConfig"