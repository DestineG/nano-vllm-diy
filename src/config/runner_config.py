import os
from dataclasses import dataclass
import torch

@dataclass(slots=True)
class RunnerConfig:
    model_path: str

    max_batched_seq_len: int = 16384
    max_num_seqs: int = 512

    tensor_parallel_size: int = 1
    enforce_eager: bool = False

    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    dtype: torch.dtype = torch.float16
    max_seq_len: int = 4096
    num_key_value_heads: int = 16
    head_dim: int = 64
    num_hidden_layers: int = 16
    model_dim: int = 1024

    def __post_init__(self):
        assert os.path.isdir(self.model_path)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
