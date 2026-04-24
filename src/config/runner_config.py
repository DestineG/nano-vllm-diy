import os
from dataclasses import dataclass
import torch

@dataclass(slots=True)
class RunnerConfig:
    max_batched_seq_len: int = 16384
    max_num_seqs: int = 512
    _max_seq_len: int = 8192
    chunked_prefill_size: int = 512

    tensor_parallel_size: int = 2
    enforce_eager: bool = False

    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # 由外部传入的模型配置参数
    model_path: str = None
    dtype: torch.dtype = None
    max_seq_len: int = None
    num_key_value_heads: int = None
    head_dim: int = None
    num_hidden_layers: int = None
    model_dim: int = None

    def __post_init__(self):
        assert os.path.exists(self.model_path), f"Model path {self.model_path} does not exist"
        assert self.dtype, "dtype must be specified"
        assert self.max_seq_len, "max_seq_len must be specified"
        assert self.num_key_value_heads, "num_key_value_heads must be specified"
        assert self.head_dim, "head_dim must be specified"
        assert self.num_hidden_layers, "num_hidden_layers must be specified"
        assert self.model_dim, "model_dim must be specified"

        self.max_seq_len = min(self._max_seq_len, self.max_seq_len)

        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8