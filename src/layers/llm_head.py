import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from src.utils.math import divide
from src.utils.context import Context
    
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp: tuple[int, int] = (0, 1)
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank, self.tp_world_size = tp

        self.num_embeddings_per_partition = divide(num_embeddings, self.tp_world_size)
        self.vocab_start_idx = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader
    
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor
    ):
        assert loaded_weight.shape[0] == self.num_embeddings, f"Expected weight shape[0] {self.num_embeddings}, but got {loaded_weight.shape[0]}"
        assert loaded_weight.shape[1] == self.embedding_dim, f"Expected weight shape[1] {self.embedding_dim}, but got {loaded_weight.shape[1]}"
        local_weight = loaded_weight.chunk(self.tp_world_size, 0)[self.tp_rank]
        param.data.copy_(local_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_world_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将非法位置的 token_id 设置为 0
            masked_x = mask * (x - self.vocab_start_idx)
        # (batch_seq_len, ) -> (batch_seq_len, embedding_dim)
        y = F.embedding(masked_x, self.weight)
        if self.tp_world_size > 1:
            # 将非法位置的词向量设置为 0，避免污染非法位置的词向量
            y_mask = mask.unsqueeze(1) * y
            dist.all_reduce(y_mask)
        return y_mask if self.tp_world_size > 1 else y

class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp: tuple[int, int] = (0, 1)
    ):
        super().__init__(num_embeddings, embedding_dim, tp)

    def forward(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
        if ctx.is_prefill:
            last_indices = ctx.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_world_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
