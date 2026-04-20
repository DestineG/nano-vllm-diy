import torch
from torch import nn
import torch.nn.functional as F

from src.utils.math import divide

class LinearBase(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp: tuple[int, int] = (0, 1)
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank, self.tp_world_size = tp

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class Linear(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp: tuple[int, int] = (0, 1)
    ):
        super().__init__(in_features, out_features, bias, tp)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        assert loaded_weight.shape == param.shape, f"Expected weight shape {param.shape}, but got {loaded_weight.shape}"
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp: tuple[int, int] = (0, 1)
    ):
        assert out_features % tp[1] == 0, "out_features must be divisible by tp_world_size"
        super().__init__(in_features, divide(out_features, tp[1]), bias, tp)
    
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        assert loaded_weight.shape[0] == self.out_features * self.tp_world_size, f"Expected weight shape ({self.out_features * self.tp_world_size}, {self.in_features}), but got {loaded_weight.shape}"
        if loaded_weight.ndim == 2:
            assert loaded_weight.shape[1] == self.in_features, f"Expected weight shape ({self.out_features * self.tp_world_size}, {self.in_features}), but got {loaded_weight.shape}"
        local_weight = loaded_weight.chunk(self.tp_world_size, 0)[self.tp_rank]
        param.data.copy_(local_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features_list: list[int],
        bias: bool = False,
        tp: tuple[int, int] = (0, 1)
    ):
        assert all(out_features % tp[1] == 0 for out_features in out_features_list), "out_features must be divisible by tp_world_size"
        super().__init__(in_features, sum(out_features_list), bias, tp)
        self.out_features_list = out_features_list
    
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int
    ):
        assert loaded_weight.shape[0] == self.out_features_list[loaded_shard_id], f"Expected weight shape ({self.out_features_list[loaded_shard_id]}, {self.in_features}), but got {loaded_weight.shape}"
        if loaded_weight.ndim == 2:
            assert loaded_weight.shape[1] == self.in_features, f"Expected weight shape ({self.out_features_list[loaded_shard_id]}, {self.in_features}), but got {loaded_weight.shape}"
        loaded_start_index = divide(sum(self.out_features_list[:loaded_shard_id]), self.tp_world_size)
        loaded_size = divide(self.out_features_list[loaded_shard_id], self.tp_world_size)
        param_data = param.data.narrow(0, loaded_start_index, loaded_size)
        local_weight = loaded_weight.chunk(self.tp_world_size, 0)[self.tp_rank]
        param_data.copy_(local_weight)

class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        model_dim: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        bias: bool = False,
        tp: tuple[int, int] = (0, 1),
    ):
        assert num_heads % tp[1] == 0, "num_heads must be divisible by tp_world_size"
        assert num_kv_heads % tp[1] == 0, "num_kv_heads must be divisible by tp_world_size"
        super().__init__(model_dim, (num_heads + 2 * num_kv_heads) * head_dim, bias, tp)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
    
    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str
    ):
        assert loaded_shard_id in ['q', 'k', 'v'], f"Expected loaded_shard_id to be one of ['q', 'k', 'v'], but got {loaded_shard_id}"
        if loaded_weight.ndim == 2:
            assert loaded_weight.shape[1] == self.in_features, f"Expected weight shape ({self.out_features}, {self.in_features}), but got {loaded_weight.shape}"
        if loaded_shard_id == "q":
            load_start_idx = 0
            expected_out_features = self.num_heads * self.head_dim
        elif loaded_shard_id == "k":
            load_start_idx = self.num_heads * self.head_dim
            expected_out_features = self.num_kv_heads * self.head_dim
        else:
            load_start_idx = (self.num_heads + self.num_kv_heads) * self.head_dim
            expected_out_features = self.num_kv_heads * self.head_dim
        assert loaded_weight.shape[0] == expected_out_features, f"Expected weight shape ({expected_out_features}, {self.in_features}), but got {loaded_weight.shape}"
        param_data = param.data.narrow(0, divide(load_start_idx, self.tp_world_size), divide(expected_out_features, self.tp_world_size))
        local_weight = loaded_weight.chunk(self.tp_world_size, 0)[self.tp_rank]
        param_data.copy_(local_weight)

class RowParallelLinear(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        tp: tuple[int, int] = (0, 1)
    ):
        assert in_features % tp[1] == 0, "in_features must be divisible by tp_world_size"
        super().__init__(divide(in_features, tp[1]), out_features, bias, tp)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        if loaded_weight.ndim == 2:
            assert loaded_weight.shape[0] == self.out_features, f"Expected weight shape ({self.out_features}, {self.in_features}), but got {loaded_weight.shape}"
            assert loaded_weight.shape[1] == self.in_features * self.tp_world_size, f"Expected weight shape ({self.out_features}, {self.in_features * self.tp_world_size}), but got {loaded_weight.shape}"
            local_weight = loaded_weight.chunk(self.tp_world_size, 1)[self.tp_rank]
            param.data.copy_(local_weight)
        else:
            assert loaded_weight.shape[0] == self.out_features, f"Expected weight shape ({self.out_features},), but got {loaded_weight.shape}"
            param.data.copy_(loaded_weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(y)
        return y